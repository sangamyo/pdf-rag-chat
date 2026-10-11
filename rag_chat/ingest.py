"""Turn a PDF into retrievable chunks of three kinds:

- text:   page text split into overlapping windows (table/figure regions excluded)
- table:  tables detected by PyMuPDF, stored as Markdown plus a cropped snapshot
- figure: raster images AND vector-drawn charts, cropped to PNG and described by Claude
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import pymupdf

from . import config, vision

Progress = Callable[[str, float], None]


@dataclass
class Chunk:
    id: str
    doc: str              # source file name
    page: int             # 1-indexed
    kind: str             # "text" | "table" | "figure"
    content: str          # text that is embedded and shown to Claude
    image: Optional[str] = None   # path to a PNG snapshot (tables and figures)
    subtype: Optional[str] = None  # figure type reported by Claude (chart, diagram, ...)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# ---------------------------------------------------------------- geometry

def _overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Fraction of `a`'s area that lies inside `b`."""
    inter = pymupdf.Rect(a) & b
    return 0.0 if inter.is_empty or a.is_empty else inter.get_area() / a.get_area()


def _merge_rects(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    rects = [pymupdf.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        out: list[pymupdf.Rect] = []
        for r in rects:
            for i, o in enumerate(out):
                if r.intersects(o):
                    out[i] = o | r
                    merged = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def _big_enough(r: pymupdf.Rect) -> bool:
    return r.width >= config.MIN_FIGURE_PT and r.height >= config.MIN_FIGURE_PT


def _render(page: pymupdf.Page, clip: pymupdf.Rect, path: Path) -> bytes:
    zoom = min(2.5, config.MAX_IMAGE_EDGE_PX / max(clip.width, clip.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
    png = pix.tobytes("png")
    path.write_bytes(png)
    return png


# ---------------------------------------------------------------- detection

def _find_figure_regions(page: pymupdf.Page, table_rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    page_rect = page.rect
    candidates: list[pymupdf.Rect] = []

    # Raster images as placed on the page.
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page_rect
        if _big_enough(r):
            candidates.append(r)

    # Vector graphics (most charts in generated PDFs are drawn, not embedded images).
    try:
        for r in page.cluster_drawings():
            r = pymupdf.Rect(r) & page_rect
            if _big_enough(r) and r.get_area() < 0.9 * page_rect.get_area():
                candidates.append(r)
    except Exception:
        pass

    # Drop anything that is really a detected table (tables are drawn with lines too).
    candidates = [r for r in candidates if not any(_overlap_ratio(r, t) > 0.6 for t in table_rects)]

    regions = _merge_rects(candidates)
    labels = [
        pymupdf.Rect(b[:4]) for b in page.get_text("blocks")
        if b[6] == 0 and len(b[4].strip()) <= LABEL_MAX_CHARS
    ]
    regions = _merge_rects([_absorb_labels(r, labels) for r in regions])
    return [(r + (-6, -6, 6, 6)) & page_rect for r in regions if _big_enough(r)]


LABEL_MAX_CHARS = 80   # tick labels, axis titles, legends, captions
LABEL_GAP_PT = 36


def _gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return max(dx, dy)


def _absorb_labels(region: pymupdf.Rect, labels: list[pymupdf.Rect]) -> pymupdf.Rect:
    """Grow a drawing's bounding box to take in short text blocks next to it, so tick
    labels and axis titles (which clustering of vector shapes misses) are in the crop."""
    region = pymupdf.Rect(region)
    for _ in range(3):
        grown = region
        for lab in labels:
            if _gap(region, lab) <= LABEL_GAP_PT:
                grown = grown | lab
        if grown == region:
            break
        region = grown
    return region


MIN_TABLE_FILL = 0.4


def _fill_ratio(table) -> float:
    cells = [c for row in table.extract() for c in row]
    return sum(1 for c in cells if c and c.strip()) / max(len(cells), 1)


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= config.CHUNK_CHARS:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + config.CHUNK_CHARS, len(text))
        if end < len(text):
            # Prefer to break on a paragraph, then sentence, then word boundary.
            for sep in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(sep, start + config.CHUNK_CHARS // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - config.CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


# ---------------------------------------------------------------- main entry

def ingest_pdf(pdf_bytes: bytes, doc_name: str, out_dir: Path, progress: Optional[Progress] = None) -> list[Chunk]:
    """Extract chunks from a PDF, writing figure/table snapshots into out_dir/images."""
    progress = progress or (lambda msg, frac: None)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[Chunk] = []
    figure_jobs: list[tuple[str, int, bytes, str, Path]] = []

    for pno, page in enumerate(doc, start=1):
        progress(f"Reading page {pno}/{doc.page_count}", 0.4 * pno / doc.page_count)
        page_text = page.get_text("text")

        # --- tables
        table_rects: list[pymupdf.Rect] = []
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        for ti, table in enumerate(tables):
            md = table.to_markdown(clean=True).strip()
            if not md or table.row_count < 2 or _fill_ratio(table) < MIN_TABLE_FILL:
                continue  # chart gridlines also form "tables", but with nearly empty cells
            rect = pymupdf.Rect(table.bbox)
            table_rects.append(rect)
            img_path = img_dir / f"p{pno}_table{ti}.png"
            _render(page, (rect + (-6, -6, 6, 6)) & page.rect, img_path)
            chunks.append(Chunk(
                id=f"p{pno}-table{ti}", doc=doc_name, page=pno, kind="table",
                content=f"Table from page {pno}:\n{md}", image=str(img_path),
            ))

        # --- figures (raster + vector)
        figure_rects = _find_figure_regions(page, table_rects)
        for fi, rect in enumerate(figure_rects):
            img_path = img_dir / f"p{pno}_fig{fi}.png"
            png = _render(page, rect, img_path)
            figure_jobs.append((f"p{pno}-fig{fi}", pno, png, page_text, img_path))

        # --- body text, skipping blocks that belong to a table or figure
        excluded = table_rects + figure_rects
        blocks = [
            b[4] for b in page.get_text("blocks", sort=True)
            if b[6] == 0 and not any(_overlap_ratio(pymupdf.Rect(b[:4]), r) > 0.5 for r in excluded)
        ]
        for ci, piece in enumerate(_split_text("\n".join(blocks))):
            chunks.append(Chunk(id=f"p{pno}-text{ci}", doc=doc_name, page=pno, kind="text", content=piece))

    # --- describe figures with Claude vision, in parallel
    total = len(figure_jobs)
    if total:
        done = 0

        def describe(job):
            cid, pno, png, page_text, img_path = job
            fig_type, desc = vision.describe_figure(png, pno, page_text)
            return Chunk(id=cid, doc=doc_name, page=pno, kind="figure", content=desc,
                         image=str(img_path), subtype=fig_type)

        with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as pool:
            for chunk in pool.map(describe, figure_jobs):
                chunks.append(chunk)
                done += 1
                progress(f"Described figure {done}/{total}", 0.4 + 0.5 * done / total)

    chunks.sort(key=lambda c: (c.page, ["text", "table", "figure"].index(c.kind), c.id))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks], indent=1))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**c) for c in json.loads(path.read_text())]
