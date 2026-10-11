"""Answer questions over retrieved chunks, attaching table/figure images so Claude
reads charts and tables directly rather than relying only on extracted text."""
from pathlib import Path
from typing import Iterator

from . import config, llm
from .index import Index
from .ingest import Chunk

SYSTEM_PROMPT = """You answer questions about the user's PDF documents using only the retrieved excerpts provided in each message. Excerpts can be body text, tables (as Markdown), or figures (a text description of a chart or image). Some table and figure excerpts are followed by their image; when one is, treat the image as the primary source and read values from it directly, using the text only as support.

Rules:
- Base every claim on the excerpts. If they don't contain the answer, say what is missing rather than guessing; you may suggest what the user could ask instead.
- Cite sources inline as [p. N] (or [document, p. N] when several documents are loaded), using the page numbers in the excerpt headers.
- When reading numbers off a chart, say they are approximate if the chart has no data labels.
- Use Markdown tables when comparing several values. Keep answers concise."""


def _source_header(n: int, chunk: Chunk) -> str:
    label = chunk.kind if not chunk.subtype else f"{chunk.kind} ({chunk.subtype})"
    return f'<excerpt id="{n}" document="{chunk.doc}" page="{chunk.page}" type="{label}">'


def build_parts(question: str, hits: list[tuple[Chunk, float]]) -> list[llm.Part]:
    # Figures need the image most; tables are already readable as Markdown.
    with_image = [c for c, _ in hits if c.image and Path(c.image).exists()]
    with_image.sort(key=lambda c: c.kind != "figure")
    attach = {c.id for c in with_image[:config.MAX_IMAGES_PER_ANSWER]}

    parts: list[llm.Part] = []
    for n, (chunk, _) in enumerate(hits, start=1):
        parts.append(("text", f"{_source_header(n, chunk)}\n{chunk.content}\n</excerpt>"))
        if chunk.id in attach:
            parts.append(("image", Path(chunk.image).read_bytes()))
    parts.append(("text", f"Question: {question}"))
    return parts


def retrieval_query(question: str, history: list[dict]) -> str:
    """Fold in the previous question so short follow-ups ("and in 2022?") still retrieve."""
    previous = [m["content"] for m in history if m["role"] == "user"]
    return f"{previous[-1]}\n{question}" if previous else question


def answer(question: str, index: Index, history: list[dict], k: int = config.TOP_K
           ) -> tuple[list[tuple[Chunk, float]], Iterator[str]]:
    """Return (retrieved hits, stream of answer text).

    `history` is prior turns as plain {"role", "content": str} dicts. Earlier turns are
    replayed as text only (no retrieved excerpts) to keep the request small; only
    the current turn carries the excerpts and images.
    """
    hits = index.search(retrieval_query(question, history), k=k)
    recent = history[-config.HISTORY_MESSAGES:] if config.HISTORY_MESSAGES else []
    return hits, llm.stream(SYSTEM_PROMPT, recent, build_parts(question, hits))
