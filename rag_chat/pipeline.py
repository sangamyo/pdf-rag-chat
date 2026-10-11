"""Ingest-once cache: each PDF is processed into data/<sha256>/ and reused afterwards."""
from typing import Optional

from . import config
from .index import Index
from .ingest import Progress, file_hash, ingest_pdf


def load_or_build(pdf_bytes: bytes, doc_name: str, progress: Optional[Progress] = None) -> Index:
    progress = progress or (lambda msg, frac: None)
    directory = config.DATA_DIR / file_hash(pdf_bytes)

    cached = Index.load(directory)
    if cached is not None:
        progress("Loaded from cache", 1.0)
        return cached

    directory.mkdir(parents=True, exist_ok=True)
    chunks = ingest_pdf(pdf_bytes, doc_name, directory, progress)
    progress("Embedding chunks", 0.95)
    index = Index.build(chunks)
    index.save(directory)
    progress("Done", 1.0)
    return index
