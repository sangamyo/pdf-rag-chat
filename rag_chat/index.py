"""Hybrid retrieval: local dense embeddings + BM25, fused with reciprocal rank fusion.

BM25 catches exact terms (names, figures, table labels) that small embedding models
blur; dense vectors catch paraphrases.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk, load_chunks, save_chunks

RRF_K = 60
CANDIDATES = 30


@lru_cache(maxsize=1)
def embedder():
    # ONNX runtime via fastembed: same bge model as sentence-transformers, no PyTorch.
    from fastembed import TextEmbedding
    return TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(embedder().embed(texts, batch_size=32)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _embed_text(chunk: Chunk) -> str:
    return f"[{chunk.kind}{'/' + chunk.subtype if chunk.subtype else ''}, page {chunk.page}] {chunk.content}"


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25Okapi([_tokens(c.content) for c in chunks]) if chunks else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Index":
        vectors = _embed([_embed_text(c) for c in chunks]) if chunks else np.zeros((0, 1))
        return cls(chunks, vectors)

    @classmethod
    def merge(cls, indexes: list["Index"]) -> "Index":
        chunks = [c for ix in indexes for c in ix.chunks]
        vectors = np.vstack([ix.vectors for ix in indexes if len(ix.chunks)]) if chunks else np.zeros((0, 1))
        return cls(chunks, vectors)

    def save(self, directory: Path) -> None:
        save_chunks(self.chunks, directory / "chunks.json")
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "meta.json").write_text(json.dumps({"embed_model": config.EMBED_MODEL}))

    @classmethod
    def load(cls, directory: Path) -> "Index | None":
        try:
            meta = json.loads((directory / "meta.json").read_text())
            chunks = load_chunks(directory / "chunks.json")
        except FileNotFoundError:
            return None
        if meta.get("embed_model") != config.EMBED_MODEL:
            # Extraction (and paid figure descriptions) are reusable; only re-embed.
            return cls.build(chunks)
        return cls(chunks, np.load(directory / "vectors.npy"))

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        n = min(CANDIDATES, len(self.chunks))

        q = _embed([config.EMBED_QUERY_PREFIX + query])[0]
        dense_rank = np.argsort(-(self.vectors @ q))[:n]
        bm25_rank = np.argsort(-self.bm25.get_scores(_tokens(query)))[:n]

        scores: dict[int, float] = {}
        for ranking in (dense_rank, bm25_rank):
            for rank, idx in enumerate(ranking):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        best = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], s) for i, s in best]
