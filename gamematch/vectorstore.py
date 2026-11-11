from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VectorIndex:
    """Поиск по косинусному сходству (векторы должны быть L2-нормированы)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.dim = self.vectors.shape[1]
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.vectors)
        else:
            log.info("faiss не установлен — использую numpy-поиск")

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return scores[0], idx[0]
        sims = (self.vectors @ q[0]).astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return sims[idx], idx
