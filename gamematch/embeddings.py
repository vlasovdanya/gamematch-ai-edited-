from __future__ import annotations

import logging
import zlib
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from .config import Settings
from .text import stem, tokens

log = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class OpenAIEmbedder(Embedder):
    """Эмбеддинги OpenAI через langchain-openai."""

    def __init__(self, settings: Settings):
        from langchain_openai import OpenAIEmbeddings

        self.name = f"openai-{settings.embed_model}"
        self._emb = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            out.extend(self._emb.embed_documents(texts[i : i + 256]))
        return _normalize(np.array(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(self._emb.embed_query(text), dtype=np.float32))


class LocalEmbedder(Embedder):
    """Лёгкий TF-IDF с хешированием (только numpy). Работает без интернета и ключей.
    Качество ниже, чем у OpenAI, но для разработки, тестов и демо без ключа достаточно."""

    name = "local-tfidf"

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    def _features(self, text: str) -> Counter:
        toks = [stem(t) for t in tokens(text)]
        feats: Counter = Counter()
        for t in toks:
            feats[t] += 1
        for a, b in zip(toks, toks[1:]):
            feats[f"{a}_{b}"] += 1
        return feats

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) % self.dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in self._features(text).items():
            v[self._bucket(f)] += 1.0 + np.log(c)
        return v * self.idf

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            for b in {self._bucket(f) for f in self._features(text)}:
                df[b] += 1
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1.0).astype(np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))


def get_embedder(settings: Settings) -> Embedder:
    if settings.use_openai:
        try:
            return OpenAIEmbedder(settings)
        except Exception as exc:  # нет пакета, неверный ключ и т.п.
            log.warning("OpenAI-эмбеддинги недоступны (%s), включаю локальный режим", exc)
    return LocalEmbedder()
