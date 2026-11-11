from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass

import numpy as np

from .config import Settings, get_settings
from .data import Game, file_hash, load_games
from .embeddings import Embedder, LocalEmbedder, get_embedder
from .text import norm_title
from .vectorstore import VectorIndex

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    game: Game
    sim: float


class Catalog:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.games: list[Game] = load_games(self.settings.data_path)
        self.by_id = {g.id: g for g in self.games}
        self.embedder: Embedder = get_embedder(self.settings)
        self.index = VectorIndex(self._load_or_embed())
        self.titles = self._build_title_map()
        log.info("Каталог: %d игр, эмбеддер=%s, индекс=%s", len(self.games), self.embedder.name, self.index.backend)

    # ---------- индекс ----------
    def _load_or_embed(self) -> np.ndarray:
        texts = [g.embedding_text() for g in self.games]
        if isinstance(self.embedder, LocalEmbedder):  # локальный считается мгновенно
            return self.embedder.embed_documents(texts)
        cache = self.settings.cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        key = f"{self.embedder.name}-{file_hash(self.settings.data_path)}".replace("/", "_")
        path = cache / f"emb-{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.embedder.embed_documents(texts)
        np.save(path, vectors)
        return vectors

    # ---------- поиск по названию ----------
    def _build_title_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for g in self.games:
            for name in (g.title, *g.aliases):
                mapping.setdefault(norm_title(name), g.id)
        return mapping

    def find_by_title(self, name: str) -> Game | None:
        """Ищет игру по названию или алиасу: точное совпадение, вхождение, затем нечёткое."""
        q = norm_title(name)
        if len(q) < 3:
            return None
        if q in self.titles:
            return self.by_id[self.titles[q]]
        # «dark souls» → «dark souls iii»; но не «the»
        contained = [t for t in self.titles if len(q) >= 4 and (q in t.split() or f" {q} " in f" {t} ")]
        if contained:
            return self.by_id[self.titles[min(contained, key=len)]]
        close = difflib.get_close_matches(q, list(self.titles), n=1, cutoff=0.82)
        return self.by_id[self.titles[close[0]]] if close else None

    # ---------- семантический поиск ----------
    def search(self, query: str, refs: list[Game] | None = None, k: int | None = None) -> list[Candidate]:
        """Векторный поиск. Если указаны игры-образцы («как Hollow Knight»), их векторы подмешиваются в запрос."""
        k = k or self.settings.candidates
        q = self.embedder.embed_query(query)
        if refs:
            pos = {g.id: i for i, g in enumerate(self.games)}
            ref_vecs = np.stack([self.index.vectors[pos[g.id]] for g in refs])
            q = q + 0.7 * ref_vecs.mean(axis=0)
            q = q / (np.linalg.norm(q) or 1.0)
        scores, idx = self.index.search(q, k)
        return [Candidate(self.games[int(i)], float(s)) for s, i in zip(scores, idx) if i >= 0]
