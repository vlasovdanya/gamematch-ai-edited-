from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

PRICE_ORDER = {"free": 0, "low": 1, "mid": 2, "high": 3}
PRICE_LABEL = {
    "free": "бесплатно",
    "low": "недорого (до ~1000 ₽)",
    "mid": "средняя цена (до ~3000 ₽)",
    "high": "дорого (3000 ₽ и выше)",
}
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFFICULTY_LABEL = {"easy": "лёгкая", "medium": "средняя", "hard": "высокая"}


@dataclass(frozen=True)
class Game:
    id: str
    title: str
    aliases: tuple[str, ...]
    year: int
    genres: tuple[str, ...]
    tags: tuple[str, ...]
    platforms: tuple[str, ...]
    difficulty: str
    price_tier: str
    hours: float
    coop: bool
    rating: float
    description: str

    def embedding_text(self) -> str:
        """Текст, по которому строится вектор игры."""
        return (
            f"{self.title}. Жанры: {', '.join(self.genres)}. "
            f"Теги: {', '.join(self.tags)}. {self.description}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("aliases", "genres", "tags", "platforms"):
            d[key] = list(d[key])
        return d


def _split(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in (value or "").split(";") if x.strip())


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_games(path: Path) -> list[Game]:
    """Читает CSV (колонки см. data/games.csv). Строки без id/названия пропускаются."""
    games: list[Game] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = (row.get("id") or "").strip()
            title = (row.get("title") or "").strip()
            if not gid or not title or gid in seen:
                continue
            seen.add(gid)
            difficulty = (row.get("difficulty") or "medium").strip()
            price = (row.get("price_tier") or "mid").strip()
            games.append(
                Game(
                    id=gid,
                    title=title,
                    aliases=_split(row.get("aliases", "")),
                    year=int(_to_float(row.get("year"), 0)),
                    genres=_split(row.get("genres", "")),
                    tags=_split(row.get("tags", "")),
                    platforms=_split(row.get("platforms", "")),
                    difficulty=difficulty if difficulty in DIFFICULTY_ORDER else "medium",
                    price_tier=price if price in PRICE_ORDER else "mid",
                    hours=_to_float(row.get("hours"), 10.0),
                    coop=str(row.get("coop", "0")).strip().lower() in {"1", "true", "yes"},
                    rating=_to_float(row.get("rating"), 70.0),
                    description=(row.get("description") or "").strip(),
                )
            )
    if not games:
        raise ValueError(f"В файле {path} не найдено ни одной игры")
    return games


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
