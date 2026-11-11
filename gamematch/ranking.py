from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Candidate
from .data import DIFFICULTY_ORDER, PRICE_ORDER
from .profile import Profile
from .text import stems

# Гибридный скор: смысловая близость (эмбеддинги) + совпадение с тегами/жанрами + качество + сложность.
W_SIM, W_TAGS, W_QUALITY, W_PREF = 0.55, 0.20, 0.10, 0.15


@dataclass
class Ranked:
    game: object
    score: float
    sim: float
    reasons: list[str] = field(default_factory=list)


def _passes(c: Candidate, p: Profile, max_hours: float | None, active: set[str]) -> bool:
    g = c.game
    if "platform" in active and p.platform and p.platform not in g.platforms:
        return False
    if "budget" in active and p.budget and PRICE_ORDER[g.price_tier] > PRICE_ORDER[p.budget]:
        return False
    if "coop" in active and p.coop is True and not g.coop:
        return False
    if "hours" in active and max_hours is not None and g.hours > max_hours:
        return False
    return True


def effective_max_hours(p: Profile, catalog_games: dict) -> float | None:
    """«Как X, но короче» → лимит = 75% от длительности X."""
    if p.max_hours is not None:
        return p.max_hours
    if p.shorter:
        refs = [catalog_games[i].hours for i in p.ref_ids if i in catalog_games]
        return round(min(refs) * 0.75, 1) if refs else 10.0
    return None


def apply_filters(cands: list[Candidate], p: Profile, max_hours: float | None, need: int) -> tuple[list[Candidate], list[str]]:
    """Жёсткие фильтры. Если подходящих меньше `need`, ослабляем ограничения по очереди
    (часы → кооп → бюджет → платформа) и сообщаем, что именно ослабили."""
    active = {"platform", "budget", "coop", "hours"}
    relaxed: list[str] = []
    result = [c for c in cands if _passes(c, p, max_hours, active)]
    for name, label in (("hours", "длительность"), ("coop", "кооператив"), ("budget", "бюджет"), ("platform", "платформа")):
        if len(result) >= need:
            break
        is_set = {"hours": max_hours is not None, "coop": p.coop is True, "budget": p.budget is not None, "platform": p.platform is not None}[name]
        if not is_set:
            continue
        active.discard(name)
        relaxed.append(label)
        result = [c for c in cands if _passes(c, p, max_hours, active)]
    return result, relaxed


def _pref_score(game, p: Profile) -> float:
    if p.difficulty is None:
        return 0.5
    gap = abs(DIFFICULTY_ORDER[game.difficulty] - DIFFICULTY_ORDER[p.difficulty])
    return {0: 1.0, 1: 0.4}.get(gap, 0.0)


def tag_overlap(game, query_stems: set[str]) -> float:
    """Сколько тегов/жанров игры упомянуто в запросе (0…1): две совпавшие характеристики — максимум."""
    hit = sum(1 for feat in {*game.tags, *game.genres} if any(s in query_stems for s in stems(feat)))
    return min(1.0, hit / 2)


def rank(cands: list[Candidate], p: Profile, k: int = 5) -> list[Ranked]:
    if not cands:
        return []
    q_stems = set(stems(p.query_text()))
    sims = [c.sim for c in cands]
    lo, hi = min(sims), max(sims)
    span = (hi - lo) or 1.0
    ranked = []
    for c in cands:
        sim_n = (c.sim - lo) / span
        quality = max(0.0, min(1.0, c.game.rating / 100))
        pref = _pref_score(c.game, p)
        tags = tag_overlap(c.game, q_stems)
        score = W_SIM * sim_n + W_TAGS * tags + W_QUALITY * quality + W_PREF * pref
        ranked.append(Ranked(c.game, score, c.sim))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:k]
