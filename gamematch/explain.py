from __future__ import annotations

import json
import re

from .data import DIFFICULTY_LABEL, Game
from .llm import LLM
from .ranking import Ranked
from .text import norm_title, stems

SYSTEM = (
    "Ты консультант по видеоиграм. Тебе дан запрос игрока и JSON с играми из базы. "
    "Для КАЖДОЙ игры напиши 1–2 предложения по-русски: почему она подходит именно этому игроку. "
    "Используй ТОЛЬКО факты из JSON. Не упоминай другие игры, не придумывай факты и числа. "
    'Верни JSON вида {"explanations": {"<id игры>": "<текст>"}}.'
)


def matched_features(game: Game, query: str) -> list[str]:
    """Теги и жанры игры, которые пересекаются с запросом пользователя."""
    q = set(stems(query))
    found = []
    for feat in (*game.tags, *game.genres):
        if any(s in q for s in stems(feat)) and feat.lower() not in {f.lower() for f in found}:
            found.append(feat.lower())
    return found[:3]


def shared_with(game: Game, ref: Game) -> list[str]:
    """Общие теги и жанры двух игр (только то, что реально есть в базе)."""
    ref_feats = {f.lower() for f in (*ref.tags, *ref.genres)}
    out: list[str] = []
    for f in (*game.tags, *game.genres):
        if f.lower() in ref_feats and f.lower() not in out:
            out.append(f.lower())
    return out


def template_explanation(game: Game, query: str, refs: list[Game]) -> str:
    parts = []
    feats = matched_features(game, query)
    for ref in refs:
        common = shared_with(game, ref)
        if len(common) >= 2:
            parts.append(f"Похожа на {ref.title}: общее — {', '.join(common[:3])}.")
            break
    if feats:
        parts.append("Совпадает по запросу: " + ", ".join(feats) + ".")
    first = re.split(r"(?<=[.!?])\s", game.description, maxsplit=1)[0]
    parts.append(first)
    parts.append(f"Примерно {game.hours:g} ч, сложность — {DIFFICULTY_LABEL[game.difficulty]}.")
    return " ".join(parts)


def guard(text: str, game: Game, query: str, titles: dict[str, str], allowed_titles: set[str]) -> bool:
    """Проверка объяснения от LLM: нет чужих названий игр и нет чисел, которых нет в данных."""
    if not isinstance(text, str) or not (10 <= len(text) <= 600):
        return False
    low = norm_title(text)
    padded = f" {low} "
    for t_norm, gid in titles.items():
        if len(t_norm) >= 5 and gid != game.id and t_norm not in allowed_titles and f" {t_norm} " in padded:
            return False
    known = json.dumps(game.to_dict(), ensure_ascii=False) + " " + query
    for num in re.findall(r"\d+", text):
        if num not in known:
            return False
    return True


def explain(llm: LLM, ranked: list[Ranked], query: str, refs: list[Game], titles: dict[str, str]) -> tuple[dict[str, str], str]:
    """Возвращает ({id: объяснение}, источник: 'llm'|'template'|'mixed')."""
    result = {r.game.id: template_explanation(r.game, query, refs) for r in ranked}
    if not llm.available or not ranked:
        return result, "template"

    payload = {
        "query": query,
        "games": [
            {k: v for k, v in r.game.to_dict().items() if k in ("id", "title", "genres", "tags", "description", "hours", "difficulty", "coop")}
            for r in ranked
        ],
    }
    data = llm.json(SYSTEM, json.dumps(payload, ensure_ascii=False))
    expl = (data or {}).get("explanations") if isinstance(data, dict) else None
    if not isinstance(expl, dict):
        return result, "template"

    allowed = {norm_title(n) for g in refs for n in (g.title, *g.aliases)}
    used = 0
    for r in ranked:
        text = expl.get(r.game.id)
        if guard(text, r.game, query, titles, allowed):
            result[r.game.id] = text.strip()
            used += 1
    source = "llm" if used == len(ranked) else ("mixed" if used else "template")
    return result, source
