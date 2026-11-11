from __future__ import annotations

from functools import lru_cache

from .agent import GameMatchAgent


@lru_cache(maxsize=1)
def get_agent() -> GameMatchAgent:
    return GameMatchAgent()


def save_feedback(session_id: str, rating: int, comment: str = "") -> None:
    """Оценка пользователя (CSAT, цель ≥ 4/5) — дописываем в JSONL-файл."""
    import json
    import time

    from .config import get_settings

    path = get_settings().cache_dir / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": int(time.time()), "session_id": session_id, "rating": int(rating), "comment": comment[:500]}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
