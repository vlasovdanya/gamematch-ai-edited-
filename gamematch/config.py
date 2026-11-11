from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # .env подхватываем, если установлен python-dotenv
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_path: Path
    cache_dir: Path
    openai_api_key: str | None
    llm_model: str
    embed_model: str
    force_offline: bool
    llm_provider: str = "openai"
    top_k: int = 5            # сколько игр показываем пользователю
    candidates: int = 200     # сколько кандидатов берём из векторного поиска
    max_questions: int = 3    # максимум уточняющих вопросов за один раз

    @property
    def use_openai(self) -> bool:
        """OpenAI используется, только если есть ключ и не включён офлайн-режим."""
        return bool(self.openai_api_key) and not self.force_offline


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        data_path=Path(os.getenv("GAMEMATCH_DATA", ROOT / "data" / "games.csv")),
        cache_dir=Path(os.getenv("GAMEMATCH_CACHE", ROOT / ".cache")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_model=os.getenv("GAMEMATCH_LLM_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("GAMEMATCH_EMBED_MODEL", "text-embedding-3-small"),
        force_offline=_flag("GAMEMATCH_OFFLINE"),
        llm_provider=os.getenv("GAMEMATCH_LLM_PROVIDER", "openai"),
    )
