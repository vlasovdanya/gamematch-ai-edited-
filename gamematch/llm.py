from __future__ import annotations

import json
import logging
import re
import requests

from .config import Settings

log = logging.getLogger(__name__)


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class LLM:
    def __init__(self, settings: Settings):
        self._chat = None
        if settings.use_openai:
            try:
                from langchain_openai import ChatOpenAI

                chat = ChatOpenAI(
                    model=settings.llm_model, temperature=0, api_key=settings.openai_api_key,
                    timeout=20, max_retries=1,
                )
                try:
                    chat = chat.bind(response_format={"type": "json_object"})
                except Exception:  # pragma: no cover
                    pass
                self._chat = chat
            except Exception as exc:
                log.warning("LLM недоступен (%s), работаю по правилам", exc)

    @property
    def available(self) -> bool:
        return self._chat is not None

    def json(self, system: str, user: str) -> dict | None:
        """Вызов модели с ответом в виде JSON. При любой ошибке возвращает None."""
        if self._chat is None:
            return None
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            return parse_json(getattr(msg, "content", "") or "")
        except Exception as exc:
            log.warning("Ошибка вызова LLM: %s", exc)
            return None


class NullLLM:
    """Заглушка: ведёт себя как LLM без ключа (всегда отдаёт None)."""

    available = False

    def json(self, system: str, user: str) -> dict | None:  # noqa: A003
        return None

class OllamaLLM(LLM):
    """Класс для работы с локальной моделью через Ollama."""
    def __init__(self, settings: Settings):
        self.url = "http://localhost:11434/api/generate"
        self.model = settings.llm_model if settings.llm_model != "gpt-4o-mini" else "llama3"
        self._available = True
        if settings.force_offline:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def json(self, system: str, user: str) -> dict | None:
        if not self.available:
            return None
        prompt = f"System: {system}\nUser: {user}\nOutput JSON only:"
        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }, timeout=30)
            response.raise_for_status()
            return parse_json(response.json().get("response", ""))
        except Exception as exc:
            log.warning("Ошибка вызова Ollama: %s", exc)
            return None
