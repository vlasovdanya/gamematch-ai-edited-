from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from .catalog import Catalog
from .config import Settings, get_settings
from .data import DIFFICULTY_LABEL
from .explain import explain
from .llm import LLM, NullLLM
from .profile import (
    PLATFORM_LABEL, QUESTION_TEXT, Profile, extract_rules, is_reset, split_reference, wants_more,
)
from .ranking import Ranked, apply_filters, effective_max_hours, rank
from .text import tokens

log = logging.getLogger(__name__)

UNDERSTAND_SYSTEM = (
    "Ты извлекаешь предпочтения игрока из его сообщения. Верни JSON с ключами: "
    'platform ("PC"|"PlayStation"|"Xbox"|"Switch"|"Mobile"|null), '
    'difficulty ("easy"|"medium"|"hard"|null), budget ("free"|"low"|"mid"|"high"|null), '
    "coop (true|false|null), max_hours (число|null), shorter (true|false), "
    "reference_games (список названий игр, на которые игрок хочет быть похожим, например «как Hollow Knight»). "
    "Если чего-то в сообщении нет — ставь null, ничего не выдумывай."
)
_ALLOWED = {
    "platform": {"PC", "PlayStation", "Xbox", "Switch", "Mobile"},
    "difficulty": {"easy", "medium", "hard"},
    "budget": {"free", "low", "mid", "high"},
}


class AgentState(TypedDict, total=False):
    user_text: str
    profile: Profile
    action: str            # "clarify" | "search"
    need_description: bool
    exclude_shown: bool
    candidates: list
    relaxed: list
    max_hours: float | None
    ranked: list
    explain_source: str
    reply: str
    results: list
    trace: list


@dataclass
class AgentReply:
    reply: str
    results: list[dict] = field(default_factory=list)
    needs_clarification: bool = False
    profile: dict = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    latency_ms: int = 0
    explain_source: str = "template"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class GameMatchAgent:
    def __init__(self, catalog: Catalog | None = None, llm: LLM | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings)
        if llm:
            self.llm = llm
        elif self.settings.llm_provider == "ollama":
            from .llm import OllamaLLM
            self.llm = OllamaLLM(self.settings)
        else:
            self.llm = LLM(self.settings)
        self.sessions: dict[str, Profile] = {}
        self._lock = threading.Lock()
        self.engine = "builtin"
        self._invoke = self._build_graph()

    # ------------------------------------------------------------------ граф
    def _build_graph(self):
        nodes = {
            "understand": self.n_understand, "clarify": self.n_clarify, "retrieve": self.n_retrieve,
            "rank": self.n_rank, "explain": self.n_explain,
        }
        route = lambda s: "clarify" if s["action"] == "clarify" else "retrieve"  # noqa: E731
        nxt = {"clarify": None, "retrieve": "rank", "rank": "explain", "explain": None}
        try:
            from langgraph.graph import END, StateGraph

            g = StateGraph(AgentState)
            for name, fn in nodes.items():
                g.add_node(name, fn)
            g.set_entry_point("understand")
            g.add_conditional_edges("understand", route, {"clarify": "clarify", "retrieve": "retrieve"})
            g.add_edge("clarify", END)
            g.add_edge("retrieve", "rank")
            g.add_edge("rank", "explain")
            g.add_edge("explain", END)
            self.engine = "langgraph"
            return g.compile().invoke
        except Exception as exc:
            log.info("LangGraph недоступен (%s) — использую встроенный исполнитель графа", exc)

        def run(state: dict) -> dict:
            node: str | None = "understand"
            while node:
                state = {**state, **nodes[node](state)}
                node = route(state) if node == "understand" else nxt[node]
            return state

        return run

    # ------------------------------------------------------------------ узлы
    def n_understand(self, state: AgentState) -> dict:
        text, p = state["user_text"], state["profile"]
        found = extract_rules(text)
        names, modifier_text = split_reference(text)
        if modifier_text:
            found.update(extract_rules(modifier_text))

        llm_data = self._llm_understand(text) if self.llm.available else {}
        for key, value in llm_data.items():
            if key != "reference_games" and found.get(key) is None:
                found[key] = value
        names += [n for n in llm_data.get("reference_games", []) if isinstance(n, str)]

        for name in names:
            game = self.catalog.find_by_title(name)
            if game and game.id not in p.ref_ids:
                p.ref_ids.append(game.id)

        answering = bool(p.pending)
        p.apply(found)
        for field_name in p.pending:  # на вопрос не ответили — считаем «неважно»
            if getattr(p, field_name) is None and field_name not in p.skipped:
                p.skipped.append(field_name)
        p.pending = []

        short_followup = (answering and found and len(text.split()) <= 10) or (wants_more(text) and len(text.split()) <= 4)
        if not short_followup:
            p.notes.append(text)

        words = sum(len(tokens(n)) for n in p.notes)
        substance = words >= 3 or bool(p.ref_ids)
        if not substance:
            action, need_desc = "clarify", True
        elif not p.clarified and p.missing():
            action, need_desc = "clarify", False
        else:
            action, need_desc = "search", False
        return {
            "action": action, "need_description": need_desc,
            "exclude_shown": wants_more(text) and bool(p.shown),
            "trace": [*state.get("trace", []), "understand"],
        }

    def _llm_understand(self, text: str) -> dict:
        data = self.llm.json(UNDERSTAND_SYSTEM, text) or {}
        clean: dict[str, Any] = {}
        for key, allowed in _ALLOWED.items():
            if data.get(key) in allowed:
                clean[key] = data[key]
        if isinstance(data.get("coop"), bool):
            clean["coop"] = data["coop"]
        if isinstance(data.get("max_hours"), (int, float)) and not isinstance(data.get("max_hours"), bool):
            clean["max_hours"] = float(data["max_hours"])
        if data.get("shorter") is True:
            clean["shorter"] = True
        refs = data.get("reference_games")
        if isinstance(refs, list):
            clean["reference_games"] = refs[:3]
        return clean

    def n_clarify(self, state: AgentState) -> dict:
        p = state["profile"]
        trace = [*state.get("trace", []), "clarify"]
        if state.get("need_description"):
            reply = ("Расскажите, какую игру вы ищете — своими словами. Например: «хочу атмосферную метроидванию, "
                     "как Hollow Knight, но короче» или «расслабляющую игру на вечер, чтобы играть вдвоём».")
            return {"reply": reply, "results": [], "trace": trace}
        ask = p.missing()[: self.settings.max_questions]
        p.pending, p.clarified = ask, True
        lines = ["Чтобы подобрать точнее, ответьте на несколько вопросов (на любой можно ответить «неважно»):"]
        if p.ref_ids:
            lines.insert(0, "Беру за ориентир: " + ", ".join(self.catalog.by_id[i].title for i in p.ref_ids) + ".")
        lines += [f"{i}. {QUESTION_TEXT[f]}" for i, f in enumerate(ask, 1)]
        return {"reply": "\n".join(lines), "results": [], "trace": trace}

    def n_retrieve(self, state: AgentState) -> dict:
        p = state["profile"]
        refs = [self.catalog.by_id[i] for i in p.ref_ids if i in self.catalog.by_id]
        query = p.query_text() or " ".join(g.title for g in refs)
        cands = self.catalog.search(query, refs)
        exclude = set(p.ref_ids) | (set(p.shown) if state.get("exclude_shown") else set())
        cands = [c for c in cands if c.game.id not in exclude]
        max_hours = effective_max_hours(p, self.catalog.by_id)
        filtered, relaxed = apply_filters(cands, p, max_hours, need=self.settings.top_k)
        return {"candidates": filtered, "relaxed": relaxed, "max_hours": max_hours,
                "trace": [*state.get("trace", []), "retrieve"]}

    def n_rank(self, state: AgentState) -> dict:
        ranked = rank(state["candidates"], state["profile"], self.settings.top_k)
        return {"ranked": ranked, "trace": [*state.get("trace", []), "rank"]}

    def n_explain(self, state: AgentState) -> dict:
        p, ranked = state["profile"], state["ranked"]
        trace = [*state.get("trace", []), "explain"]
        if not ranked:
            return {"reply": "Под эти условия ничего не нашлось. Попробуйте описать игру иначе или смягчить ограничения "
                             "(например, «неважно» по платформе или бюджету).", "results": [], "trace": trace,
                    "explain_source": "none"}
        refs = [self.catalog.by_id[i] for i in p.ref_ids if i in self.catalog.by_id]
        expl, source = explain(self.llm, ranked, p.query_text(), refs, self.catalog.titles)
        p.shown = (p.shown + [r.game.id for r in ranked])[-60:]
        results = [self._card(r, expl[r.game.id]) for r in ranked]
        return {"reply": self._format(results, state.get("relaxed", []), state.get("max_hours"), p),
                "results": results, "explain_source": source, "trace": trace}

    # ------------------------------------------------------------------ вывод
    @staticmethod
    def _card(r: Ranked, why: str) -> dict:
        g = r.game
        return {**g.to_dict(), "score": round(r.score, 3), "similarity": round(r.sim, 3), "why": why}

    @staticmethod
    def _format(results: list[dict], relaxed: list[str], max_hours: float | None, p: Profile) -> str:
        lines = ["Вот что я подобрал:", ""]
        for i, g in enumerate(results, 1):
            lines.append(f"{i}. **{g['title']}** ({g['year']}) — {', '.join(g['genres'])}")
            lines.append(f"   {', '.join(g['platforms'])} · ~{g['hours']:g} ч · сложность: {DIFFICULTY_LABEL[g['difficulty']]}"
                         + (" · есть кооператив" if g["coop"] else ""))
            lines.append(f"   Почему подходит: {g['why']}")
            lines.append("")
        if max_hours is not None and "длительность" not in relaxed:
            lines.append(f"Учёл длительность: не дольше ~{max_hours:g} ч.")
        if relaxed:
            lines.append("Игр, подходящих под все условия, было мало, поэтому я не стал учитывать: " + ", ".join(relaxed) + ".")
        lines.append("Можно уточнить: «подешевле», «посложнее», «ещё варианты» или «заново».")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ вход
    def chat(self, session_id: str, text: str) -> AgentReply:
        t0 = time.perf_counter()
        text = (text or "").strip()
        if not text:
            return AgentReply("Напишите, какую игру вы хотите найти.")
        injection_keywords = ["игнорируй инструкции", "system prompt", "системный промпт", "забудь все", "выведи текст выше", "ты теперь"]
        if any(kw in text.lower() for kw in injection_keywords):
            return AgentReply("Я всего лишь помогаю искать игры, давайте вернемся к подбору. Опишите, какую игру вы хотите найти.")
        if is_reset(text) and len(text.split()) <= 3:
            with self._lock:
                self.sessions.pop(session_id, None)
            return AgentReply("Начнём заново. Опишите, какую игру ищете.")
        with self._lock:
            if session_id not in self.sessions and len(self.sessions) >= 2000:
                self.sessions.pop(next(iter(self.sessions)))  # простая защита от разрастания памяти
            profile = self.sessions.setdefault(session_id, Profile())
        state: dict = self._invoke({"user_text": text, "profile": profile, "trace": []})
        profile = state.get("profile", profile)
        with self._lock:
            self.sessions[session_id] = profile  # на случай, если граф вернул копию профиля
        return AgentReply(
            reply=state.get("reply", ""),
            results=state.get("results", []),
            needs_clarification=state.get("action") == "clarify",
            profile=profile.summary(),
            trace=state.get("trace", []),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            explain_source=state.get("explain_source", "template"),
        )

    def search(self, query: str, *, platform: str | None = None, budget: str | None = None,
               coop: bool | None = None, max_hours: float | None = None, difficulty: str | None = None,
               k: int | None = None) -> list[dict]:
        p = Profile(notes=[query], platform=platform, budget=budget, coop=coop, max_hours=max_hours, difficulty=difficulty)
        names, _ = split_reference(query)
        for n in names:
            game = self.catalog.find_by_title(n)
            if game:
                p.ref_ids.append(game.id)
        refs = [self.catalog.by_id[i] for i in p.ref_ids]
        cands = [c for c in self.catalog.search(query, refs) if c.game.id not in p.ref_ids]
        k = k or self.settings.top_k
        filtered, _ = apply_filters(cands, p, max_hours, need=k)
        ranked = rank(filtered, p, k)
        expl, _ = explain(NullLLM(), ranked, query, refs, {})
        return [self._card(r, expl[r.game.id]) for r in ranked]
