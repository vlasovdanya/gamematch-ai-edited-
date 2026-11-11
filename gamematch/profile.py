from __future__ import annotations

import re
from dataclasses import dataclass, field

PLATFORM_KEYWORDS: dict[str, list[str]] = {
    "PC": ["pc", "пк", "steam", "стим", "компьютер", "ноутбук", "windows"],
    "PlayStation": ["ps4", "ps5", "ps", "playstation", "плейстейшн", "плойк", "пс4", "пс5", "сони"],
    "Xbox": ["xbox", "иксбокс", "хбокс"],
    "Switch": ["switch", "свитч", "nintendo", "нинтендо"],
    "Mobile": ["мобил", "телефон", "android", "андроид", "ios", "iphone", "айфон", "смартфон", "планшет"],
}
PLATFORM_LABEL = {"PC": "ПК", "PlayStation": "PlayStation", "Xbox": "Xbox", "Switch": "Nintendo Switch", "Mobile": "телефон"}

EASY_WORDS = ["легк", "прост", "расслаб", "спокойн", "казуал", "уютн", "без стресс", "новичк", "проще", "легче", "полегче", "попроще", "не напряг"]
HARD_WORDS = ["сложн", "хардкор", "челлендж", "хард", "трудн", "испытан", "сложнее", "труднее", "посложн", "потрудн", "вызов"]
MEDIUM_WORDS = ["средн", "умеренн", "нормальн"]

FREE_WORDS = ["бесплатн", "free", "f2p", "фри ту плей", "даром"]
LOW_WORDS = ["дешев", "недорог", "бюджетн", "подешевл", "подешевле", "копейк"]
HIGH_WORDS = ["без ограничен", "неважно сколько стоит", "дорог", "любой бюджет", "деньги не проблем", "премиум"]

COOP_YES = ["кооп", "вдвоем", "вдвоём", "с друг", "с друз", "с подруг", "с приятел", "с компани", "с девушк", "с парн", "с жен", "с муж", "с сестр", "с брат",
            "мультиплеер", "co-op", "coop", "вместе", "компанией", "с семь", "с ребенк", "с ребёнк"]
COOP_NO = ["без кооп", "не нужен кооп", "не кооп", "одному", "одиночн", "соло", "сингл", "singleplayer", "single-player", "для себя", "сам по себе", "играю один", "играть один", "в одиночку"]

SHORT_WORDS = ["короче", "покороче", "короткую", "короткая", "короткое", "недолг", "на вечер", "за вечер"]
MORE_WORDS = ["ещё", "еще", "другие", "другое", "другую", "иные", "ещё варианты", "еще варианты"]
RESET_WORDS = ["заново", "сначала", "/reset", "/start", "очисти", "сброс"]

REF_RE = re.compile(
    r"(?:\bкак\b|\bтипа\b|\bвроде\b|\bпохож\w*\s+на\b|\bлайк\b|\blike\b)\s+"
    r"([A-Za-zА-Яа-яЁё0-9'’:!\-\. ]{2,60}?)"
    r"(?:\s*,?\s+(?:но|только|зато|but)\s+(.{2,80}?))?\s*(?:[.!?,;]|$)",
    re.I,
)


def _has(text: str, words: list[str]) -> bool:
    for w in words:
        if len(w) <= 3 or w in ("ps", "pc", "ios"):
            if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text):
                return True
        elif re.search(rf"(?<!\w){re.escape(w)}", text):
            return True
    return False


def _budget_from_number(text: str) -> str | None:
    """«до 1000 рублей», «до 20$» → уровень цены."""
    m = re.search(r"(?:до|не больше|максимум|в пределах|around|under)\s*(\d[\d\s]{0,7})\s*(₽|руб|р\b|rub|\$|долл|usd|евро|€)?", text)
    if not m or not m.group(2):
        return None
    amount = int(re.sub(r"\s", "", m.group(1)) or 0)
    if m.group(2) in ("$", "долл", "usd", "евро", "€"):
        amount *= 100
    if amount <= 0:
        return "free"
    if amount <= 1200:
        return "low"
    if amount <= 3000:
        return "mid"
    return "high"


def extract_rules(text: str) -> dict:
    """Достаёт из текста платформу, сложность, бюджет, кооп, ограничение по часам."""
    t = text.lower().replace("ё", "е")
    out: dict = {}

    for platform, words in PLATFORM_KEYWORDS.items():
        if _has(t, words):
            out["platform"] = platform
            break

    if re.search(r"(?<!\w)(не|без)\s+(слишком\s+|очень\s+|особо\s+)?(сложн|хард|напряг)", t):
        out["difficulty"] = "easy"
    elif _has(t, EASY_WORDS):
        out["difficulty"] = "easy"
    elif _has(t, HARD_WORDS):
        out["difficulty"] = "hard"
    elif _has(t, MEDIUM_WORDS):
        out["difficulty"] = "medium"

    budget = _budget_from_number(t)
    if budget:
        out["budget"] = budget
    elif _has(t, FREE_WORDS):
        out["budget"] = "free"
    elif _has(t, LOW_WORDS):
        out["budget"] = "low"
    elif _has(t, HIGH_WORDS):
        out["budget"] = "high"

    if _has(t, COOP_NO):
        out["coop"] = False
    elif _has(t, COOP_YES):
        out["coop"] = True
    elif re.search(r"(?<!\w)один(?!\w)", t):
        out["coop"] = False

    m = re.search(r"до\s*(\d{1,3})\s*(?:час|ч\b)", t)
    if m:
        out["max_hours"] = float(m.group(1))
    if _has(t, SHORT_WORDS):
        out["shorter"] = True
    return out


def split_reference(text: str) -> tuple[list[str], str]:
    """«Хочу атмосферное, как Hollow Knight, но короче» → (["Hollow Knight"], "короче")."""
    names: list[str] = []
    modifiers: list[str] = []
    for m in REF_RE.finditer(text):
        names.append(m.group(1).strip())
        if m.group(2):
            modifiers.append(m.group(2).strip())
    return names, " ".join(modifiers)


def wants_more(text: str) -> bool:
    return _has(text.lower(), MORE_WORDS)


def is_reset(text: str) -> bool:
    return _has(text.lower(), RESET_WORDS)


QUESTION_ORDER = ["platform", "difficulty", "budget", "coop"]
QUESTION_TEXT = {
    "platform": "На какой платформе будете играть (ПК, PlayStation, Xbox, Switch, телефон)?",
    "difficulty": "Какая сложность вам комфортна: расслабленная, средняя или хардкорная?",
    "budget": "Какой бюджет: бесплатно, до ~1000 ₽, до ~3000 ₽ или без ограничений?",
    "coop": "Нужен кооператив или будете играть в одиночку?",
}


@dataclass
class Profile:
    notes: list[str] = field(default_factory=list)     # свободные описания пользователя
    platform: str | None = None
    difficulty: str | None = None
    budget: str | None = None
    coop: bool | None = None
    max_hours: float | None = None
    shorter: bool = False
    ref_ids: list[str] = field(default_factory=list)   # игры-образцы («как X»)
    pending: list[str] = field(default_factory=list)   # поля, о которых только что спросили
    skipped: list[str] = field(default_factory=list)   # поля, на которые пользователь не ответил
    shown: list[str] = field(default_factory=list)     # уже показанные игры
    clarified: bool = False                            # уточняющий раунд уже был

    def apply(self, data: dict) -> None:
        for key in ("platform", "difficulty", "budget", "coop", "max_hours"):
            if data.get(key) is not None:
                setattr(self, key, data[key])
        if data.get("shorter"):
            self.shorter = True

    def missing(self) -> list[str]:
        return [f for f in QUESTION_ORDER if getattr(self, f) is None and f not in self.skipped]

    def query_text(self) -> str:
        return " ".join(self.notes)[-700:]

    def summary(self) -> dict:
        return {
            "platform": self.platform, "difficulty": self.difficulty, "budget": self.budget,
            "coop": self.coop, "max_hours": self.max_hours, "refs": list(self.ref_ids),
        }
