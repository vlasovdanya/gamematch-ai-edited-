import json
import unittest

import tests  # noqa: F401
from gamematch.agent import GameMatchAgent
from gamematch.catalog import Catalog
from gamematch.explain import guard
from gamematch.llm import NullLLM


class FakeLLM:
    """Поддельная модель: проверяем ветки с LLM без сети."""

    available = True

    def __init__(self, hallucinate=False):
        self.hallucinate = hallucinate
        self.calls = []

    def json(self, system, user):
        self.calls.append(system[:20])
        if system.startswith("Ты извлекаешь"):
            return {"platform": "PC", "difficulty": "bogus", "coop": None, "max_hours": None,
                    "reference_games": ["Hades"]}
        payload = json.loads(user)
        out = {}
        for i, g in enumerate(payload["games"]):
            text = f"Хорошо подходит под запрос: {g['genres'][0]} на пару вечеров."
            if self.hallucinate and i == 0:
                text = "Это лучшая игра, её сыграли 999 миллионов человек."
            if self.hallucinate and i == 1:
                text = "Похожа на Elden Ring по духу."
            out[g["id"]] = text
        return {"explanations": out}


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()
        cls.agent = GameMatchAgent(catalog=cls.catalog, llm=NullLLM())

    def test_catalog_titles(self):
        c = self.catalog
        self.assertEqual(c.find_by_title("hollow knight").id, "hollow-knight")
        self.assertEqual(c.find_by_title("Ведьмак 3").id, "the-witcher-3")
        self.assertEqual(c.find_by_title("dark souls").id, "dark-souls-3")
        self.assertIsNone(c.find_by_title("несуществующая игра xyz"))
        self.assertIsNone(c.find_by_title("в детстве"))

    def test_dialog_clarify_then_search(self):
        a, sid = self.agent, "t1"
        r1 = a.chat(sid, "Хочу атмосферную метроидванию, как Hollow Knight, но короче")
        self.assertTrue(r1.needs_clarification)
        self.assertEqual(r1.trace, ["understand", "clarify"])
        self.assertLessEqual(r1.reply.count("\n") , 4)  # заголовок + ≤3 вопроса (+ строка «ориентир»)
        r2 = a.chat(sid, "ПК, средняя, до 1000 рублей")
        self.assertFalse(r2.needs_clarification)
        self.assertEqual(r2.trace, ["understand", "retrieve", "rank", "explain"])
        self.assertEqual(len(r2.results), 5)
        for g in r2.results:
            self.assertIn("PC", g["platforms"])
            self.assertIn(g["price_tier"], {"free", "low"})
            self.assertNotEqual(g["id"], "hollow-knight")      # образец не рекомендуем
            self.assertLess(g["hours"], 30)                   # «короче» чем Hollow Knight
            self.assertIn(g["id"], self.catalog.by_id)         # только реальные игры из базы

    def test_unanswered_questions_are_skipped(self):
        a, sid = self.agent, "t2"
        a.chat(sid, "Нужен мрачный хоррор на вечер")
        r = a.chat(sid, "неважно")
        self.assertFalse(r.needs_clarification)
        self.assertTrue(r.results)

    def test_all_in_one_message_skips_questions(self):
        r = self.agent.chat("t3", "Хочу кооперативную игру с друзьями на ПК, лёгкую, любой бюджет")
        self.assertFalse(r.needs_clarification)
        self.assertEqual(len(r.results), 5)
        self.assertTrue(all(g["coop"] and "PC" in g["platforms"] for g in r.results))

    def test_greeting_asks_for_description(self):
        r = self.agent.chat("t4", "привет")
        self.assertTrue(r.needs_clarification)
        self.assertEqual(r.results, [])

    def test_more_variants_do_not_repeat(self):
        a, sid = self.agent, "t5"
        a.chat(sid, "хочу сложную игру про боссов и тёмное фэнтези")
        first = a.chat(sid, "ПК, любая, любой бюджет")
        more = a.chat(sid, "ещё варианты")
        ids1 = {g["id"] for g in first.results}
        ids2 = {g["id"] for g in more.results}
        self.assertTrue(ids1 and ids2)
        self.assertFalse(ids1 & ids2)

    def test_followup_refines(self):
        a, sid = self.agent, "t6"
        a.chat(sid, "хочу уютную игру про ферму и расслабление")
        a.chat(sid, "ПК, лёгкая, неважно")
        r = a.chat(sid, "а подешевле?")
        self.assertEqual(r.profile["budget"], "low")

    def test_reset(self):
        a = self.agent
        a.chat("t7", "хочу стратегию с пошаговыми боями")
        a.chat("t7", "заново")
        self.assertNotIn("t7", a.sessions)

    def test_impossible_filters_are_relaxed_and_reported(self):
        self.agent.chat("t8", "Хочу хоррор на Switch, бесплатный, кооператив")
        r = self.agent.chat("t8", "неважно")
        self.assertTrue(r.results)
        self.assertTrue(all("Switch" in g["platforms"] for g in r.results))  # платформу не ослабляем первой
        self.assertIn("Игр, подходящих под все условия", r.reply)

    def test_direct_search(self):
        res = self.agent.search("метроидвания в тёмном мире", platform="PC")
        self.assertEqual(len(res), 5)
        self.assertTrue(all("PC" in g["platforms"] for g in res))

    def test_every_title_in_reply_is_from_db(self):
        r = self.agent.chat("t9", "хочу космос, исследование и загадку, ПК, средняя, до 3000 рублей")
        for g in r.results:
            self.assertIn(f"**{g['title']}**", r.reply)
            self.assertEqual(self.catalog.by_id[g["id"]].title, g["title"])


class LLMBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def test_llm_understanding_is_validated_and_merged(self):
        llm = FakeLLM()
        a = GameMatchAgent(catalog=self.catalog, llm=llm)
        r = a.chat("l1", "Хочу динамичную игру с красивой графикой на вечер")
        self.assertEqual(r.profile["platform"], "PC")          # взято из ответа LLM
        self.assertIsNone(r.profile["difficulty"])             # «bogus» отброшено валидацией
        self.assertEqual(r.profile["refs"], ["hades"])         # образец найден в базе по названию

    def test_guard_replaces_hallucinated_explanations(self):
        llm = FakeLLM(hallucinate=True)
        a = GameMatchAgent(catalog=self.catalog, llm=llm)
        a.chat("l2", "Хочу динамичную игру с красивой графикой на вечер")
        r = a.chat("l2", "неважно")
        self.assertEqual(r.explain_source, "mixed")
        why = [g["why"] for g in r.results]
        self.assertFalse(any("999" in w or "Elden Ring" in w for w in why))
        self.assertTrue(any("на пару вечеров" in w for w in why))   # нормальные объяснения LLM сохранены

    def test_guard_function(self):
        g = self.catalog.by_id["hades"]
        titles = self.catalog.titles
        self.assertTrue(guard("Динамичный рогалик, на один забег уходит около часа.", g, "рогалик", titles, set()))
        self.assertFalse(guard("Лучше, чем Dark Souls III во всём.", g, "рогалик", titles, set()))
        self.assertFalse(guard("Игра идёт 500 часов.", g, "рогалик", titles, set()))


if __name__ == "__main__":
    unittest.main()
