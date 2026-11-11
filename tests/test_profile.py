import unittest

import tests  # noqa: F401  (включает офлайн-режим)
from gamematch.profile import extract_rules, split_reference, wants_more


class RulesTest(unittest.TestCase):
    def test_full_answer(self):
        r = extract_rules("ПК, лёгкая, до 1000 рублей, один")
        self.assertEqual((r["platform"], r["difficulty"], r["budget"], r["coop"]), ("PC", "easy", "low", False))

    def test_negated_difficulty(self):
        self.assertEqual(extract_rules("только не слишком сложную")["difficulty"], "easy")

    def test_comparatives(self):
        self.assertEqual(extract_rules("а посложнее?")["difficulty"], "hard")
        self.assertEqual(extract_rules("попроще бы")["difficulty"], "easy")
        self.assertEqual(extract_rules("а подешевле?")["budget"], "low")

    def test_free_mobile(self):
        r = extract_rules("бесплатную на телефоне")
        self.assertEqual((r["platform"], r["budget"]), ("Mobile", "free"))

    def test_coop(self):
        self.assertTrue(extract_rules("хочу играть с друзьями")["coop"])
        self.assertFalse(extract_rules("без кооператива")["coop"])

    def test_budget_numbers(self):
        self.assertEqual(extract_rules("до 2500 руб")["budget"], "mid")
        self.assertEqual(extract_rules("до 5 часов")["max_hours"], 5.0)
        self.assertNotIn("budget", extract_rules("до 5 часов"))

    def test_no_false_platform(self):
        self.assertNotIn("platform", extract_rules("хочу игру про npc и spirits"))

    def test_reference(self):
        names, mod = split_reference("Хочу атмосферное, как Hollow Knight, но короче")
        self.assertEqual(names, ["Hollow Knight"])
        self.assertEqual(mod, "короче")

    def test_more(self):
        self.assertTrue(wants_more("покажи ещё варианты"))
        self.assertFalse(wants_more("хочу стратегию"))


if __name__ == "__main__":
    unittest.main()
