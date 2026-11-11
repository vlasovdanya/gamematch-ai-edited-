import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gamematch.data import load_games

def generate():
    games = load_games(ROOT / "data" / "games.csv")
    scenarios = ["новичок", "опытный игрок", "как X, но Y", "кооператив", "жанр"]
    testset = []

    for i in range(100):
        g = random.choice(games)
        scen = random.choice(scenarios)
        
        if scen == "как X, но Y":
            mod = random.choice(["короче", "дешевле", "сложнее", "проще", "с кооперативом"])
            msg1 = f"Хочу что-то атмосферное, как {g.title}, но {mod}"
            msg2 = "ПК, средняя, любой бюджет"
            tags = list(g.tags)[:3]
        elif scen == "кооператив":
            msg1 = "Хочу игру для двоих, чтобы играть вместе"
            msg2 = "ПК, лёгкая, любой бюджет"
            tags = ["кооператив", "для двоих"]
        else:
            tags = list(g.tags)[:3]
            msg1 = f"Хочу игру с тегами: {', '.join(tags)}"
            msg2 = "ПК, средняя, любой бюджет"

        testset.append({
            "id": f"auto-{i}", "scenario": scen,
            "messages": [msg1, msg2], "relevant_tags": tags
        })

    out = ROOT / "evaluation" / "testset_100.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сгенерировано 100 тестов в {out}")

if __name__ == "__main__":
    generate()
