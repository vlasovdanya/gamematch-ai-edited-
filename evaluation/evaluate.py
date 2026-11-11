from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gamematch.agent import GameMatchAgent  # noqa: E402
from gamematch.text import norm_title  # noqa: E402

TARGETS = {"precision@5": 0.85, "faithfulness": 0.90, "latency_p95_s": 5.0}


def is_relevant(game: dict, tags: list[str]) -> bool:
    feats = {x.lower() for x in (*game["tags"], *game["genres"])}
    return any(t.lower() in feats for t in tags)


def faithful(agent: GameMatchAgent, reply, ref_ids: list[str]) -> float:
    if not reply.results:
        return 1.0
    allowed_refs = {t for t, gid in agent.catalog.titles.items() if gid in ref_ids}
    ok = 0
    for g in reply.results:
        db = agent.catalog.by_id.get(g["id"])
        if not db or db.title != g["title"] or f"**{g['title']}**" not in reply.reply:
            continue
        why = f" {norm_title(g['why'])} "
        record = f" {norm_title(db.description)} "
        foreign = [t for t, gid in agent.catalog.titles.items()
                   if gid != g["id"] and len(t) >= 6 and t not in allowed_refs
                   and f" {t} " in why and f" {t} " not in record]
        ok += 0 if foreign else 1
    return ok / len(reply.results)


def run(testset_path: Path) -> dict:
    agent = GameMatchAgent()
    cases = json.loads(testset_path.read_text(encoding="utf-8"))
    rows, latencies = [], []
    for case in cases:
        sid = f"eval-{case['id']}"
        reply = None
        for msg in case["messages"]:
            reply = agent.chat(sid, msg)
        assert reply is not None
        latencies.append(reply.latency_ms / 1000)
        rel = [is_relevant(g, case["relevant_tags"]) for g in reply.results]
        rows.append({
            "id": case["id"], "scenario": case["scenario"],
            "precision": sum(rel) / len(rel) if rel else 0.0,
            "faithfulness": faithful(agent, reply, agent.sessions[sid].ref_ids),
            "latency_s": reply.latency_ms / 1000,
            "titles": [g["title"] for g in reply.results],
        })
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    summary = {
        "mode": {"engine": agent.engine, "llm": agent.llm.available, "embedder": agent.catalog.embedder.name,
                 "index": agent.catalog.index.backend, "games": len(agent.catalog.games)},
        "precision@5": round(statistics.mean(r["precision"] for r in rows), 3),
        "faithfulness": round(statistics.mean(r["faithfulness"] for r in rows), 3),
        "latency_p95_s": round(p95, 3),
        "cases": rows,
    }
    summary["passed"] = {
        "precision@5": summary["precision@5"] >= TARGETS["precision@5"],
        "faithfulness": summary["faithfulness"] >= TARGETS["faithfulness"],
        "latency_p95_s": summary["latency_p95_s"] <= TARGETS["latency_p95_s"],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(ROOT / "evaluation" / "testset.json"))
    ap.add_argument("--json", default=None, help="куда сохранить отчёт")
    args = ap.parse_args()
    s = run(Path(args.testset))
    print(f"Режим: {s['mode']}")
    print(f"{'тест':<16}{'сценарий':<16}{'P@5':>6}{'faith':>7}{'сек':>7}")
    for r in s["cases"]:
        print(f"{r['id']:<16}{r['scenario']:<16}{r['precision']:>6.2f}{r['faithfulness']:>7.2f}{r['latency_s']:>7.3f}")
    print("-" * 52)
    for key, target in TARGETS.items():
        sign = "≤" if key.startswith("latency") else "≥"
        mark = "✓" if s["passed"][key] else "✗"
        print(f"{mark} {key:<15} = {s[key]:<7} (цель {sign} {target})")
    if args.json:
        Path(args.json).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
