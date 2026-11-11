from __future__ import annotations

import argparse
import csv
import re
import sys

DEFAULT_MAP = {
    "title": "Name", "description": "About the game", "genres": "Genres", "tags": "Tags", "released": "Release date",
    "price": "Price", "playtime": "Average playtime forever", "positive": "Positive", "negative": "Negative",
    "categories": "Categories",
}
HARD = ("difficult", "souls-like", "soulslike", "hardcore", "roguelike", "challenging")
EASY = ("relaxing", "casual", "cozy", "family friendly", "wholesome", "cute")
OUT_FIELDS = ["id", "title", "aliases", "year", "genres", "tags", "platforms", "difficulty", "price_tier",
              "hours", "coop", "rating", "description"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def num(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ".").strip())
    except ValueError:
        return default


def lst(value: str, limit: int = 12) -> str:
    items = [x.strip().replace(";", " ") for x in re.split(r"[,;]", value or "") if x.strip()]
    return ";".join(items[:limit])


def convert(row: dict, m: dict) -> dict | None:
    title = (row.get(m["title"]) or "").strip()
    desc = re.sub(r"\s+", " ", (row.get(m["description"]) or "")).strip()
    if not title:
        return None
        
    tags_raw = (row.get(m["tags"]) or "") + "," + (row.get(m["genres"]) or "")
    low = tags_raw.lower()
    difficulty = "hard" if any(w in low for w in HARD) else ("easy" if any(w in low for w in EASY) else "medium")
    price = num(row.get(m["price"]))
    tier = "free" if price <= 0 else "low" if price <= 15 else "mid" if price <= 35 else "high"
    pos, neg = num(row.get(m["positive"])), num(row.get(m["negative"]))
    year = re.search(r"(19|20)\d{2}", row.get(m["released"]) or "")
    
    if len(desc) < 40:
        desc = f"Игра в жанре {lst(row.get(m['genres'], '')).replace(';', ', ')}. Ключевые особенности: {lst(row.get(m['tags'], '')).replace(';', ', ')}."

    return {
        "id": slug(title) or slug(desc[:30]), "title": title, "aliases": "",
        "year": year.group(0) if year else "", "genres": lst(row.get(m["genres"], "")), "tags": lst(row.get(m["tags"], "")),
        "platforms": "PC", "difficulty": difficulty, "price_tier": tier,
        "hours": round(num(row.get(m["playtime"])) / 60, 1) or 10,
        "coop": int("co-op" in (row.get(m["categories"]) or "").lower()),
        "rating": round(100 * pos / (pos + neg), 1) if pos + neg else 75.0,
        "description": desc[:600],
        "_reviews": pos + neg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/games_import.csv")
    ap.add_argument("--map", default="", help="title=Колонка,description=Колонка,…")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--min-reviews", type=int, default=100)
    args = ap.parse_args()

    m = dict(DEFAULT_MAP)
    for pair in filter(None, args.map.split(",")):
        k, _, v = pair.partition("=")
        m[k.strip()] = v.strip()

    csv.field_size_limit(sys.maxsize)
    rows, seen = [], set()
    with open(args.src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [v for v in m.values() if v not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"В файле нет колонок: {missing}\nДоступные колонки: {reader.fieldnames}\nУкажите их через --map")
        for row in reader:
            g = convert(row, m)
            if g and g["_reviews"] >= args.min_reviews and g["id"] not in seen:
                seen.add(g["id"])
                rows.append(g)
    rows.sort(key=lambda g: g["_reviews"], reverse=True)
    rows = rows[: args.limit]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Сохранено игр: {len(rows)} → {args.out}")


if __name__ == "__main__":
    main()
