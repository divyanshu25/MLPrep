#!/usr/bin/env python3
"""Spaced-repetition queue over practice/queue.tsv. Pure stdlib, no deps.

Usage:
    uv run python practice/review.py due                    # what to re-attempt today
    uv run python practice/review.py add <item_id> <path>   # new item, due tomorrow
    uv run python practice/review.py pass <item_id>         # interval doubles (cap 16)
    uv run python practice/review.py fail <item_id>         # interval resets to 1
    uv run python practice/review.py stats                  # mastery histogram

Why these mechanics: passing an item doubles its interval (1->2->4->8->16 days)
because each successful effortful retrieval raises storage strength, buying a
longer gap before the next one is needed; failing resets to 1 because a miss
means retrieval strength collapsed and the item needs the short-gap ladder again.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

QUEUE = Path(__file__).resolve().parent / "queue.tsv"
COLS = ["item_id", "source_path", "added", "last_result", "interval_days", "due"]
CAP = 16


def load():
    if not QUEUE.exists():
        QUEUE.write_text("\t".join(COLS) + "\n")
    lines = QUEUE.read_text().splitlines()[1:]
    return [dict(zip(COLS, ln.split("\t"))) for ln in lines if ln.strip()]


def save(rows):
    out = ["\t".join(COLS)] + ["\t".join(r[c] for c in COLS) for r in rows]
    QUEUE.write_text("\n".join(out) + "\n")


def find(rows, item_id):
    for r in rows:
        if r["item_id"] == item_id:
            return r
    sys.exit(f"error: no item '{item_id}' in {QUEUE.name} (see 'due' for ids)")


def cmd_due(rows):
    today = date.today().isoformat()
    hits = sorted((r for r in rows if r["due"] <= today),
                  key=lambda r: (r["due"], r["item_id"]))
    if not hits:
        print("Nothing due. Do a mixed set or a drill instead.")
        return
    for r in hits:
        print(f"{r['due']}  {r['item_id']:<26} {r['interval_days']:>2}d  {r['source_path']}")
    print(f"-- {len(hits)} due: attempt cold, then 'pass'/'fail' each one --")


def cmd_add(rows, item_id, source_path):
    if any(r["item_id"] == item_id for r in rows):
        sys.exit(f"error: '{item_id}' already queued")
    today = date.today()
    rows.append({"item_id": item_id, "source_path": source_path,
                 "added": today.isoformat(), "last_result": "new",
                 "interval_days": "1",
                 "due": (today + timedelta(days=1)).isoformat()})
    save(rows)
    print(f"added {item_id}, due tomorrow")


def cmd_mark(rows, item_id, result):
    r = find(rows, item_id)
    iv = min(int(r["interval_days"]) * 2, CAP) if result == "pass" else 1
    r.update(last_result=result, interval_days=str(iv),
             due=(date.today() + timedelta(days=iv)).isoformat())
    save(rows)
    print(f"{item_id}: {result}, next due {r['due']} (interval {iv}d)")


def cmd_stats(rows):
    buckets = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for r in rows:
        iv = int(r["interval_days"])
        buckets[min((b for b in buckets if iv <= b), default=CAP)] += 1
    print(f"{len(rows)} items by interval (right = closer to mastered):")
    for b, n in sorted(buckets.items()):
        print(f"  <={b:>2}d  {'#' * n} {n}")


def main():
    args = sys.argv[1:]
    rows = load()
    if args[:1] == ["due"]:
        cmd_due(rows)
    elif args[:1] == ["add"] and len(args) == 3:
        cmd_add(rows, args[1], args[2])
    elif args[:1] in (["pass"], ["fail"]) and len(args) == 2:
        cmd_mark(rows, args[1], args[0])
    elif args[:1] == ["stats"]:
        cmd_stats(rows)
    else:
        sys.exit(__doc__.strip())


if __name__ == "__main__":
    main()
