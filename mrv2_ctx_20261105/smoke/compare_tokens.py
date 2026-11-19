"""Compare two probe outputs request by request: divergent fraction and the first
divergence position. Used for A/A (same config twice) before any A/B reading.

  python compare_tokens.py <a.jsonl> <b.jsonl>
"""

import json
import sys


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path)) if r}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(a, b):
    ids = sorted(set(a) & set(b))
    rows = []
    for i in ids:
        ta, tb = a[i]["token_ids"], b[i]["token_ids"]
        if ta is None or tb is None:
            rows.append({"id": i, "error": "token_ids missing"})
            continue
        rows.append({"id": i, "len": [len(ta), len(tb)], "first_divergence": first_divergence(ta, tb)})
    ok = [r for r in rows if "error" not in r]
    div = [r["first_divergence"] for r in ok if r["first_divergence"] is not None]
    return {
        "compared": len(ok),
        "missing": sorted(set(a) ^ set(b)),
        "errors": [r for r in rows if "error" in r],
        "divergent": len(div),
        "first_divergence_positions": sorted(div),
        "rows": rows,
    }


def main():
    json.dump(compare(load(sys.argv[1]), load(sys.argv[2])), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
