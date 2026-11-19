"""K-rank flips across context in r2 (RULES.md).

  python k_rank.py > RESULT.json
"""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

A = Path(__file__).resolve().parent.parent / "costmap_20261107/run_20261111r2/analysis"
OPTIONS = ["Z", "K1", "K3", "K5"]
METRIC = "delivered_ms_per_token"


def load():
    per_rep = defaultdict(dict)  # (bs, ctx, rep) -> {config: value}
    for r in csv.DictReader(open(A / "cells.csv")):
        if r["valid"] == "True":
            per_rep[(int(r["bs"]), int(r["ctx"]), int(r["rep"]))][r["config"]] = float(r[METRIC])
    pairs = {}
    for r in csv.DictReader(open(A / "pairs.csv")):
        if r["metric"] == METRIC:
            pairs[(int(r["bs"]), int(r["ctx"]), r["base"], r["other"])] = (float(r["log_ratio"]), r["verdict"])
    return per_rep, pairs


def pair(pairs, bs, ctx, a, b):
    """Direction of a vs b: -1 if a is faster (lower ms/token), +1 if slower; with the registered verdict."""
    if (bs, ctx, a, b) in pairs:
        lr, v = pairs[(bs, ctx, a, b)]  # log(other / base) = log(b / a)
        return (-1 if lr > 0 else 1), v, lr
    lr, v = pairs[(bs, ctx, b, a)]
    return (1 if lr > 0 else -1), v, -lr


def main():
    per_rep, pairs = load()
    grid = defaultdict(list)
    for bs, ctx, rep in per_rep:
        if rep == 0:
            grid[bs].append(ctx)
    out = {"rules": "RULES.md", "metric": METRIC, "options": OPTIONS, "pair_flips": [], "best": {}, "best_changes": []}
    for bs in sorted(grid):
        ctxs = sorted(grid[bs])
        for a, b in itertools.combinations(OPTIONS, 2):
            for c1, c2 in itertools.combinations(ctxs, 2):
                d1, v1, lr1 = pair(pairs, bs, c1, a, b)
                d2, v2, lr2 = pair(pairs, bs, c2, a, b)
                if not (v1 == v2 == "exceeds_aa" and d1 != d2):
                    continue
                reps = []
                for rep in (0, 1):
                    x1, x2 = per_rep[(bs, c1, rep)], per_rep[(bs, c2, rep)]
                    reps.append((x1[a] < x1[b]) == (d1 < 0) and (x2[a] < x2[b]) == (d2 < 0))
                out["pair_flips"].append(dict(bs=bs, pair=[a, b], ctx=[c1, c2], log_ratio_b_over_a=[lr1, lr2],
                                              repeated=all(reps), per_launch=reps))
        best_by_ctx = {}
        for c in ctxs:
            mean = {o: (per_rep[(bs, c, 0)][o] + per_rep[(bs, c, 1)][o]) / 2 for o in OPTIONS}
            order = sorted(OPTIONS, key=mean.get)
            _, v, lr = pair(pairs, bs, c, order[0], order[1])
            per_launch = [min(OPTIONS, key=per_rep[(bs, c, rep)].get) for rep in (0, 1)]
            off = (per_rep[(bs, c, 0)]["OFF"] + per_rep[(bs, c, 1)]["OFF"]) / 2
            best_by_ctx[c] = dict(order=order, mean_ms_per_token={o: round(mean[o], 4) for o in OPTIONS},
                                  off_ms_per_token=round(off, 4), best_vs_second=dict(verdict=v, log_ratio=lr),
                                  resolved=v == "exceeds_aa", best_per_launch=per_launch)
        out["best"][bs] = best_by_ctx
        resolved = {c: r["order"][0] for c, r in best_by_ctx.items() if r["resolved"]}
        if len(set(resolved.values())) > 1:
            rep_ok = all(len({best_by_ctx[c]["best_per_launch"][rep] for c in resolved}) > 1
                         and all(best_by_ctx[c]["best_per_launch"][rep] == resolved[c] for c in resolved)
                         for rep in (0, 1))
            out["best_changes"].append(dict(bs=bs, best=resolved, repeated=rep_ok))
    repeated = [f for f in out["pair_flips"] if f["repeated"]] + [c for c in out["best_changes"] if c["repeated"]]
    out["outcome"] = "candidate" if repeated else "none"
    json.dump(out, __import__("sys").stdout, indent=1)


if __name__ == "__main__":
    main()
