"""Per-cell r2 table for the Speculators Open Research Hub entry, from committed raw outputs only.

  python build_hub_table.py > HUB_TABLE.md

Sources (relative to mrv2_ctx_20261105/):
  costmap_20261107/run_20261111r2/evidence/costmap_out/<CONFIG>_r<rep>/grid_b<B>_c<C>.json   vllm bench serve output
  costmap_20261107/run_20261111r2/analysis/cells.csv                                        step period, delivered ms/token
Each value is the mean of the two launches (r0, r1).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "costmap_20261107/run_20261111r2"
EV = RUN / "evidence/costmap_out"
CONFIGS = ["OFF", "Z", "K1", "K3", "K5"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, spec):
    return "N/A" if x is None else format(x, spec)


def main():
    cells = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(RUN / "analysis/cells.csv")):
        if r["valid"] == "True":
            key = (int(r["bs"]), int(r["ctx"]), r["config"])
            cells[key]["step"].append(float(r["step_period_ms"]))
            cells[key]["dtok"].append(float(r["delivered_ms_per_token"]))
    bench = defaultdict(lambda: defaultdict(list))
    for (bs, ctx, cfg) in cells:
        for rep in (0, 1):
            d = json.load(open(EV / f"{cfg}_r{rep}" / f"grid_b{bs}_c{ctx}.json"))
            for k in ("output_throughput", "median_tpot_ms", "median_itl_ms", "spec_decode_acceptance_length"):
                bench[(bs, ctx, cfg)][k].append(d.get(k))
    print("| Batch (concurrency) | Context | Config | K | Acceptance length | Output tokens/s | Median TPOT (ms) "
          "| Median ITL (ms) | Step period (ms) | Delivered ms per token |")
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for bs, ctx in sorted({(b, c) for b, c, _ in cells}):
        for cfg in CONFIGS:
            c, b = cells[(bs, ctx, cfg)], bench[(bs, ctx, cfg)]
            al = mean(b["spec_decode_acceptance_length"])
            al_s = "N/A (no drafter)" if cfg == "OFF" else ("N/A (K = 0, no drafts)" if cfg == "Z" else fmt(al, ".2f"))
            k = {"OFF": "—", "Z": "0"}.get(cfg, cfg[1:])
            print(f"| {bs} | {ctx // 1024}k | {cfg} | {k} | {al_s} | {fmt(mean(b['output_throughput']), '.1f')} "
                  f"| {fmt(mean(b['median_tpot_ms']), '.2f')} | {fmt(mean(b['median_itl_ms']), '.2f')} "
                  f"| {fmt(mean(c['step']), '.2f')} | {fmt(mean(c['dtok']), '.3f')} |")


if __name__ == "__main__":
    main()
