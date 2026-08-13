#!/usr/bin/env python3
"""
Aggregate the ctx-tax sweep into Tax(ctx) + mechanism signals.

Reads:
  RESULTS_DIR/grid/{arm}/ctx_{ctx}/measure_*.json      -> median_tpot_ms, p99_tpot_ms
  RESULTS_DIR/grid/{arm}/ctx_{ctx}/snapshot_measure_*  -> preemptions, prefix-cache deltas

Emits Tax(ctx) = (TPOT[dsd_k0] - TPOT[no_spec]) / TPOT[no_spec] and, per arm/ctx,
preemption counts and prefix-cache hit rate. Primary endpoint = Tax(max ctx) -
Tax(min ctx).
"""
import json, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
ARMS = ["no_spec", "dsd_k0"]


def cell(arm, ctx):
    d = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
    tpots, p99s = [], []
    preempt, hits, queries = 0.0, 0.0, 0.0
    n_snap = 0
    for f in sorted(d.glob("measure_*.json")):
        if f.name.startswith("snapshot"):
            continue
        r = json.loads(f.read_text())
        if r.get("median_tpot_ms") is not None:
            tpots.append(r["median_tpot_ms"])
        if r.get("p99_tpot_ms") is not None:
            p99s.append(r["p99_tpot_ms"])
    for sf in sorted(d.glob("snapshot_measure_*.json")):
        sn = json.loads(sf.read_text())
        preempt += sn.get("num_preemptions_delta", 0.0)
        hits += sn.get("prefix_cache_hits_delta", 0.0)
        queries += sn.get("prefix_cache_queries_delta", 0.0)
        n_snap += 1
    return {
        "tpot_mean": statistics.mean(tpots) if tpots else float("nan"),
        "tpot_std": statistics.pstdev(tpots) if len(tpots) > 1 else 0.0,
        "p99_mean": statistics.mean(p99s) if p99s else float("nan"),
        "n": len(tpots),
        "preempt_total": preempt,
        "prefix_hit_rate": (hits / queries) if queries else float("nan"),
        "n_snap": n_snap,
    }


def discover_ctxs():
    ctxs = set()
    for arm in ARMS:
        base = RESULTS_DIR / "grid" / arm
        if base.exists():
            for d in base.glob("ctx_*"):
                try:
                    ctxs.add(int(d.name.split("_")[1]))
                except ValueError:
                    pass
    return sorted(ctxs)


def main():
    ctxs = discover_ctxs()
    rows = []
    taxes = {}
    print(f"{'ctx':>7} | {'TPOT_A(nospec)':>16} | {'TPOT_B(dsd_k0)':>16} | "
          f"{'Tax%':>7} | {'preempt_A/B':>12} | {'hitrate_A/B':>13}")
    print("-" * 90)
    for ctx in ctxs:
        a, b = cell("no_spec", ctx), cell("dsd_k0", ctx)
        tax = (b["tpot_mean"] - a["tpot_mean"]) / a["tpot_mean"] * 100 if a["tpot_mean"] else float("nan")
        taxes[ctx] = tax
        rows.append({"ctx": ctx, "no_spec": a, "dsd_k0": b, "tax_pct": tax})
        print(f"{ctx:>7} | {a['tpot_mean']:>7.2f}±{a['tpot_std']:>4.2f}    | "
              f"{b['tpot_mean']:>7.2f}±{b['tpot_std']:>4.2f}    | {tax:>+6.2f} | "
              f"{a['preempt_total']:>5.0f}/{b['preempt_total']:<5.0f} | "
              f"{a['prefix_hit_rate']:>5.2f}/{b['prefix_hit_rate']:<5.2f}")
    if len(ctxs) >= 2:
        lo, hi = ctxs[0], ctxs[-1]
        print("-" * 90)
        print(f"PRIMARY: Tax({hi}) - Tax({lo}) = {taxes[hi]-taxes[lo]:+.2f} pp "
              f"(Tax {lo}={taxes[lo]:+.2f}%, Tax {hi}={taxes[hi]:+.2f}%)")
    (RESULTS_DIR / "ctxtax_summary.json").write_text(
        json.dumps({"rows": rows, "tax_by_ctx": taxes}, indent=2))


if __name__ == "__main__":
    main()
