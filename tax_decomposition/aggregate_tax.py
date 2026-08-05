#!/usr/bin/env python3
"""
Aggregate tax-decomposition grid results (pre-reg §4/§6). Run at desk after the
rental dump; the rental only collects raw JSON.

Reads (measure seeds only, warmups discarded):
  RESULTS_DIR/grid/{arm}/ctx_{ctx}/measure_{0,1,2}.json   -> median_tpot_ms (p50)
  RESULTS_DIR/grid/{arm}/ctx_{ctx}/snapshot_measure_*.json -> preemptions, drafts

Field names grep-verified against pr49986_runbook/raw/aggregate/bench_on.json:
  TPOT p50 == median_tpot_ms (no separate p50 key); p99 == p99_tpot_ms.

Prints per-ctx Tax(V1), Tax(V2), graph-mode component, residual, baseline guard,
Gate T, and the spot-check drift. Does not write anything into the pre-reg.
"""
import json, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
CTXS = [400, 4000]
ARMS = ["v1_no_spec", "v1_dsd_k0", "v2_no_spec", "v2_dsd_k0", "spot"]


def load_arm_ctx(arm, ctx):
    d = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
    tpots, p99s, preempts, drafts = [], [], [], []
    for s in range(3):
        f = d / f"measure_{s}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("median_tpot_ms") is not None:
            tpots.append(r["median_tpot_ms"])
        if r.get("p99_tpot_ms") is not None:
            p99s.append(r["p99_tpot_ms"])
        sf = d / f"snapshot_measure_{s}.json"
        if sf.exists():
            sn = json.loads(sf.read_text())
            preempts.append(sn.get("num_preemptions_delta", 0))
            drafts.append(sn.get("spec_decode_num_drafts_delta", 0))
    if not tpots:
        return None
    return {
        "n": len(tpots),
        "tpot_mean": statistics.mean(tpots),
        "tpot_std": statistics.stdev(tpots) if len(tpots) > 1 else 0.0,
        "p99_mean": statistics.mean(p99s) if p99s else float("nan"),
        "preempt_total": sum(preempts),
        "drafts_total": sum(drafts),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def main():
    data = {}
    for arm in ARMS:
        for ctx in CTXS:
            r = load_arm_ctx(arm, ctx)
            if r:
                data[(arm, ctx)] = r

    print(f"{'arm':14}{'ctx':>6}{'n':>3}{'TPOT_p50_ms':>18}"
          f"{'preempt':>9}{'drafts':>9}")
    for (arm, ctx), r in sorted(data.items()):
        flag = "  <-- PREEMPT" if r["preempt_total"] > 0 else ""
        print(f"{arm:14}{ctx:>6}{r['n']:>3}"
              f"{r['tpot_mean']:>10.2f}±{r['tpot_std']:<6.2f}"
              f"{r['preempt_total']:>9}{r['drafts_total']:>9}{flag}")

    print("\n=== Tax decomposition (TPOT p50, pre-reg §4) ===")
    for ctx in CTXS:
        A, B = data.get(("v1_no_spec", ctx)), data.get(("v1_dsd_k0", ctx))
        C, D = data.get(("v2_no_spec", ctx)), data.get(("v2_dsd_k0", ctx))
        tax_v1 = (B["tpot_mean"] - A["tpot_mean"]) / A["tpot_mean"] if A and B else None
        tax_v2 = (D["tpot_mean"] - C["tpot_mean"]) / C["tpot_mean"] if C and D else None
        print(f"\nctx={ctx}:")
        print(f"  Tax(V1) = {pct(tax_v1)}   Tax(V2) = {pct(tax_v2)}"
              f"{'  (V2 missing — Gate B fail?)' if tax_v2 is None else ''}")
        if tax_v1 is not None and tax_v2 is not None:
            print(f"  graph-mode component = {pct(tax_v1 - tax_v2)}p   "
                  f"residual (Tax V2) = {pct(tax_v2)}")
        if A and C:
            drift = (C["tpot_mean"] - A["tpot_mean"]) / A["tpot_mean"]
            print(f"  baseline guard (no_spec V1 vs V2 drift) = {pct(drift)} "
                  f"(large => confounded)")

    print("\n=== Gate T (ctx=400, pre-reg §3) ===")
    A, B = data.get(("v1_no_spec", 400)), data.get(("v1_dsd_k0", 400))
    if A and B:
        t = (B["tpot_mean"] - A["tpot_mean"]) / A["tpot_mean"]
        verdict = ("PROCEED (>=10%)" if t >= 0.10 else
                   "#49986 CORRECTION (tax vanished, <5%)" if t < 0.05 else
                   "INCONCLUSIVE (5-10%)")
        print(f"  Tax(V1)@400 = {pct(t)} -> {verdict}")
    else:
        print("  (v1 arms @400 missing)")

    S, B = data.get(("spot", 400)), data.get(("v1_dsd_k0", 400))
    if S and B:
        d = (S["tpot_mean"] - B["tpot_mean"]) / B["tpot_mean"]
        print(f"\n=== Spot S: main-V1 vs branch-V1 @400 drift = {pct(d)} "
              f"(large => branch altered V1, re-scope) ===")


if __name__ == "__main__":
    main()
