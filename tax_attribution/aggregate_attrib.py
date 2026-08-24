#!/usr/bin/env python3
"""
Aggregate the 2x2 attribution grid into the graph term, the pool term and the
residual, and check each pre-registered prediction (PREREGISTRATION.md).

Reads:
  RESULTS_DIR/grid/{arm}/ctx_{ctx}_c{conc}/measure_*.json           -> tpot
  RESULTS_DIR/grid/{arm}/ctx_{ctx}_c{conc}/snapshot_measure_*.json  -> preempt, drafts, census
  RESULTS_DIR/pools.json                                            -> launch-order guard
"""
import json, math, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
MATERIAL_FRAC = float(os.environ.get("MATERIAL_FRAC", "0.10"))
BASE = "base_full_high"
ARMS = ["base_full_high", "base_piece_high", "base_full_low", "base_piece_low", "dsd_k0_low"]
TERMS = {
    "graph": "base_piece_high",
    "pool": "base_full_low",
    "total": "dsd_k0_low",
}


def cell(arm, conc):
    matches = list((RESULTS_DIR / "grid" / arm).glob(f"ctx_*_c{conc}"))
    if not matches:
        return None
    d = matches[0]
    tpots = [json.loads(f.read_text()).get("median_tpot_ms")
             for f in sorted(d.glob("measure_*.json")) if not f.name.startswith("snapshot")]
    tpots = [x for x in tpots if x is not None]
    preempt = drafts = 0.0
    decode = prefill = 0.0
    for sf in sorted(d.glob("snapshot_measure_*.json")):
        sn = json.loads(sf.read_text())
        preempt += sn.get("num_preemptions_delta", 0.0)
        drafts += sn.get("spec_decode_num_draft_tokens_delta", 0.0)
        census = sn.get("iteration_tokens_census") or {}
        decode += census.get("decode_only_steps", 0.0)
        prefill += census.get("prefill_heavy_steps", 0.0)
    n = len(tpots)
    mean = statistics.mean(tpots) if n else float("nan")
    std = statistics.pstdev(tpots) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n,
            "sem": (std / math.sqrt(n)) if n else float("nan"),
            "preempt": preempt, "draft_tokens": drafts,
            "decode_steps": decode, "prefill_steps": prefill}


def discover_concurrencies():
    found = set()
    for arm in ARMS:
        base = RESULTS_DIR / "grid" / arm
        if base.exists():
            for d in base.glob("ctx_*_c*"):
                try:
                    found.add(int(d.name.rsplit("_c", 1)[1]))
                except (IndexError, ValueError):
                    pass
    return sorted(found, reverse=True)


def launch_order_ok():
    path = RESULTS_DIR / "pools.json"
    if not path.exists():
        return False, "pools.json missing; launch order was not recorded"
    entries = json.loads(path.read_text())
    for arm in ARMS:
        mine = [e for e in entries if e["arm"] == arm]
        if not mine:
            continue
        if not any(e["discarded"] for e in mine):
            return False, f"{arm} has no discarded first launch"
        pools = {e["tokens"] for e in mine if not e["discarded"]}
        if len(pools) > 1:
            return False, f"{arm} measured more than one pool {sorted(pools)}"
    return True, "each arm has a discarded first launch and one measured pool"


def verdict(residual, sem, floor):
    """Fail only when a residual is both resolvable and worth resolving.

    With three measured runs 2*sem is small enough that sampling noise alone
    flips a verdict, so a residual has to clear the noise band and a share of
    the total term before it counts against a prediction.
    """
    resolvable = abs(residual) > 2 * sem
    material = abs(residual) > floor
    if resolvable and material:
        return "FAILS"
    if resolvable:
        return "holds (residual resolvable but immaterial)"
    return "holds"


def report(conc):
    cells = {arm: cell(arm, conc) for arm in ARMS}
    missing = [a for a, c in cells.items() if c is None or c["n"] == 0]
    if missing:
        print(f"  incomplete: {missing}")
        return None
    base = cells[BASE]
    print(f"{'arm':>16} | {'TPOT ms':>16} | {'vs base':>9} | {'preempt':>7} | "
          f"{'drafts':>7} | {'steps d/p':>12}")
    print("  " + "-" * 84)
    for arm in ARMS:
        c = cells[arm]
        delta = c["mean"] - base["mean"]
        pct = 100 * delta / base["mean"]
        print(f"{arm:>16} | {c['mean']:>8.3f}±{c['std']:<6.3f} | "
              f"{pct:>+8.2f}% | {c['preempt']:>7.0f} | {c['draft_tokens']:>7.0f} | "
              f"{c['decode_steps']:>5.0f}/{c['prefill_steps']:<5.0f}")

    terms = {k: cells[a]["mean"] - base["mean"] for k, a in TERMS.items()}
    residual_add = terms["total"] - terms["graph"] - terms["pool"]
    sem_add = math.sqrt(sum(cells[a]["sem"] ** 2 for a in
                            (BASE, TERMS["graph"], TERMS["pool"], TERMS["total"])))
    residual_spec = cells["dsd_k0_low"]["mean"] - cells["base_piece_low"]["mean"]
    sem_spec = math.hypot(cells["dsd_k0_low"]["sem"], cells["base_piece_low"]["sem"])

    print("  " + "-" * 84)
    for name, key in (("graph term", "graph"), ("pool term", "pool"), ("total", "total")):
        print(f"  {name:<12} {terms[key]:>+8.3f} ms  ({100*terms[key]/base['mean']:>+6.2f}%)")
    floor = MATERIAL_FRAC * abs(terms["total"])
    p1 = verdict(residual_add, sem_add, floor)
    p3 = verdict(residual_spec, sem_spec, floor)
    print(f"  additivity   {residual_add:>+8.3f} ms  (2*sem {2*sem_add:.3f}, "
          f"material {floor:.3f}) -> P1 {p1}")
    print(f"  spec residual{residual_spec:>+8.3f} ms  (2*sem {2*sem_spec:.3f}, "
          f"material {floor:.3f}) -> P3 {p3}")
    return {"concurrency": conc,
            "cells": {a: cells[a] for a in ARMS},
            "terms_ms": terms,
            "additivity_residual_ms": residual_add, "additivity_2sem": 2 * sem_add,
            "spec_residual_ms": residual_spec, "spec_residual_2sem": 2 * sem_spec,
            "material_floor_ms": floor, "p1": p1, "p3": p3}


def main():
    ok, why = launch_order_ok()
    print(f"launch-order guard: {'OK' if ok else 'BLOCKED'} — {why}")
    if not ok:
        print("pool-derived readings are not comparable across arms; timing rows still stand")
    out = []
    for conc in discover_concurrencies():
        print(f"\n===== concurrency {conc} =====")
        r = report(conc)
        if r:
            out.append(r)
    if len(out) >= 2:
        hi, lo = out[0], out[-1]
        kv_hi, kv_lo = hi["terms_ms"]["pool"], lo["terms_ms"]["pool"]
        gr_hi, gr_lo = hi["terms_ms"]["graph"], lo["terms_ms"]["graph"]
        print(f"\nP2 (pool term is a batch effect): pool {kv_lo:+.3f} ms at c={lo['concurrency']} "
              f"-> {kv_hi:+.3f} ms at c={hi['concurrency']}; "
              f"graph {gr_lo:+.3f} -> {gr_hi:+.3f}")
        p2 = (abs(kv_hi) > abs(kv_lo)
              and abs(kv_lo) <= max(lo["additivity_2sem"], lo["material_floor_ms"]))
        print(f"P2 {'holds' if p2 else 'FAILS or is unclear'}")
    (RESULTS_DIR / "attrib_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
