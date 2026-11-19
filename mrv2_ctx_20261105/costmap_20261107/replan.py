"""Tier 1 re-estimate from the measured pilot (PILOT.md §4). Credit and price are typed in at
decision time; nothing here reads a stored balance.

  python replan.py <pilot_out> --credit 7.33 --usd-per-hour 3.40 [--box-overhead-min 20]

Tier 1 = OFF, Z, K1, K2, K4, K5 and K3 again (one launch each). Each launch is estimated as
the pilot launch wall time x LAUNCH_MARGIN (K5 decodes more drafts per step than K3 and OFF
fewer; the pilot gives only K3). Cuts, in the order fixed in DESIGN.md §4: (1) drop K2 and K4,
(2) drop the 2k and 8k columns except BS 8. Never cut: K3 repeat, OFF, Z, sentinels, raw files.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_costmap as A  # noqa: E402

LAUNCH_MARGIN = 1.25
COST_MARGIN = 1.2
TIER1 = ("OFF", "Z", "K1", "K2", "K4", "K5", "K3")


def pilot_facts(rep):
    (L,) = [x for x in rep["launches"] if x["config"] == "K3" and x["rep"] == 0]
    run = [c for c in L["cells"] if "wall_s" in c]
    cell_s = {c["id"]: c["wall_s"] for c in run}
    return {"status": L["status"], "startup_s": L.get("startup_s"), "launch_wall_s": L.get("wall_s"),
            "cells_run": len(run), "cell_wall_s_total": round(sum(cell_s.values()), 1), "cell_wall_s": cell_s,
            "valid_share": round(sum(c["valid"] for c in run) / len(run), 4) if run else None,
            "infeasible": [c["id"] for c in run if "infeasible_preempted" in c["invalid"]],
            "sentinel_drift_log": L["sentinel_drift_log"]}


def option(name, configs, facts, drop_cells, overhead_min, price, credit):
    cut_s = sum(facts["cell_wall_s"].get(c, 0) for c in drop_cells)
    per_launch = (facts["launch_wall_s"] - cut_s) * LAUNCH_MARGIN
    hours = (overhead_min * 60 + per_launch * len(configs)) / 3600
    usd = hours * price
    return {"option": name, "configs": list(configs), "dropped_cells": sorted(drop_cells),
            "per_launch_min": round(per_launch / 60, 1), "hours": round(hours, 2), "usd": round(usd, 2),
            "usd_with_margin": round(usd * COST_MARGIN, 2), "fits_credit": usd * COST_MARGIN <= credit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_out")
    ap.add_argument("--credit", type=float, required=True)
    ap.add_argument("--usd-per-hour", type=float, required=True)
    ap.add_argument("--box-overhead-min", type=float, default=20.0)
    a = ap.parse_args()
    facts = pilot_facts(A.report(a.pilot_out))
    out = {"pilot": {k: v for k, v in facts.items() if k != "cell_wall_s"}, "inputs": vars(a)}
    if facts["status"] != "complete" or facts["launch_wall_s"] is None:
        out["decision"] = "pilot not complete: report cause, no tier 1 estimate, no re-rental"
        json.dump(out, sys.stdout, indent=1)
        return
    cols = [c["id"] for c in A.plan(20261107) if c["kind"] == "grid" and c["ctx"] in (2048, 8192) and c["bs"] != 8]
    opts = [option("full", TIER1, facts, [], a.box_overhead_min, a.usd_per_hour, a.credit),
            option("cut1_drop_K2_K4", [c for c in TIER1 if c not in ("K2", "K4")], facts, [],
                   a.box_overhead_min, a.usd_per_hour, a.credit),
            option("cut2_also_drop_2k_8k_except_bs8", [c for c in TIER1 if c not in ("K2", "K4")], facts, cols,
                   a.box_overhead_min, a.usd_per_hour, a.credit)]
    out["options"] = opts
    fit = [o["option"] for o in opts if o["fits_credit"]]
    out["decision"] = (f"first option that fits: {fit[0]}" if fit else
                       "no option fits the credit with margin: top up or wait; do not cut repeats, controls or raw files")
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
