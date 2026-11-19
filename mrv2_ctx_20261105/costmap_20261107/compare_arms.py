"""Server comparison verdict (kernel_probe_20261117/server_cmp/PREREG_SERVER_66852_DRAFT_20261119.md §6-§7).
Per stage-1 shape: log step period of arm 0 and arm P over their launches; test cells are those whose target forward
runs at M = 32, every other shape is a control.

  python compare_arms.py OUT > compare_arms.json
"""

import json
import math
import os
import sys

import numpy as np

import analyze_costmap as A

TEST_M = 32
TEST_MIN, CONTROL_MAX = 0.03, 0.02
REPS = 3
STATEMENT = {
    "lowers": "Under these conditions, the #66852 (16, 32] config lowered the server step at M = 32; controls unchanged.",
    "not_detected": "No step change beyond the registered threshold was detected.",
    "not_attributable": "Arms differ at cells the config does not change; cause open.",
    "controls_incomplete": "Every evaluated control held, but not all 8 were evaluable; the observed change at M = 32 "
                           "is reported with the missing controls' reasons, and its attribution to the config is "
                           "not evaluable.",
    "not_evaluable": "No verdict: a test cell lacks the registered launches or ran at an unplanned M.",
}
N_CONTROLS = 8


def cell_stats(x0, xp):
    """x0, xp: step periods (ms) of one cell; d > 0 means P faster."""
    l0, lp = np.log(x0), np.log(xp)
    d = float(l0.mean() - lp.mean())
    s = float(max(np.ptp(l0), np.ptp(lp)))
    return {"d": round(d, 5), "s": round(s, 5), "separated": bool(max(xp) < min(x0)),
            "saved_ms": round(float(np.mean(x0) - np.mean(xp)), 4)}


def decide(cells):
    """cells: id -> {"m": planned M, "m_ok": bool, "0": [...], "P": [...]}. Returns the verdict and per-cell rows."""
    rows, missing_tests, breaches, lowers = {}, [], [], []
    for cid, c in sorted(cells.items()):
        test = c["m"] == TEST_M
        row = {"m": c["m"], "role": "test" if test else "control", "n0": len(c["0"]), "nP": len(c["P"])}
        if not c["m_ok"] or len(c["0"]) < REPS or len(c["P"]) < REPS:
            row["status"] = "m_not_as_planned" if not c["m_ok"] else "missing_launches"
            if test:
                missing_tests.append(cid)
            rows[cid] = row
            continue
        row.update(cell_stats(c["0"], c["P"]))
        if test:
            row["threshold"] = round(max(2 * row["s"], math.log(1 / (1 - TEST_MIN))), 5)
            row["lowers"] = row["d"] > row["threshold"] and row["separated"]
            if row["lowers"]:
                lowers.append(cid)
        else:
            row["threshold"] = round(max(2 * row["s"], math.log(1 / (1 - CONTROL_MAX))), 5)
            row["holds"] = abs(row["d"]) <= row["threshold"]
            if not row["holds"]:
                breaches.append(cid)
        row["status"] = "evaluated"
        rows[cid] = row
    tests = [cid for cid, c in cells.items() if c["m"] == TEST_M]
    controls = [cid for cid, r in rows.items() if r["role"] == "control"]
    evaluated = sum(rows[c]["status"] == "evaluated" for c in controls)
    # prereg §7 rule 4 (user, 2026-11-19): test cell invalid > breach among evaluated controls > fewer than 8 > §7
    if not tests or missing_tests:
        outcome = "not_evaluable"
    elif breaches:
        outcome = "not_attributable"
    elif evaluated < N_CONTROLS:
        outcome = "controls_incomplete"
    elif set(lowers) == set(tests):
        outcome = "lowers"
    else:
        outcome = "not_detected"
    return {"outcome": outcome, "statement": STATEMENT[outcome], "tests": sorted(tests), "lowers": sorted(lowers),
            "test_rule_met": bool(tests) and not missing_tests and set(lowers) == set(tests),
            "control_breaches": sorted(breaches), "controls_evaluated": evaluated, "controls_expected": N_CONTROLS,
            "controls_missing": {c: rows[c]["status"] for c in controls if rows[c]["status"] != "evaluated"},
            "controls_absent": max(0, N_CONTROLS - len(controls)),
            "missing_tests": sorted(missing_tests), "cells": rows}


def target_tokens(launch_dir, cell, k):
    """Target-forward graph tokens over the cell's steady steps (prereg §6)."""
    rows = [r for r in A.load_trace(launch_dir) if cell["t0_ns"] <= r["t_ns"] <= cell["t1_ns"]]
    st = A.steps(rows)
    _, ok, _ = A.classify(st, cell["bs"], k)
    return sorted({s["target"]["graph_tokens"] for s, o in zip(st, ok) if o and s["target"]})


def load(out_dir):
    cells, launches = {}, []
    for L in map(json.loads, open(os.path.join(out_dir, "launches.jsonl"))):
        launches.append({key: L.get(key) for key in ("dir", "arm", "rep", "status")})
        if L.get("status") != "complete" or L.get("arm") not in ("0", "P"):
            continue
        d = os.path.join(out_dir, L["dir"])
        rep = A.load_launch(out_dir, L)
        raw = {c["id"]: c for c in map(json.loads, open(os.path.join(d, "cells.jsonl")))}
        for c in rep["cells"]:
            if c["kind"] != "grid":
                continue
            k = A.cell_k(L["config"], c["bs"])
            m = c["bs"] * (k + 1)
            e = cells.setdefault(c["id"], {"m": m, "m_ok": True, "0": [], "P": [], "seen": [], "acceptance": {"0": [], "P": []}})
            toks = target_tokens(d, raw[c["id"]], k) if "t0_ns" in raw.get(c["id"], {}) else []
            e["seen"].append({"launch": L["dir"], "graph_tokens": toks, "valid": c.get("valid"), "invalid": c.get("invalid")})
            if toks and toks != [m]:
                e["m_ok"] = False
            if c.get("valid"):
                e[L["arm"]].append(c["step_period_ms"])
                e["acceptance"][L["arm"]].append(c.get("acceptance_rate"))
    return cells, launches


def report(out_dir):
    cells, launches = load(out_dir)
    v = decide(cells)
    for cid, c in cells.items():
        v["cells"][cid].update(step_ms={"0": c["0"], "P": c["P"]}, acceptance=c["acceptance"], seen=c["seen"])
    v["launches"] = launches
    return v


if __name__ == "__main__":
    json.dump(report(sys.argv[1]), sys.stdout, indent=1)
