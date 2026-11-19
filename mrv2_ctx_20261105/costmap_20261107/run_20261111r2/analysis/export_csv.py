"""Export the registered report (costmap_report_local.json == box report) to flat CSVs for the dashboard."""
import csv, json, sys
from pathlib import Path
R = Path(sys.argv[1]); rep = json.load(open(R / "costmap_report_local.json")); out = R / "analysis"
ORDER = {f"{L['config']}_r{L['rep']}": i + 1 for i, L in enumerate(rep["launches"])}
with open(out / "cells.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["launch", "config", "k", "rep", "launch_order", "shape", "bs", "ctx", "step_period_ms",
                                   "delivered_ms_per_token", "ms_per_output_token", "acceptance_rate", "valid"])
    for L in rep["launches"]:
        for c in L["cells"]:
            if c["kind"] != "grid":
                continue
            w.writerow([f"{L['config']}_r{L['rep']}", L["config"], L["k"], L["rep"], ORDER[f"{L['config']}_r{L['rep']}"], c["id"],
                        c["bs"], c["ctx"], c["step_period_ms"], c["delivered_ms_per_token"], c["ms_per_output_token"],
                        c.get("acceptance_rate"), c["valid"]])
shapes = {c["id"]: (c["bs"], c["ctx"]) for c in rep["launches"][0]["cells"]}
with open(out / "pairs.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["metric", "pair", "base", "other", "shape", "bs", "ctx", "log_ratio", "aa_spread", "verdict"])
    for m, s in rep["stage1"].items():
        for pair, rows in s["pairs"].items():
            other, base = pair.split("_vs_")
            for cid, r in rows.items():
                w.writerow([m, pair, base, other, cid, *shapes[cid], r["log_ratio"], r["aa_spread"], r["verdict"]])
with open(out / "launches.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["launch", "config", "rep", "launch_order", "startup_s", "wall_s", "sentinel_drift_log", "status"])
    for L in rep["launches"]:
        n = f"{L['config']}_r{L['rep']}"
        w.writerow([n, L["config"], L["rep"], ORDER[n], L["startup_s"], L["wall_s"], L["sentinel_drift_log"], L["status"]])
