"""Cost map analysis and gates (DESIGN.md §5-6, PILOT.md), fixed before the run.

  python analyze_costmap.py <costmap_out> > costmap_report.json

<costmap_out>/launches.jsonl lists launches (config, k, rep, dir, status); each launch dir has
cells.jsonl and metrics.jsonl (runner) and trace/mrv2_trace_*.jsonl (smoke/trace hook).

What the step time is (EngineCore.step_with_batch_queue, async scheduling, queue size 2 on
MRv2): one busy-loop iteration schedules step N, submits it, then waits for step N-1's result
in EngineCoreProc._wait_for_output (`ev=wait`), which polls the result and meanwhile handles
client requests (`ev=input`). Each `sched` row starts a step; the dispatch / replay / wait /
input rows until the next `sched` row belong to it. `step_period` = interval between adjacent
steady `sched` rows = the loop period. `wait_call` = time inside `_wait_for_output` (until the
result was taken, request handling included; not GPU time); `outside_wait` = period minus
that. An interval counts as GPU-bound only when the result was still pending at the start of
the wait (wait_call > WAIT_GPU_NS) and no client request was handled in it: then step N was
queued behind N-1, the GPU had no idle gap, and the period is the GPU time per step (all
work of the step; draft and verify are not split). GPU kernel time is not measured.
"""

import glob
import json
import os
import random
import sys
from collections import Counter

import numpy as np

from costmap_cmd import OUT_LEN

CTX = (1024, 2048, 4096, 8192, 16384, 32768)
BS = (1, 2, 4, 8, 16, 32, 64)
BS_CAP = {32768: 8, 16384: 16, 8192: 32}
DENSE_CTX, DENSE_BS = 4096, (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
SENTINEL = (8, 4096)
MIN_STEPS, FULL_SHARE, MAX_DROP_SHARE = 40, 0.99, 0.01
WAIT_GPU_NS = 50_000
CTX_SLACK = OUT_LEN + 8
TRAIN_CTX, TRAIN_BS = (1024, 4096, 16384, 32768), (1, 4, 16, 64)
DECISION_BAND = (0.3, 0.85)
CONFIG_K = {"OFF": None, "Z": 0, "K1": 1, "K2": 2, "K3": 3, "K4": 4, "K5": 5}
EARLY_CELLS = 3
LAUNCH_VALID_SHARE = 0.90
GPU_MIB = 20000
# Stage 1 (STAGE1_ODD_PREREG.md): ten shapes, each once per launch, plus the b8 x 4k repeats.
STAGE1_SHAPES = ((1, 1024), (8, 1024), (16, 1024), (1, 4096), (8, 4096), (16, 4096), (1, 16384), (8, 16384),
                 (32, 1024), (32, 2048))
STAGE1_CONFIGS = ("OFF", "Z", "K1", "K3", "K5")


def plan(seed, name="pilot"):
    """Cells in run order: warm-up sentinel, first half, sentinel, second half, sentinel."""
    if name == "stage1":
        cells = [{"kind": "grid", "bs": b, "ctx": c, "seed": 20000 + i} for i, (b, c) in enumerate(STAGE1_SHAPES)]
    else:
        cells = [{"kind": "grid", "bs": b, "ctx": c} for c in CTX for b in BS if b <= BS_CAP.get(c, 64)]
        cells += [{"kind": "dense", "bs": b, "ctx": DENSE_CTX} for b in DENSE_BS]
        for i, c in enumerate(cells):
            c["seed"] = 10000 + i
    random.Random(seed).shuffle(cells)
    half = len(cells) // 2
    sent = lambda tag, s: {"kind": tag, "bs": SENTINEL[0], "ctx": SENTINEL[1], "seed": s}
    out = [sent("warmup", 9000)] + cells[:half] + [sent("sentinel", 9001)] + cells[half:] + [sent("sentinel", 9002)]
    for i, c in enumerate(out):
        c["order"] = i
        c["id"] = f"{c['kind']}_b{c['bs']}_c{c['ctx']}" + ("" if c["kind"] in ("grid", "dense") else f"_{c['seed']}")
        c["role"] = "observation" if c["kind"] == "dense" else "control"
    return out


def load_trace(launch_dir):
    rows = []
    for p in glob.glob(os.path.join(launch_dir, "trace", "mrv2_trace_*.jsonl")):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    rows.sort(key=lambda r: r["t_ns"])
    return rows


def _int(x):
    return None if x is None else int(x)


def steps(rows):
    """One record per `sched` row with the rows that follow it until the next `sched` row."""
    out = []
    for r in rows:
        if r["ev"] == "sched":
            out.append({"t": r["t_ns"], "bs": r["bs"], "prefill_rows": r["prefill_rows"],
                        "verified": {int(a): b for a, b in (r.get("verified") or {}).items()},
                        "ctx": r.get("ctx_snapshot"), "target": None, "target_replay": False,
                        "drafter_prefill": None, "drafter_decode": [], "wait_ns": None, "inputs": 0,
                        "tokens": r.get("tokens")})
        elif out:
            s = out[-1]
            if r["ev"] == "dispatch" and r["role"] == "target" and s["target"] is None:
                s["target"] = {"num_reqs": _int(r["num_reqs"]), "num_tokens": _int(r["num_tokens"]),
                               "q": _int(r["q"]), "mode": r["desc"]["mode"], "graph_reqs": _int(r["desc"]["reqs"]),
                               "graph_tokens": _int(r["desc"]["tokens"])}
            elif r["ev"] == "replay" and r["role"] == "target":
                s["target_replay"] = True
            elif r["ev"] == "dispatch" and r["role"] == "drafter_prefill" and s["drafter_prefill"] is None:
                s["drafter_prefill"] = r["desc"]["mode"]
            elif r["ev"] == "dispatch" and r["role"] == "drafter_decode":
                s["drafter_decode"].append(r["desc"]["mode"])
            elif r["ev"] == "wait":
                s["wait_ns"] = (s["wait_ns"] or 0) + r["wait_ns"]
            elif r["ev"] == "input":
                s["inputs"] += 1
    return out


def verify_ok(s, bs, k):
    if not k:
        return sum(n for w, n in s["verified"].items() if w) == 0
    return s["verified"] == {k: bs}


def target_path(s):
    t = s["target"]
    if t is None:
        return "none"
    mode = t["mode"] if t["mode"] != "FULL" or s["target_replay"] else "FULL_no_replay"
    return f"{mode}/r{t['graph_reqs']}"


def classify(st, bs, k):
    """Steady candidates, steady steps and attribution for one cell window."""
    q = 1 if not k else k + 1
    cand = [s["bs"] == bs and s["prefill_rows"] == 0 for s in st]
    ok = [c and verify_ok(s, bs, k) for s, c in zip(st, cand)]
    attr = [o and s["target"] is not None and s["target"]["num_reqs"] == bs and s["target"]["q"] == q
            and s["target"]["num_tokens"] == bs * q for s, o in zip(st, ok)]
    return cand, ok, attr


def summarize_cell(rows, metrics, cell, k):
    base = {key: cell.get(key) for key in ("id", "kind", "role", "bs", "ctx", "order", "seed")}
    if "t0_ns" not in cell:
        return {**base, "invalid": ["not_run: " + cell.get("skipped", "missing")]}
    w = [r for r in rows if cell["t0_ns"] <= r["t_ns"] <= cell["t1_ns"]]
    st = steps(w)
    cand, ok, attr = classify(st, cell["bs"], k)
    iv = [i for i in range(len(st) - 1) if ok[i] and ok[i + 1]]
    per = np.array([st[i + 1]["t"] - st[i]["t"] for i in iv], float) / 1e6
    waits = [st[i]["wait_ns"] for i in iv]
    has_wait = [x is not None for x in waits]
    inputs = np.array([st[i]["inputs"] for i in iv])
    out = {**base, "wall_s": round((cell["t1_ns"] - cell["t0_ns"]) / 1e9, 1), "steps_in_window": len(st),
           "steady_candidates": sum(cand), "steady_steps": sum(ok), "steady_intervals": len(iv),
           "dropped_verify_width": sum(cand) - sum(ok), "attributed_steps": sum(attr)}
    first = next((i for i, o in enumerate(ok) if o), None)
    last = max((i for i, o in enumerate(ok) if o), default=None)
    out["steps_before_steady"] = first
    out["prefill_tokens_before_steady"] = sum(s["tokens"] or 0 for s in st[:first] if s["prefill_rows"]) if first is not None else None
    out["prefill_steps_after_first_steady"] = sum(1 for s in st[first:] if s["prefill_rows"]) if first is not None else None
    out["steps_after_steady"] = len(st) - 1 - last if last is not None else None
    out["inputs_in_steady_intervals"] = int(inputs.sum()) if len(iv) else 0
    paths = Counter(target_path(st[i]) for i in iv)
    out["target_paths"] = dict(paths)
    out["target_full_share"] = round(sum(n for p, n in paths.items() if p.startswith("FULL/")) / len(iv), 4) if iv else None
    out["target_graph_reqs"] = sorted({st[i]["target"]["graph_reqs"] for i in iv if st[i]["target"]})
    out["drafter_paths"] = dict(Counter(f"{st[i]['drafter_prefill']}|{'/'.join(st[i]['drafter_decode'])}" for i in iv))
    ctxs = [s["ctx"] for s, o in zip(st, ok) if o and s["ctx"] is not None]
    out["ctx_snapshot_p10_p50_p90"] = [int(np.quantile(ctxs, q)) for q in (0.1, 0.5, 0.9)] if ctxs else None
    m = [x for x in metrics if cell["t0_ns"] <= x["t_ns"] <= cell["t1_ns"]]
    out["metrics_samples"] = len(m)
    if m:
        out["kv_usage_p50_max"] = [round(float(np.median([x["kv"] for x in m])), 4), round(max(x["kv"] for x in m), 4)]
    d = cell.get("metrics_delta") or {}
    drafts, acc = d.get("vllm:spec_decode_num_draft_tokens_total"), d.get("vllm:spec_decode_num_accepted_tokens_total")
    out["draft_tokens"], out["accepted_tokens"] = drafts, acc
    out["acceptance_rate"] = round(acc / drafts, 4) if drafts else None
    if len(iv):
        rng = np.random.default_rng(cell["order"])
        boot = np.median(rng.choice(per, size=(1000, len(per))), axis=1)
        out["step_period_ms"] = round(float(np.median(per)), 4)
        out["step_period_p10_p90_ms"] = [round(float(np.quantile(per, 0.1)), 4), round(float(np.quantile(per, 0.9)), 4)]
        out["step_period_within_cell_ci95_ms"] = [round(float(np.quantile(boot, 0.025)), 4),
                                                  round(float(np.quantile(boot, 0.975)), 4)]
        out["step_period_by_target_path_ms"] = {p: round(float(np.median([per[j] for j, i in enumerate(iv)
                                                                          if target_path(st[i]) == p])), 4) for p in paths}
        if all(has_wait):
            wv = np.array(waits, float) / 1e6
            gb = (wv > WAIT_GPU_NS / 1e6) & (inputs == 0)
            out["wait_call_ms_p50"] = round(float(np.median(wv)), 4)
            out["outside_wait_ms_p50"] = round(float(np.median(per - wv)), 4)
            out["gpu_bound_share"] = round(float(gb.mean()), 4)
            if gb.any():
                out["step_period_gpu_bound_ms"] = round(float(np.median(per[gb])), 4)
    if len(iv) and (not k or out["acceptance_rate"] is not None):
        # Cell-level acceptance from the server counters (whole cell, not only steady steps).
        tok = 1.0 + (k or 0) * (out["acceptance_rate"] or 0.0)
        out["tokens_per_row_step"] = round(tok, 4)
        out["ms_per_output_token"] = round(out["step_period_ms"] / (cell["bs"] * tok), 4)
    if cell.get("bench_tpot_ms") is not None:
        out["bench_median_tpot_ms"] = cell["bench_tpot_ms"]
        # Delivered denominator: client-side time per delivered token of one request (median over the
        # cell's requests, whole decode, not only steady steps); divided by BS for the server as a whole.
        out["delivered_ms_per_token"] = round(cell["bench_tpot_ms"] / cell["bs"], 4)
    reasons = []
    if cell.get("timeout"):
        reasons.append("timeout")
    elif cell.get("rc"):
        reasons.append("bench_failed")
    if cell.get("preempted"):
        reasons.append("infeasible_preempted")
    if len(iv) < MIN_STEPS:
        reasons.append("insufficient_steady_steps")
    if sum(cand) and (sum(cand) - sum(ok)) / sum(cand) > MAX_DROP_SHARE:
        reasons.append("verify_width_mismatch")
    if sum(ok) and (sum(ok) - sum(attr)) / sum(ok) > MAX_DROP_SHARE:
        reasons.append("unattributed_steps")
    if ctxs and not (0.98 * cell["ctx"] <= out["ctx_snapshot_p10_p50_p90"][1] <= cell["ctx"] + CTX_SLACK):
        reasons.append("ctx_mismatch")
    if iv and not all(has_wait):
        reasons.append("missing_wait_rows")
    if not m:
        reasons.append("missing_metrics_samples")
    if cell["role"] == "control" and iv and out["target_full_share"] < FULL_SHARE:
        reasons.append("target_not_full")
    out["invalid"] = reasons
    out["valid"] = not reasons
    return out


def load_launch(out_dir, L):
    d = os.path.join(out_dir, L["dir"])
    rows = load_trace(d)
    mpath = os.path.join(d, "metrics.jsonl")
    metrics = [json.loads(x) for x in open(mpath)] if os.path.exists(mpath) else []
    cpath = os.path.join(d, "cells.jsonl")
    cells = [json.loads(x) for x in open(cpath)] if os.path.exists(cpath) else []
    seen = {c["id"] for c in cells}
    cells += [dict(c, skipped="not reached") for c in plan(L.get("plan_seed", 20261107), L.get("plan", "pilot"))
              if c["id"] not in seen]
    L = dict(L, cells=[summarize_cell(rows, metrics, c, CONFIG_K[L["config"]]) for c in cells])
    L["trace_rows"] = len(rows)
    L["sentinel_drift_log"] = drift(L)
    return L


def early_gate(cells):
    """First EARLY_CELLS measured cells after the warm-up (PILOT.md §2.1)."""
    first = [c for c in cells if c["kind"] != "warmup" and "wall_s" in c][:EARLY_CELLS]
    reasons = []
    if len(first) < EARLY_CELLS:
        reasons.append(f"only {len(first)} measured cells")
    for c in first:
        if c["invalid"] and "infeasible_preempted" not in c["invalid"]:
            reasons.append(f"{c['id']}: {','.join(c['invalid'])}")
    if sum(c["valid"] for c in first) < 2:
        reasons.append("fewer than 2 valid cells")
    return {"pass": not reasons, "reasons": reasons, "cells": [c["id"] for c in first]}


def launch_gate(L, smoke_gates, server_log, gpu):
    """End-of-launch gate (PILOT.md §2.2)."""
    reasons = []
    if "Using V2 Model Runner" not in server_log:
        reasons.append("no MRv2 log line")
    if "TRITON_ATTN" not in server_log:
        reasons.append("no TRITON_ATTN in log")
    for key in ("mrv2_used", "target_full_captured", "target_full_replayed"):
        if not smoke_gates.get(key):
            reasons.append(f"smoke gate {key} false")
    by_app = any(a.get("used_mib", 0) >= GPU_MIB for a in gpu.get("apps", []))
    by_mem = (gpu.get("memory_used_mib") or 0) >= GPU_MIB
    if not (by_app or by_mem):
        reasons.append(f"no GPU use >= {GPU_MIB} MiB observed (compute apps or memory.used)")
    run = [c for c in L["cells"] if c["kind"] in ("grid", "dense") and "wall_s" in c]
    feasible = [c for c in run if "infeasible_preempted" not in c["invalid"]]
    share = sum(c["valid"] for c in feasible) / len(feasible) if feasible else 0.0
    if share < LAUNCH_VALID_SHARE:
        reasons.append(f"valid share {share:.3f} < {LAUNCH_VALID_SHARE}")
    if L["sentinel_drift_log"] is None:
        reasons.append("sentinel drift not computable")
    cand = sum(c.get("steady_candidates", 0) for c in run)
    drop = sum(c.get("dropped_verify_width", 0) for c in run)
    if cand and drop / cand > MAX_DROP_SHARE:
        reasons.append(f"verify width mismatch {drop}/{cand}")
    return {"pass": not reasons, "reasons": reasons, "valid_share": round(share, 4),
            "cells_run": len(run), "cells_infeasible": len(run) - len(feasible),
            "gpu_use_basis": "compute_apps" if by_app else "memory_used" if by_mem else None}


def drift(L):
    s = [c for c in L["cells"] if c["kind"] == "sentinel" and c.get("valid")]
    if len(s) < 2:
        return None
    return round(float(np.log(s[-1]["step_period_ms"] / s[0]["step_period_ms"])), 4)


def table(launches):
    """config -> rep -> cell id -> summary, complete launches only."""
    t = {}
    for L in launches:
        if L.get("status") == "complete":
            t.setdefault(L["config"], {})[L["rep"]] = {c["id"]: c for c in L["cells"]}
    return t


def periods(t, config, cid):
    return [reps[cid]["step_period_ms"] for reps in (t.get(config, {})[r] for r in sorted(t.get(config, {})))
            if cid in reps and reps[cid].get("valid")]


def aa_spread(t, config="K3"):
    """Per-cell |log(period rep1 / period rep0)| for a configuration measured twice."""
    reps = t.get(config, {})
    if len(reps) < 2:
        return None
    r0, r1 = (reps[r] for r in sorted(reps)[:2])
    d = {cid: abs(float(np.log(r1[cid]["step_period_ms"] / r0[cid]["step_period_ms"])))
         for cid in r0 if cid in r1 and r0[cid].get("valid") and r1[cid].get("valid")
         and r0[cid]["kind"] in ("grid", "dense")}
    if not d:
        return None
    v = np.array(list(d.values()))
    return {"config": config, "cells": len(d), "median_abs_log_ratio": round(float(np.median(v)), 4),
            "max_abs_log_ratio": round(float(v.max()), 4), "per_cell": {k: round(x, 4) for k, x in d.items()}}


def marginal(t, cells, spread):
    """m_j = period(K=j+1)/period(K=j) - 1 per cell (K=0 is Z), acceptance-free bounds on c_j."""
    out = {}
    for j in range(0, 5):
        lo_cfg, hi_cfg = ("Z" if j == 0 else f"K{j}"), f"K{j + 1}"
        rows = {}
        for cid in cells:
            a, b = periods(t, lo_cfg, cid), periods(t, hi_cfg, cid)
            if not a or not b:
                continue
            m = float(np.mean(b)) / float(np.mean(a)) - 1
            lo, hi = m, (j + 1) * m
            where = "below" if hi < DECISION_BAND[0] else "above" if lo > DECISION_BAND[1] else "overlap"
            rows[cid] = {"m": round(m, 4), "c_bounds": [round(lo, 4), round(hi, 4)], "vs_band": where,
                         "exceeds_aa": None if spread is None else bool(abs(np.log1p(m)) > spread["max_abs_log_ratio"]),
                         "reps": [len(a), len(b)]}
        out[f"j{j}"] = rows
    return out


def design(bs, ctx, k, form):
    bs, ctx, k = (np.asarray(x, float) for x in (bs, ctx, k))
    one = np.ones_like(bs)
    if form == "M2":
        return np.column_stack([one, k, bs, k * bs, bs * ctx, k * bs * ctx])
    return np.column_stack([one, bs, bs * ctx])


def fit_models(t):
    """M1 per K and M2 over K1..K5, fitted on training cells, error on held-out cells."""
    pts = []
    for cfg, k in CONFIG_K.items():
        if not k:
            continue
        for reps in t.get(cfg, {}).values():
            for c in reps.values():
                if c["kind"] in ("grid", "dense") and c.get("valid"):
                    train = c["kind"] == "grid" and c["ctx"] in TRAIN_CTX and c["bs"] in TRAIN_BS
                    pts.append((k, c["bs"], c["ctx"] / 1024, c["step_period_ms"], train, c["id"]))
    if not pts:
        return None
    res = {}
    for form in ("M1", "M2"):
        coefs, held = {}, []
        groups = sorted({p[0] for p in pts}) if form == "M1" else [None]
        for g in groups:
            sel = [p for p in pts if g is None or p[0] == g]
            tr = [p for p in sel if p[4]]
            if len(tr) < design([1], [1], [1], form).shape[1]:
                continue
            X = design([p[1] for p in tr], [p[2] for p in tr], [p[0] for p in tr], form)
            coef, *_ = np.linalg.lstsq(X, np.array([p[3] for p in tr]), rcond=None)
            for p in sel:
                if not p[4]:
                    held.append((p, float((design([p[1]], [p[2]], [p[0]], form) @ coef)[0])))
            coefs[str(g)] = [round(float(x), 6) for x in coef]
        err = [abs(yh - p[3]) / p[3] for p, yh in held]
        res[form] = {"coef": coefs, "heldout_cells": len(held),
                     "heldout_rel_error_median": round(float(np.median(err)), 4) if err else None,
                     "heldout_rel_error_p90": round(float(np.quantile(err, 0.9)), 4) if err else None,
                     "heldout": [{"k": p[0], "id": p[5], "step_period_ms": p[3], "pred_ms": round(yh, 4)}
                                 for p, yh in held]}
    return res


def compare_stage1(t, configs=STAGE1_CONFIGS, metric="step_period_ms"):
    """Pairwise config differences per shape (STAGE1_ODD_PREREG.md §6).

    Only shapes valid for every config in both launches are compared. For configs a, b at a shape:
    d = log(mean b / mean a); each config's A/A spread = |log(rep1 / rep0)|. The difference counts
    ("exceeds_aa") only if |d| exceeds the larger spread and both launch pairs (rep0 vs rep0, rep1 vs
    rep1) have the same sign; opposite signs are "inconclusive"; otherwise "within_aa".
    """
    def val(c, rep, cid):
        x = t[c][rep][cid]
        return x.get(metric) if x.get("valid") else None

    full = [c for c in configs if len(t.get(c, {})) >= 2]
    reps = {c: sorted(t[c])[:2] for c in full}
    cids = set.intersection(*(set(t[c][r]) for c in full for r in reps[c])) if full else set()
    common = sorted(cid for cid in cids if all(val(c, r, cid) is not None for c in full for r in reps[c])
                    and t[full[0]][reps[full[0]][0]][cid]["kind"] in ("grid", "dense"))
    out = {"metric": metric, "configs_with_two_launches": full, "common_shapes": common,
           "not_compared": sorted(cids - set(common)), "pairs": {}}
    for i, a in enumerate(full):
        for b in full[i + 1:]:
            rows = {}
            for cid in common:
                va, vb = [val(a, r, cid) for r in reps[a]], [val(b, r, cid) for r in reps[b]]
                d = float(np.log(np.mean(vb) / np.mean(va)))
                spread = max(abs(float(np.log(va[1] / va[0]))), abs(float(np.log(vb[1] / vb[0]))))
                signs = {np.sign(np.log(vb[k] / va[k])) for k in (0, 1)}
                verdict = "inconclusive" if len(signs) > 1 else "exceeds_aa" if abs(d) > spread else "within_aa"
                rows[cid] = {"log_ratio": round(d, 4), "aa_spread": round(spread, 4), "verdict": verdict}
            out["pairs"][f"{b}_vs_{a}"] = rows
    return out


def report(out_dir):
    launches = [load_launch(out_dir, L) for L in map(json.loads, open(os.path.join(out_dir, "launches.jsonl")))]
    t = table(launches)
    ids = sorted({c["id"] for L in launches for c in L["cells"] if c["kind"] in ("grid", "dense")})
    spread = aa_spread(t)
    return {"launches": launches, "aa_spread_K3": spread,
            "z_over_off": {cid: round(float(np.mean(periods(t, "Z", cid)) / np.mean(periods(t, "OFF", cid))), 4)
                           for cid in ids if periods(t, "Z", cid) and periods(t, "OFF", cid)},
            "marginal": marginal(t, ids, spread), "cost_model": fit_models(t),
            "stage1": {m: compare_stage1(t, metric=m) for m in ("step_period_ms", "delivered_ms_per_token",
                                                                "ms_per_output_token")}}


if __name__ == "__main__":
    json.dump(report(sys.argv[1]), sys.stdout, indent=1)
