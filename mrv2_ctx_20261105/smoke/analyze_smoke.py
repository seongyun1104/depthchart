"""Smoke gates from mrv2_trace files of one launch. Each check is reported on
its own; nothing is converted to another graph mode.

  python analyze_smoke.py <trace_dir> [--ctx-arm] > gates.json

Checks (all from observed calls, not from settings):
  mrv2_used                target graph manager (MRv2's ModelCudaGraphManager) captured
  target_full_captured     target FULL graphs > 0
  target_full_replayed     uniform-decode dispatches that chose FULL, and as many
                           target replays of those descriptors
  drafter_prefill_*        same for the drafter's first step
  drafter_decode_*         K-specialised drafter decode graphs: specialised to FULL,
                           and replayed; `unspecialised` if no specialise call came
  ctx_changes_k            (--ctx-arm) two contexts at one batch size chose different K
"""

import glob
import json
import sys
from collections import Counter, defaultdict


def load(trace_dir):
    rows = []
    for path in sorted(glob.glob(f"{trace_dir}/mrv2_trace_*.jsonl")):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["t_ns"])


def key(desc):
    return (desc["mode"], desc["tokens"], desc["reqs"], desc["q"], desc.get("k"))


def role_summary(rows, role):
    caps = [r for r in rows if r["ev"] == "capture" and r["role"] == role]
    disp = [r for r in rows if r["ev"] == "dispatch" and r["role"] == role]
    reps = Counter(key(r["desc"]) for r in rows if r["ev"] == "replay" and r["role"] == role)
    uniform = [r for r in disp if r["q"] is not None]
    full_disp = Counter(key(r["desc"]) for r in disp if r["desc"]["mode"] == "FULL")
    return {
        "captures": len(caps),
        "captured_by_mode": sum((Counter(c["graphs"]) for c in caps), Counter()),
        "capture_seconds": [c["seconds"] for c in caps],
        "capture_free_drop_bytes": [c["free_drop_bytes"] for c in caps],
        "dispatches": len(disp),
        "uniform_dispatches": len(uniform),
        "uniform_dispatch_modes": Counter(r["desc"]["mode"] for r in uniform),
        "mixed_dispatch_modes": Counter(r["desc"]["mode"] for r in disp if r["q"] is None),
        "full_dispatches": sum(full_disp.values()),
        "replays": sum(reps.values()),
        "replays_match_full_dispatches": reps == full_disp,
        "pw_calls": sum(1 for r in rows if r["ev"] == "pw" and r["role"] == role),
    }


def gates(rows, ctx_arm):
    t = role_summary(rows, "target")
    dp = role_summary(rows, "drafter_prefill")
    dd = role_summary(rows, "drafter_decode")
    spec = [r for r in rows if r["ev"] == "specialize"]
    spec_modes = defaultdict(Counter)
    for r in spec:
        spec_modes[r["k"]][r["desc"]["mode"]] += 1
    dd_rep = Counter(
        key(r["desc"]) for r in rows if r["ev"] == "replay" and r["role"] == "drafter_decode"
    )
    out = {
        "mrv2_used": t["captures"] > 0,
        "target_full_captured": t["captured_by_mode"].get("FULL", 0) > 0,
        "target_full_replayed": t["uniform_dispatch_modes"].get("FULL", 0) > 0
        and t["replays_match_full_dispatches"],
        "drafter_prefill_full_captured": dp["captured_by_mode"].get("FULL", 0) > 0,
        "drafter_prefill_full_replayed": dp["uniform_dispatch_modes"].get("FULL", 0) > 0
        and dp["replays_match_full_dispatches"],
        "drafter_decode": (
            "unspecialised"
            if not spec
            else {
                "specialised_full_by_k": {k: m.get("FULL", 0) for k, m in spec_modes.items()},
                "specialised_none_by_k": {k: m.get("NONE", 0) for k, m in spec_modes.items()},
                "replays": sum(dd_rep.values()),
            }
        ),
        "detail": {"target": t, "drafter_prefill": dp, "drafter_decode": dd},
    }
    if ctx_arm:
        by_bs = defaultdict(dict)
        for r in rows:
            if r["ev"] == "sched" and r["ctx"] is not None:
                by_bs[r["bs"]].setdefault(r["k"], []).append(r["ctx"])
        out["ctx_changes_k"] = any(len(ks) > 1 for ks in by_bs.values())
        out["k_by_bs"] = {
            bs: {k: [min(c), max(c)] for k, c in ks.items()} for bs, ks in by_bs.items()
        }
    sched = [r for r in rows if r["ev"] == "sched"]
    out["steps"] = {
        "n": len(sched),
        "k": Counter(r["k"] for r in sched),
        "skip_drafts": sum(1 for r in sched if r["skip_drafts"]),
        "bs_quantiles": quantiles([r["bs"] for r in sched]),
    }
    return out


def quantiles(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {p: s[min(len(s) - 1, int(p / 100 * len(s)))] for p in (10, 50, 90)}


def main():
    rows = load(sys.argv[1])
    json.dump(gates(rows, "--ctx-arm" in sys.argv), sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
