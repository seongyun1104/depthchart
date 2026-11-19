"""Projection only (no server run): r2's measured step per (config, shape) minus gate 2's isolated per-step FP8 GEMM
saving of the #66852 (16, 32] config at that shape's target M. Not a measurement of the config on the server.

- saving(M) = Σ over the 6 Gemma shapes of calls/step × (t_arm0 − t_armP), mean of gate 2's two passes; calls/step
  as analyze_gemm.CALLS (50 sliding + 10 global layers, lm_head excluded: BF16, not this dispatch).
- M outside (16, 32]: the P arm selects main's kernel (gate 2 identity, bitwise-equal outputs) ⇒ saving 0 by
  construction, also at M gate 2 did not time (2, 4, 6, 8, 48, 192).
- "scaled": saving × r, r = r2's b16·1k K1 − Z step increase / gate 2's isolated arm-0 increase M 16 → 32
  (server observed less than the isolated GEMM sum once; one transition, one context — a reading, not a law).
- Acceptance unchanged (gate 2: P outputs bitwise equal to main at M ≤ 32); the drafter is assumed not to run this
  FP8 dispatch (unquantized assistant; not verified here).

  python m32_projection.py <gate2 evidence/costmap_out> <r2 analysis/cells.csv> > m32_projection.json
"""

import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gate2", "run_g2"))
sys.path.insert(0, os.path.join(HERE, ".."))
from analyze_gate2 import load  # noqa: E402
from analyze_gemm import CALLS  # noqa: E402

K = {"OFF": None, "Z": 0, "K1": 1, "K3": 3, "K5": 5}


def per_step_ms(t, arm, m):
    """Σ calls × µs over the Gemma shapes, mean of the two passes, in ms."""
    return statistics.mean(sum(c * t[(p, arm, k, n, m)] for (k, n), c in CALLS.items()) for p in (1, 2)) / 1e3


def main(gate2_out, cells_csv):
    t = load(gate2_out)
    timed = sorted({key[4] for key in t})
    saving = {m: per_step_ms(t, "0", m) - per_step_ms(t, "P", m) for m in timed}
    iso_16_32 = per_step_ms(t, "0", 32) - per_step_ms(t, "0", 16)
    cells = {}
    for r in csv.DictReader(open(cells_csv)):
        cells.setdefault((r["config"], int(r["bs"]), int(r["ctx"])), []).append(r)
    step = {key: statistics.mean(float(r["step_period_ms"]) for r in v) for key, v in cells.items()}
    server_16_32 = step[("K1", 16, 1024)] - step[("Z", 16, 1024)]
    ratio = server_16_32 / iso_16_32
    rows = []
    for (cfg, bs, ctx), v in sorted(cells.items(), key=lambda x: (x[0][1], x[0][2], list(K).index(x[0][0]))):
        k = K[cfg]
        m = bs * (k + 1) if k is not None else bs
        in_range = 16 < m <= 32
        if in_range and m not in saving:
            raise SystemExit(f"M {m} in (16, 32] not timed in gate 2")
        sv = saving[m] if in_range else 0.0
        acc = v[0]["acceptance_rate"]
        tok = 1.0 + (k or 0) * (float(acc) if acc not in ("", "None") else 0.0)
        s = step[(cfg, bs, ctx)]
        rows.append({"config": cfg, "k": "" if k is None else k, "bs": bs, "ctx": ctx, "shape": f"b{bs}·{ctx // 1024}k",
                     "m": m, "in_range": in_range, "acceptance": round((tok - 1) / k, 4) if k else "",
                     "step_r2_ms": round(s, 3), "saving_iso_ms": round(sv, 3), "saving_scaled_ms": round(sv * ratio, 3),
                     "step_proj_iso_ms": round(s - sv, 3), "step_proj_scaled_ms": round(s - sv * ratio, 3),
                     "ms_tok_r2": round(s / (bs * tok), 4), "ms_tok_proj_iso": round((s - sv) / (bs * tok), 4),
                     "ms_tok_proj_scaled": round((s - sv * ratio) / (bs * tok), 4)})
    meta = [{"label": "m32_projection.py", "value": open(os.path.abspath(__file__)).read(), "type": "code", "lang": "python"},
            {"label": "Input (gate 2)", "value": "gate2/run_g2/evidence/costmap_out time/p{1,2}_{0,P} (2026-11-18)"},
            {"label": "Input (r2)", "value": "costmap_20261107/run_20261111r2/analysis/cells.csv (2026-11-11)"},
            {"label": "saving_iso_ms by timed M", "value": json.dumps({m: round(v, 3) for m, v in saving.items()})},
            {"label": "transfer ratio r", "value": f"{server_16_32:.3f} / {iso_16_32:.3f} = {ratio:.3f}"}]
    return {"rows": rows, "meta": meta}


if __name__ == "__main__":
    json.dump(main(sys.argv[1], sys.argv[2]), sys.stdout, ensure_ascii=False, indent=1)
