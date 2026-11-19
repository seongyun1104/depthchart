"""Read the harness CSVs (TFLOP/s) back into ms and apply the verdict rules fixed in VERDICT_RULES.md.
  python analyze_gemm.py <costmap_out> > gemm_verdict.json
"""
import csv
import glob
import json
import os
import sys

FP8 = "fp8-channel-w-token-a-noquant"   # this model's scheme, GEMM only
FP8_Q = "fp8-channel-w-token-a"         # same, with the per-token activation quant in the timed region
BF16 = "torch-bf16"
SLIDING, GLOBAL = 50, 10
# (K, N) as the harness labels them; per-step call counts for Gemma-4 31B (60 layers + lm_head)
CALLS = {(5376, 16384): SLIDING, (5376, 20480): GLOBAL, (8192, 5376): SLIDING, (16384, 5376): GLOBAL,
         (5376, 43008): SLIDING + GLOBAL, (21504, 5376): SLIDING + GLOBAL}
LM_HEAD = (5376, 262144)
# observed step increases at 1k context (r2 medians, ms) and the transition's M values
OBSERVED = {"b16 Z->K1 (16->32)": (16, 32, 2.33), "b8 Z->K1 (8->16)": (8, 16, 0.02),
            "b32 Z->K1 (32->64)": (32, 64, 0.04), "b16 K3->K5 (64->96)": (64, 96, 5.30),
            "b32 K1->K3 (64->128)": (64, 128, 7.10), "b32 K3->K5 (128->192)": (128, 192, 11.73)}


def load(out_dir):
    t = {}  # (K, N) -> provider -> {M: ms}
    for d in glob.glob(os.path.join(out_dir, "bench_fp8_res_n*_k*")):
        n, k = (int(x[1:]) for x in os.path.basename(d).split("_")[3:5])
        for p in glob.glob(os.path.join(d, "*.csv")):
            for row in csv.DictReader(open(p)):
                m = int(float(row["batch_size"]))
                for col, v in row.items():
                    prov = col.split(" (")[0]  # harness labels columns "name (TFLOP/s (larger is better))"
                    if prov in (FP8, FP8_Q, BF16) and v:
                        tflops = float(v)
                        t.setdefault((k, n), {}).setdefault(prov, {})[m] = 2 * m * n * k / (tflops * 1e12) * 1e3
    return t


def drops(ms, tol=0.02):
    """Points where a larger M is cheaper than a smaller one by more than tol, and what a running max (as in
    adaptive verification's build_table, np.maximum.accumulate) would overwrite there."""
    xs = sorted(ms)
    out, top = [], 0.0
    for a, b in zip(xs, xs[1:]):
        top = max(top, ms[a])
        if ms[b] < ms[a] * (1 - tol):
            out.append({"from_M": a, "to_M": b, "ms": [round(ms[a], 4), round(ms[b], 4)],
                        "running_max_overwrites_by": round(top / ms[b] - 1, 3)})
    return out


def step_ms(t, m, prov):
    s = sum(c * t[kn][prov][m] for kn, c in CALLS.items())
    return s + t[LM_HEAD][BF16][m]


def main(out_dir):
    t = load(out_dir)
    per_shape = {}
    for kn, provs in sorted(t.items()):
        ms = provs.get(FP8 if kn != LM_HEAD else BF16, {})
        if not ms:
            continue
        r = lambda a, b: round(ms[b] / ms[a], 3) if a in ms and b in ms else None
        per_shape[f"K{kn[0]}_N{kn[1]}"] = {
            "ms_at": {m: round(ms[m], 4) for m in (1, 8, 16, 17, 32, 64, 65, 96, 128, 129, 192) if m in ms},
            "ratio_17_over_16": r(16, 17), "ratio_32_over_16": r(16, 32), "ratio_64_over_32": r(32, 64),
            "ratio_65_over_64": r(64, 65), "ratio_129_over_128": r(128, 129),
            "split_2x16_over_32": round(2 * ms[16] / ms[32], 3) if 16 in ms and 32 in ms else None,
            "split_4x16_over_64": round(4 * ms[16] / ms[64], 3) if 16 in ms and 64 in ms else None,
            "ratio_96_over_64": r(64, 96), "ratio_128_over_64": r(64, 128),
            "drops": drops(ms)}
    model = {}
    if all(kn in t for kn in CALLS) and LM_HEAD in t:
        for prov in (FP8, FP8_Q):
            model[prov] = {name: {"gemm_delta_ms": round(step_ms(t, b, prov) - step_ms(t, a, prov), 3),
                                  "observed_step_delta_ms": obs}
                           for name, (a, b, obs) in OBSERVED.items()}
        ms_all = sorted(t[LM_HEAD][BF16])
        curve = {m: step_ms(t, m, FP8) for m in ms_all}
        model["gemm_step_ms_at"] = {m: round(v, 3) for m, v in curve.items()}
        model["gemm_step_drops"] = drops(curve)
    print(json.dumps({"per_shape": per_shape, "model_step": model}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
