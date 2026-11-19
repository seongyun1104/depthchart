"""Gate 2 analysis, exactly VERDICT_RULES_GATE2.md (approved 2026-11-17). For the N=2048 control shape the noise is that
shape's own arm-0 pass difference (the rules name ν for the Gemma sum only).
  python analyze_gate2.py evidence/costmap_out > RESULT_TABLES.txt
"""
import csv
import glob
import json
import os
import sys

GEMMA = [(5376, 16384), (5376, 20480), (8192, 5376), (16384, 5376), (5376, 43008), (21504, 5376)]
CTRL = (5376, 2048)
ARMS = ["W", "0", "1", "2", "3", "4", "P"]


def load(out):
    t = {}  # (pass, arm, K, N, M) -> µs
    for d in glob.glob(os.path.join(out, "time", "p*_*")):
        p, arm = os.path.basename(d)[1:].split("_")
        for f in glob.glob(os.path.join(d, "bench_fp8_res_n*_k*", "*.csv")):
            sub = os.path.basename(os.path.dirname(f))
            n, k = (int(x[1:]) for x in sub.replace("bench_fp8_res_", "").split("_"))
            for row in csv.reader(open(f)):
                if row[0] == "batch_size":
                    continue
                m, tf = int(float(row[0])), float(row[1])
                t[(int(p), arm, k, n, m)] = 2 * m * n * k / (tf * 1e12) * 1e6
    return t


def main(out):
    t = load(out)
    ms = sorted({key[4] for key in t})
    S = {(p, a, m): sum(t[(p, a, k, n, m)] for k, n in GEMMA) for p in (1, 2) for a in ARMS for m in ms}
    C = {(p, a, m): t[(p, a, CTRL[0], CTRL[1], m)] for p in (1, 2) for a in ARMS for m in ms}
    mean = lambda d, a, m: (d[(1, a, m)] + d[(2, a, m)]) / 2
    nu = {m: abs(S[(1, "0", m)] - S[(2, "0", m)]) for m in ms}
    nuc = {m: abs(C[(1, "0", m)] - C[(2, "0", m)]) for m in ms}

    print("# S(arm, M) = Σ 6 Gemma FP8 GEMM, µs, mean of two passes   (ν = |arm0 p1 − p2|)")
    print("M     " + "".join(f"{a:>9}" for a in ARMS) + "      ν")
    for m in ms:
        print(f"{m:<5} " + "".join(f"{mean(S, a, m):9.1f}" for a in ARMS) + f"  {nu[m]:6.2f}")
    print("\n# N=2048 control, µs, mean of two passes")
    for m in ms:
        print(f"{m:<5} " + "".join(f"{mean(C, a, m):9.2f}" for a in ARMS) + f"  {nuc[m]:6.2f}")

    # 1. build equivalence
    bad = [m for m in ms if abs(mean(S, "W", m) - mean(S, "0", m)) > max(2 * nu[m], 0.02 * mean(S, "0", m))]
    print("\n## 1. build equivalence W vs 0:", "PASS" if not bad else f"FAIL at M {bad}")
    for m in ms:
        print(f"  M {m}: (W−0)/0 = {100 * (mean(S, 'W', m) / mean(S, '0', m) - 1):+.2f}%")

    def ctrl_breach(arm, sum_ms, c_ms):
        br = [f"S@{m}" for m in sum_ms if abs(mean(S, arm, m) - mean(S, "0", m)) > max(2 * nu[m], 0.02 * mean(S, "0", m))]
        br += [f"N2048@{m}" for m in c_ms if abs(mean(C, arm, m) - mean(C, "0", m)) > max(2 * nuc[m], 0.02 * mean(C, "0", m))]
        return br

    def lowers(arm, at):
        rows = []
        for m in at:
            per = [S[(p, "0", m)] - S[(p, arm, m)] > max(2 * nu[m], 0.03 * S[(p, "0", m)]) for p in (1, 2)]
            rows.append((m, per))
        return all(all(per) for _, per in rows), rows

    print("\n## 2. M128 candidates (M 96 and 128, both passes; controls M≤64, M≥129, N=2048 all M)")
    corr = {a: [l for l in open(os.path.join(out, f"correct_{a}.jsonl")) if l.startswith("CORRECT")][-1].strip()
            for a in ARMS}
    verdict = {}
    for arm in "1234":
        ok, rows = lowers(arm, (96, 128))
        br = ctrl_breach(arm, [m for m in ms if m <= 64 or m >= 129], ms)
        det = "; ".join(f"M{m} Δ={100 * (1 - mean(S, arm, m) / mean(S, '0', m)):+.1f}% "
                        f"R={(mean(S, '0', m) - mean(S, arm, m)) / (mean(S, '0', m) - mean(S, '0', 64)):+.2f} "
                        f"passes={per}" for m, per in rows)
        v = "not attributable (control breach)" if br else ("lowers step" if ok else "does not lower step")
        verdict[arm] = v
        print(f"  arm {arm}: {v} | correctness {corr[arm]} | {det} | breaches {br}")

    print("\n## 3. P = #66852 (M 17, 24, 32, both passes; controls M 16, 33–64, ≥65, N=2048 at M≥33)")
    ok, rows = lowers("P", (17, 24, 32))
    br = ctrl_breach("P", [m for m in ms if m == 16 or m >= 33], [m for m in ms if m >= 33])
    det = "; ".join(f"M{m} Δ={100 * (1 - mean(S, 'P', m) / mean(S, '0', m)):+.1f}% "
                    f"R={(mean(S, '0', m) - mean(S, 'P', m)) / (mean(S, '0', m) - mean(S, '0', 16)):+.2f} "
                    f"passes={per}" for m, per in rows)
    v = "not attributable (control breach)" if br else ("lowers 16|17 step" if ok else "does not lower 16|17 step")
    verdict["P"] = v
    print(f"  arm P: {v} | correctness {corr['P']} | {det} | breaches {br}")

    print("\n## 4. per shape, mean of passes, µs (M 64 / 96 / 128 and 16 / 17 / 32)")
    for k, n in GEMMA:
        print(f"  K{k} N{n}: " + " | ".join(
            f"{a}: " + "/".join(f"{(t[(1, a, k, n, m)] + t[(2, a, k, n, m)]) / 2:.1f}" for m in (64, 96, 128, 16, 17, 32))
            for a in ARMS))
    json.dump({"verdict": verdict, "nu": nu}, open(os.path.join(out, "..", "verdict.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
