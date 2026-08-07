#!/usr/bin/env python3
"""
Aggregate ctx-uplift grid (pre-reg ctx_uplift/PREREGISTRATION.md §5/§7). Run at
desk after the dump; also run once right after the gate on the 4k anchor cells so
the anchor check is not decoration (tax lesson: Gate T became ornamental).

Reads (measure seeds only):
  RESULTS_DIR/grid/{arm}/ctx_{ctx}/measure_{0,1,2}.json
    -> output_throughput (primary), median_tpot_ms (secondary)

Prints per-cell throughput, K*(4k) / K*(32k), forcing loss both directions, the
§7 verdict (PASS/WEAK/NULL/CEILING), and the 4k anchor tax-band check. stdlib only.
"""
import json, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
CTXS = [4096, 32768]
KS = [0, 1, 2, 3, 5, 7]
SWEEP_CEIL = 7
BAR = 0.15


def load_cell(arm, ctx):
    d = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
    thr, tpot = [], []
    for s in range(3):
        f = d / f"measure_{s}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("output_throughput") is not None:
            thr.append(r["output_throughput"])
        if r.get("median_tpot_ms") is not None:
            tpot.append(r["median_tpot_ms"])
    if not thr:
        return None
    return {
        "n": len(thr),
        "thr": statistics.median(thr),
        "thr_std": statistics.stdev(thr) if len(thr) > 1 else 0.0,
        "tpot": statistics.median(tpot) if tpot else float("nan"),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def main():
    cells = {}                       # (K, ctx) -> cell
    for k in KS:
        for ctx in CTXS:
            r = load_cell(f"k{k}", ctx)
            if r:
                cells[(k, ctx)] = r
    nospec = {ctx: load_cell("no_spec", ctx) for ctx in CTXS}

    print(f"{'K':>3}{'ctx':>7}{'n':>3}{'out_tok/s':>16}{'TPOT_p50_ms':>14}")
    for k in KS:
        for ctx in CTXS:
            r = cells.get((k, ctx))
            if r:
                print(f"{k:>3}{ctx:>7}{r['n']:>3}"
                      f"{r['thr']:>9.1f}±{r['thr_std']:<6.1f}{r['tpot']:>14.2f}")
    for ctx in CTXS:
        r = nospec.get(ctx)
        if r:
            print(f"{'ns':>3}{ctx:>7}{r['n']:>3}"
                  f"{r['thr']:>9.1f}±{r['thr_std']:<6.1f}{r['tpot']:>14.2f}")

    # K*(c) = argmax throughput over the K sweep
    kstar = {}
    for ctx in CTXS:
        avail = [(k, cells[(k, ctx)]["thr"]) for k in KS if (k, ctx) in cells]
        if avail:
            kstar[ctx] = max(avail, key=lambda t: t[1])[0]
    print("\n=== K* (argmax output throughput) ===")
    for ctx in CTXS:
        print(f"  K*({ctx}) = {kstar.get(ctx, 'missing')}")

    if 4096 in kstar and 32768 in kstar and kstar[4096] != kstar[32768]:
        def thr(k, ctx):
            return cells[(k, ctx)]["thr"]
        L_4_to_32 = 1 - thr(kstar[4096], 32768) / thr(kstar[32768], 32768)
        L_32_to_4 = 1 - thr(kstar[32768], 4096) / thr(kstar[4096], 4096)
        maxL = max(L_4_to_32, L_32_to_4)
        print("\n=== Forcing loss (pre-reg §5) ===")
        print(f"  L(4k->32k) = {pct(L_4_to_32)}  (K*(4k)={kstar[4096]} imposed at 32k)")
        print(f"  L(32k->4k) = {pct(L_32_to_4)}  (K*(32k)={kstar[32768]} imposed at 4k)")
        ceiling = kstar[32768] == SWEEP_CEIL
        if maxL >= BAR:
            verdict = ("PASS + CEILING (K*(32k)=7: optimal may exceed sweep top; "
                       "forcing loss is a LOWER BOUND)" if ceiling else "PASS")
        else:
            verdict = "WEAK (K differs but forcing loss < 15%)"
        print(f"\n  VERDICT: {verdict}   (max forcing loss {pct(maxL)} vs bar {pct(BAR)})")
    elif 4096 in kstar and 32768 in kstar:
        print("\n  VERDICT: NULL (K*(4k) == K*(32k) — single K(b) suffices here). "
              "Report as-is; no post-hoc reframe (pre-reg §7).")
    else:
        print("\n  VERDICT: incomplete (missing K* at one ctx).")

    # 4k anchor: DSD-tier K=0 vs no_spec should reproduce the #49986 tax band
    a_k0, a_ns = cells.get((0, 4096)), nospec.get(4096)
    print("\n=== 4k anchor (pre-reg §9: no_spec vs DSD-K0 = #49986 tax band ~7-17%) ===")
    if a_k0 and a_ns:
        tax = (a_k0["tpot"] - a_ns["tpot"]) / a_ns["tpot"]
        ok = 0.05 <= tax <= 0.20
        print(f"  tax(K0 vs no_spec)@4k = {pct(tax)} -> "
              f"{'IN BAND (stack comparable)' if ok else 'OUT OF BAND (re-scope)'}")
    else:
        print("  (k0 or no_spec @4k missing)")


if __name__ == "__main__":
    main()
