# Pre-registration — ctx_uplift 32k precision re-run (warmup↑)

Amendment to `PREREGISTRATION.md`. Written **before** the run. Purpose: test whether the
08-08 WEAK verdict at 32k is a *precision* artifact (first-measure-seed spec-warmup) or a
genuinely flat K-plateau.

## Motivation (from RESULTS.md diagnosis, not new data)

At 32k the spec arms' `measure_0` runs 14–17% below `measure_1/2` (K3 [630.8, 729.7, 764.3],
K5 [635.7, 762.4, 763.4], K7 [639.8, 770.7, 718.9]); `no_spec`/`k0` show ±0.3%. Medians
already drop the cold `measure_0`, so this inflates **stdev** only. 3 warmup rounds do not
reach *speculative* steady-state at 32k. Fix = more warmup, not more measure seeds.

## Single change

`master_bench_ctx.py::phase_grid`: warmup rounds 3 → 6 (`range(6)` → `range(9)`,
`is_measure = i >= 6`). **No other change.** Same build SHA, same models, same `b=11`,
`PYTHONHASHSEED=0`, same 32k prefix workload, `completion_tokens ≥ 256`, duration 120 s.

## Scope (cost containment)

- **ctx = 32768 only.** 4k is already solid (K3 vs K5 ≈ 1.5–2σ) — not re-run.
- **Arms = k3, k5, k7** (the contested plateau around the 32k argmax). k1/k2 dropped
  (not near argmax). no_spec/k0 optional (tax anchor is a *separate*, already-flagged
  question; not needed for the argmax-separation test).

## Pre-registered success criteria (declared before looking at results)

1. **Primary (did the fix work?):** after warmup↑, each spec arm's `measure_0` is within
   **±3%** of its `measure_{1,2}` median (matching the no_spec/k0 ±0.3% behavior). If NOT,
   the warmup fix failed → run is **inconclusive** on precision; do not re-interpret argmax.
2. **Secondary (the thesis test), only if Primary passes:** recompute K*(32k) argmax and its
   σ-separation from K3.
   - **STRONG:** K*(32k) ∈ {5,7}, separated from K3 by **≥2σ**. → ctx→K* upward direction is
     real at tightened precision; upgrade thesis evidence, update RFC #48627.
   - **WEAK-persists:** K3/K5/K7 still overlap within 1σ after the fix. → plateau is genuinely
     flat at 32k; noise was not the cause. → **concede** (RFC note: direction suggestive,
     not separated even at tightened precision).

## Decision mapping (bound before run)

| Outcome | Action |
|---------|--------|
| STRONG | RFC #48627 update: ctx→K* separation confirmed; thesis evidence upgraded |
| WEAK-persists | Concede honestly; close the "8k+ 최우선" precision path |
| Primary fails | Inconclusive; do not spend more without a different noise hypothesis |

No soft-selling: WEAK-persists is a fully acceptable outcome and does **not** justify a
third rental. Direction ≠ magnitude; a flat plateau at 32k is a real, publishable finding.

## Cost estimate (H100 NVL 94GB @ ~$2.31/hr)

3 arms × (6 warmup + 3 measure) at 32k only ≈ 1.5–2 hr incl. cold-start burn ≈ **$4–5**.
Within the $10.34 credit with margin. HF_TOKEN preset before launch (auth pull 3 min vs 25 min).
