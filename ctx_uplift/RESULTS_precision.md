# ctx_uplift 32k precision re-run — RESULTS

Pre-registration: `PREREGISTRATION_precision.md` (frozen `691f1fa`, before data).
Raw: `precision_raw/ctx_precision_2026-08-12.tgz` (9 measure JSONs + snapshots +
spec_config + server logs + build_sha).

## Setup

1× H100 NVL (driver 580.159.03, CUDA 13.0.88), vLLM `fd355781f71e` (PR #49652 head,
"Fix dynamic-SD draft decode capture"), Gemma-4-31B QAT FP8 target + MTP draft, KV=bf16,
`--max-model-len 33024`, `b=11`, greedy, `PYTHONHASHSEED=0`. Single change vs 08-08:
**warmup rounds 3 → 6** at **32k only**, arms **k3/k5/k7**.

**Environment delta (reproducibility):** `transformers==5.10.2` **must be pinned**. The
default `uv pip install vllm` now pulls transformers 5.15.0, which raises
`AmbiguousGlobalPerLayerAttributeError` on the Gemma-4 hybrid (per-layer `head_dim`) config
and prevents the server from starting. 5.10.2 = the version the model was saved with;
satisfies vLLM's `transformers >= 5.5.3`.

## K semantics (verified)

`spec_config` confirms `method='mtp', num_spec_tokens=3/5/7` — the MTP drafter emits exactly
K, no cap at K=5 or K=7.

## Data (32k, output_throughput tok/s)

| arm | measures | median | cv | accept | acc_len | tpot ms |
|-----|----------|--------|-----|--------|---------|---------|
| k3 | [801.2, 763.7, 719.1] | 763.7 | 4.4% | 0.643 | 2.93 | 11.77 |
| k5 | [791.2, 781.5, 818.4] | **791.2** | 2.0% | 0.513 | 3.56 | **11.43** |
| k7 | [733.0, 818.6, 739.2] | 739.2 | 5.3% | 0.394 | 3.76 | 11.66 |

- **K*(32k) = 5** (highest throughput, lowest TPOT, tightest cv). Reproduces the 08-08
  argmax direction (`K*(32k)=5 > K*(4k)=3`).
- Separation: **k5 vs k3 = +3.60% (+1.12σ)**, k7 vs k3 = −3.20% (−0.67σ).

## Verdict: WEAK-persists → concede (pre-reg decision map)

STRONG required K*(32k)∈{5,7} **and** ≥2σ separation from k3. We got K*=5 but only **+1.12σ** —
k3/k5/k7 still overlap. The optimal-K plateau at 32k is real in direction but **shallow and
not statistically separated** at this batch/precision. This confirms (does not upgrade) the
08-08 WEAK verdict.

## The warmup fix worked; the residual is not warmup

Primary check (measure_0 within ±3% of median(measure_1,2)) is met only for k5 (−1.1%);
k3 (+8.1%) and k7 (−5.9%) exceed ±3%. **But the artifact it targeted is resolved:** 08-08's
signature was a *systematic, one-sided* cold `measure_0` (spec arms −14 to −17% below
`measure_1/2`). With warmup=6 the deviations are now **scattered in both directions**
(+8 / −1 / −6%) — ordinary run-to-run variance, not a cold-start transient. So:

- More warmup would **not** tighten this further; the residual is inherent variance at
  b=11 / 32k. Per pre-reg, WEAK-persists **does not justify a third rental.**
- k5 is the cleanest arm (cv 2.0%) and wins on both throughput and TPOT, which is mild
  positive signal for K*=5 — but not at the pre-registered bar.

## Cross-reference

Consistent with 08-08 (`08a9b59`): K*(32k)=5, forcing loss then +4.51%, arms within ~1σ.
The precision run tightened k5 (cv 4.4%→2.0% class) and reproduced the direction, but the
k5–k3 gap did not grow (+3.6%, ~1σ). Direction ≠ magnitude: the 32k plateau is flat.

Cost: credit $10.34 → $1.24 (~$9.1; three provision retries — `/workspace` absent, PR head
moved past the pinned SHA, transformers 5.15.0 — each rebuilt/re-downloaded on billed time).
Lesson: bake the environment fixes into the provision recipe up front.
