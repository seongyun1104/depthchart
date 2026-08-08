# Context-length uplift sweep — results (2026-08-07)

Companion to `PREREGISTRATION.md` (commit `825abd5`, frozen before data),
`master_bench_ctx.py`, `aggregate_ctx.py`, `RUNBOOK.md`. Raw under `raw/results/`
(per-arm server metrics, `spec_config_*`, `kv_capacity_*`, grid JSON + snapshots,
`AGGREGATE_OUTPUT.txt`).

**Verdict: WEAK** (pre-reg §7 matrix). `K*(4k) ≠ K*(32k)` in the predicted
direction (optimal K rises with context), but the forcing loss is far below the
pre-registered 15% bar, and the 4k anchor is out of band. Reported straight; the
pre-registered success criterion was **not** met.

## Setup

- Target `prithivMLmods/gemma-4-31B-it-qat-FP8` (hybrid attn) + MTP drafter
  `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`. 1× H100 NVL 94 GB.
- Build: PR #49652 @ `fd355781` (same commit as `tax_decomposition`), run in **V1
  mode** (no `VLLM_USE_V2_MODEL_RUNNER`). `--gpu-memory-utilization 0.90`.
- **KV cache bf16** (fp8 prohibited on this stack: fp8 KV routes to FlashInfer,
  which raises `NotImplementedError` on SM90 + sliding-window layers —
  flashinfer-ai/flashinfer#3578 — present in this build; verified in source, not
  just assumed).
- **`--max-model-len 34816`** (pre-reg §9 said 33024 = 32768+256; that was too
  tight by BOS + prefix_repetition suffix, every 32k request got HTTP 400. Raised;
  ctx endpoints and per-request KV unchanged. Deviation recorded here.)
- `b = 11`, fixed across all arms/ctx. Gate-confirmed: the engine's naive
  "Maximum concurrency 2.9x" line is the no-sliding-credit worst case
  (`pool_tokens / max_model_len`); real per-32k-request KV ≈ 3.46 GB against
  ~47 GiB pool → ~13 reachable. OOM smoke at concurrency 11, 32k: 24/24 requests
  succeeded, **0 preemptions**. See `raw/results/kv_capacity_*`.
- `n = 3` measure seeds (3 warmup discarded) per cell, single session, single
  build, no order-reversal (justified per pre-reg §6.4; end-of-session spot check
  bounds drift, below).

## K semantics (verified from runtime draft counts, not just config repr)

The `SpeculativeConfig` repr omits `num_speculative_tokens_per_batch_size`, so K
was verified from `spec_decode_num_draft_tokens / num_drafts` per step (ctx4096,
measure_0):

| arm | drafts/step | meaning |
|---|---|---|
| k0 | **0** | DSD-tier K=0 (schedule `[[1,512,0]]` applied; 0 draft tokens) |
| k1..k7 | **1, 2, 3, 5, 7** (exact) | fixed K; MTP drafter emits exactly K, no cap |

So the sweep spans K∈{0,1,2,3,5,7} as intended; K7 is a true interior point (not
a drafter cap), and `K*(32k)=5 < 7` is a real optimum, not a ceiling artifact
(pre-reg CEILING outcome does not apply).

## Throughput (median output tok/s at b=11; ± = stdev over 3 seeds)

| K | 4k out tok/s | 4k TPOT p50 | 32k out tok/s | 32k TPOT p50 |
|---|---|---|---|---|
| 0 (DSD) | 444.6 ± 2.3 | 24.03 | 326.1 ± 0.3 | 31.70 |
| 1 | 776.6 ± 16.6 | 12.46 | 531.8 ± 32.4 | 17.91 |
| 2 | 959.5 ± 30.8 | 9.43 | 676.2 ± 14.2 | 13.35 |
| **3** | **1050.1 ± 27.6** | 7.95 | 729.7 ± 69.3 | 11.16 |
| **5** | 1002.7 ± 20.7 | 8.16 | **762.4 ± 73.5** | 11.03 |
| 7 | 943.3 ± 11.5 | 8.52 | 718.9 ± 65.9 | 12.23 |
| no_spec | 548.6 ± 2.4 | 19.66 | 358.4 ± 1.5 | 29.05 |

Speculation gives ~1.9× (4k, K3 vs no_spec) and ~2.1× (32k, K5 vs no_spec).

## Result

- `K*(4k) = 3`, `K*(32k) = 5` — optimal K is higher at long context (thesis
  direction; strong form "K ceiling rises with ctx" is directionally supported,
  3 → 5).
- **Forcing loss:** `L(4k→32k) = +4.29%`, `L(32k→4k) = +4.51%`. Max **4.51%**,
  far below the pre-registered **15%** bar → **WEAK**.

### Honest caveats (why this is WEAK, not PASS)

1. **The 32k argmax is inside the noise.** At 32k the per-seed stdev is ~10%
   (K3 ±69, K5 ±74, K7 ±66 on ~730–760 tok/s); K3, K5, K7 overlap within one
   stdev. So `K*(32k)=5` is not statistically separated from K3/K7, and the
   4.5% forcing loss is comparable to the 32k measurement noise. The contradiction
   is *directionally suggestive, not resolved* at this batch/precision. (4k is
   tighter: K3 vs K5 is ~1.5–2σ, so `K*(4k)=3` is reasonably solid.)
   The 32k stdev is **largely a systematic first-measure-seed effect, not random**:
   the speculating arms' `measure_0` runs ~14–17% below `measure_1/2`
   (K3 [630.8, 729.7, 764.3], K5 [635.7, 762.4, 763.4], K7 [639.8, 770.7, 718.9]),
   while `no_spec`/`k0` show none (±0.3%) — i.e. 3 warmup rounds don't reach
   *speculative* steady-state at 32k. The reported medians already exclude the cold
   `measure_0`; it only inflates the stdev. Tightening 32k therefore needs **more
   warmup**, not more measure seeds.
2. **A more robust framing than the argmax:** high K is systematically *less
   penalized* at 32k. K7 sits 10.2% below its ctx-peak at 4k but only 5.7% below
   at 32k; K5 is 4.5% below peak at 4k but is the peak at 32k. The optimal-K
   plateau shifts up / widens with context — consistent with the thesis, modest
   in size.
3. **4k anchor OUT OF BAND.** `tax(K0 vs no_spec)@4k = +22.19%` (TPOT), vs the
   pre-registered #49986 band of ~7–17%. Per pre-reg §9 this means the stack is
   **not** established as comparable to the #49986 tax measurement, so no
   same-stack claim to that number is made. Likely cause (post-hoc, not
   pre-registered): the #49986 7.29% was at concurrency **192**; this sweep is at
   **b=11**. The K=0 spec-path bookkeeping tax amortizes over far fewer tokens at
   low batch, so a larger relative tax at b=11 is mechanistically expected — but
   this is a post-hoc explanation and does not rescue the comparability check.

### Controls

- **Spot re-check (drift):** spot re-ran k0 at both ctx at end of session.
  `drift = +1.10%` (4k), `+0.46%` (32k) — within the pre-reg drift bound (order-
  reversal skip justified by ≤1.72% + 0.39%). The WEAK result is not a drift
  artifact.
- All 8 arms × 2 ctx ran `rc=0`, 0 preemptions, 0 HTTP-400 (after the mml fix).

## Acceptance rate (from measure snapshots) — mechanism, with a workload caveat

Per-step mean accepted length `L = accepted/drafts` and `AR = accepted/draft_tokens (= L/K)`:

| K | AR@4k | AR@32k | L@4k | L@32k |
|---|---|---|---|---|
| 1 | 0.823 | 0.844 | 0.82 | 0.84 |
| 3 | 0.595 | 0.655 | 1.78 | 1.97 |
| 5 | 0.426 | 0.495 | 2.13 | 2.48 |
| 7 | 0.313 | 0.387 | 2.19 | 2.71 |

AR is higher at 32k at every K, consistent with `K*(32k) > K*(4k)` (higher K stays
acceptable longer at long context). **Mapping verified from raw:** `draft_tokens =
K × drafts` exactly (ratios 1.000/2.000/3.000/5.000/7.000 per arm) and `L ≤ K`, so the
arm→K labeling and the AR denominator are correct. (Per-position acceptance counters
were **not** captured in these snapshots, so a per-position cumulative-product
cross-check is not runnable on this raw; the check here is the exact K-mapping + `L ≤ K`.)

**Workload caveat — do not generalize the AR trend.** The context is synthetic
`prefix_repetition` (a shared/repeated prefix padded to length), which is easier to
draft than natural text, so the long-context AR uplift is **likely workload-inflated**
and needs natural-document validation before "AR rises with context" is claimed as a
general property. On natural 32k documents it could move the other way (draft staleness
/ the TTS–BudgetDraft axis). The table is reported as an observation consistent with the
K* shift, not as an established mechanism.

## What this does and does not say

- **Does:** first ctx≥4k evidence on this stack; K-semantics clean; the optimal K
  and the optimal-K plateau move up with context, in the direction MagicDec /
  the RFC #48627 thesis predict.
- **Does not:** meet the pre-registered 15% forcing-loss bar; resolve `K*(32k)`
  out of the ~10% 32k noise; establish comparability to the #49986 tax number.
  No PASS claim.

## Next (options, not run)

- Pre-reg §7 WEAK path: a second fixed batch `b'`. MagicDec predicts the low-batch
  long-ctx regime is where SD is the lever; a lower `b'` is *more* memory-bound and
  could sharpen (or concede) the contradiction. Would need a fresh rental.
- Tighter 32k precision (more seeds / longer windows) to pull `K*(32k)` out of
  noise before any stronger claim.
- Orthogonal, requested by @CZT0 on #49652: `use_dynamic_decode_query_len`
  True/False at K>0 to isolate the spec-path bookkeeping component.
