# Gemma-4-31B MTP × Spec × LMCache × DSD — Measurement Corpus (2026-07-08 → 07-14)

This file is the measurement corpus behind the vLLM RFC *Context-length-aware speculative token scheduling* (§Evidence link). We measured on Gemma-4-31B (FP8, hybrid sliding + global attention) with `prithivMLmods/gemma-4-31B-it-qat-FP8` as target and `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` (4 layers) as its MTP drafter, on a single H100 NVL 96GB, KV fp8, `max_model_len` 32768, TRITON_ATTN. Greedy sampling, fixed 200-token output via `ignore_eos`. Throughput is read from the `vllm:generation_tokens_total` delta at steady state; acceptance rate from spec-decode counter deltas; both cross-checked against Prometheus. 30s warmup on top of 120s canonical windows (or 45s for short-context / short-ramp cells, noted per section). Harness and configs are at `benchmarks/runner.py` + `benchmarks/configs/*.yaml`.

## 1. Batch-axis crossover (vLLM 0.23, short prompt ≈ 150 tok)

Reproduction of the classical short-context behavior — speculation pays until the batch saturates the compute ceiling.

| client concurrency | K=0 (tok/s) | K=1 | K=2 | K=3 | best gain |
|---|---|---|---|---|---|
| 30 | 1,400 | 2,200 | 2,200 | **2,500** | 1.79× (K=3) |
| 60 | 2,200 | 2,800 | — | **3,000** | 1.36× (K=3) |
| 128 | 3,413 | 3,413 | — | 3,413 | 1.00× (converged) |

TPOT p50 (ms): c30 22.0 / 12.5 / 12.5 / 10.0; c60 25.9 / 19.3 / — / 17.1; c128 41.1 / 31.8 / — / 32.2. Per-position acceptance for K=3 stays at 86 / 66 / 50% from c=30 through c=128 — acceptance is not what moves the gain; verify economics under the compute ceiling is. The K=0 arm was launched without a speculative-config; running with a schedule and a K=0 tier is a materially different case (§7). One note on methodology: without `ignore_eos`, benign divergence between K=0 and K>0 arms contaminates the output-token count and flips the conclusion — we hit two prior misjudgments before this was locked in.

## 2. Context-axis dose-response (vLLM 0.23, APC hit ≈ 98%, c = 256)

The heart of the RFC. At a fixed high batch, growing the shared-prefix context returns the speculative gain and grows it — up to a mid-context knee — after which the gain saturates.

| decode-time ctx (tok) | K=0 (tok/s) | K=3 (tok/s) | K3/K0 | K=0 TPOT (ms) | K=3 TPOT (ms) |
|---|---|---|---|---|---|
| ~460 | 3,107 | 3,640 | **1.17×** | 81.0 | 62.9 |
| ~970 | 2,320 | 2,897 | **1.25×** | 109.3 | 77.8 |
| ~1,990 | 2,110 | 2,915 | **1.38×** | 119.9 | **78.0** |
| ~4,096 | 1,768 | 2,397 | 1.36× [1] | 137.8 | 96.9 [2] |

[1] Third-run warm value. APC cache accumulation nudged the throughput ratio 1.28 → 1.32 → 1.36 across runs, so 1.36× is a conservative floor; the TPOT ratio (~1.4×, stable across 3 runs) is the more robust estimate.
[2] Single point. Characterized as onset of decline, not a decline curve; mapping ctx > 2k needs additional points.

The mechanism is directly visible in the TPOT column: as ctx doubles from 970 to 1,990, **K=3 TPOT stays flat (77.8 → 78.0 ms) while K=0 TPOT keeps climbing (109 → 120 ms)**. Long-context decode is memory-bandwidth-bound on the per-step KV read; verifying K drafted tokens amortizes that read across K+1 tokens. The saving grows with ctx, but not without bound. **The gain peaks around ctx ≈ 2k (1.38× throughput / 1.54× TPOT) and recedes to 1.36× / ~1.42× at 4k**, as K=3's own per-step cost starts rising past 2k (its flat 78 ms breaks to 97 ms), narrowing the gap to K=0. We did not decompose the K=3 rise; candidates are drafter SWA-window growth, draft-layer FLOPs, and target-side KV growth. Preemptions were zero and KV utilization stayed at or below 48% across all four cells, confirming pool-fit; the hit 0.0 → 0.98 config fix is what keeps the working set below the pool at c = 256, ctx = 4k.

A parallel run at c = 192, ctx ≈ 2k reads 2,060 → 2,978 tok/s = **1.45×** (TPOT 92.3 → 60.4). Cells at ctx ≈ 200 were excluded — the actual measured hit rate was 63% rather than the target 98%, so they are not comparable to the four points above.

We also ran a hit-axis probe at in = 2048, K = 3: hit 90 / 60 / 30% gave 1,976 / 910 / 660 tok/s with KV utilization 55 / 84 / 99%. The hit rate's role is capacity — a shared prefix is what keeps the working set below the pool — and low-hit long-context saturates and collapses. Caveat: the hit = 60 / 30% cells mix in an LMCache auto-fallback, so they are not pure APC numbers and should be labeled as such if cited. The conclusion (hit rate acts as a capacity condition) is unaffected.

## 3. Drafter lineage (vLLM 0.23, K = 3, c = 30)

Two drafter candidates on the same target model:

| drafter | acceptance rate | per-position | tok/s |
|---|---|---|---|
| QAT-matched | **67.7%** | 86 / 65 / 51 | **2,500** |
| original non-QAT | 51.6% | 78 / 47 / 30 | 2,200 |

Lineage match alone accounts for +14% throughput at this cell. This is why the RFC treats specific K palette values as deployment-specific — the drafter you pick moves acceptance rate materially, and the (B, ctx) surface shape is what generalizes, not the values.

## 4. LMCache (vLLM 0.23 + LMCache 0.5.1, MP connector)

The LMCache track is orthogonal to the RFC's claim but shaped the harness. Four configurations were tried:

| configuration | GPU KV pool | note |
|---|---|---|
| V1 connector alone | 123,520 (−65%) | hybrid manager force-disabled |
| V1 + MTP | crash | `KeyError: draft_model.layers.0.self_attn.attn` |
| **MP alone** | **364,354** | SupportsHMA active, 6 KV groups (5 sliding + 1 full) |
| MP + MTP | ~331k | 4-way coexistence (+FP8 KV) OK |

Two wiring requirements: netns sharing (ZMQ fixed at `localhost:5555`) and server GPU visibility (pointer via CUDA IPC). Consistency check: store → `reset_prefix_cache` → retrieve, with an external hit count of +8,704, confirms LMCache is serving; long-range integrity holds beyond the 1024 window. Under a working set of 500k tokens against a 331k pool we observed 45 recompute stampedes covered by 572 LMCache hits.

Two open items. Under nominal load at c = 64, APC gives 2,684 tok/s while LMCache-served gives 985 — root cause not isolated among serialization overhead, the 36% miss recompute, and GPU contention; `max-gpu-workers 16` had no effect, and single-GPU affinity serialization is our current guess. Under overload, `scheduler assert req_id in self.requests` fires and the engine dies — an upstream defect.

## 5. vLLM 0.24 migration smoke (2026-07-10)

Gemma-4-31B × MTP bring-up on 0.24 completes with zero argument changes, KV pool 330,966 (matching 0.23 within noise), TRITON backend retained. The prior 0.24 failure on RTX 5090 12B is therefore isolated to sm_120 specifics.

## 6. MTP × DSD tier switching (vLLM 0.24, `[[1,64,3],[65,128,1],[129,512,0]]`)

Dynamic speculative decoding paired with an MTP drafter is not documented as tested in vLLM — the DSD docs cite Eagle / E3 only — so this section is that datapoint. At startup: `WARNING vllm.py:767 "Dynamic speculative decoding ... Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE"`. The `VLLM_USE_V2_MODEL_RUNNER` env var is not required.

| client c | engine running peak | tier (aggregated) | tok/s | TPOT p50 (ms) |
|---|---|---|---|---|
| 30 | — | **K=3** (drafts/step 3.0) | 2,533 | 10.3 |
| 100 | — | **K=1** (runtime switch) | 3,089 | 31.5 |
| 192 | 192 | K=1 mixed (AR 93.3%) [3] | 2,870 | 70.5 |
| 224 | 224 | K=0 | 2,330 [4] | 75.2 |
| 256 | 256 | K=1 mixed (AR 84.1%) [3] | 2,605 | 81.2 |
| 320 | 320 | K=0 | 1,740 [4] | 110.5 |
| 400 | 400 | **K=0** (spec 0) | 1,880 [4] | 129.5 |

[3] The mixed-cell AR (93.3% / 84.1%) is read from a ~2%-volume sample of drafting steps and is therefore noise; do not cite as an acceptance measurement.
[4] The high-concurrency K = 0 cells sit at 51–68% of the 0.23 c = 128 no-spec ceiling (3,413) and are non-monotonic (c320 < c400), with a derived-value mismatch at c224 (75.2 ms → 2,979 tok/s derived vs. 2,330 measured) — the signature of a contaminated 45s window. Suspected anomalies; adjudicated in §7.

MTP × DSD tier switching works end-to-end. Some drafting still appears at running ≥ 129, meaning the DSD index (per-step scheduled request count) fluctuates step-to-step through admission ramps and momentarily passes through lower tiers. Boundary behavior is therefore probabilistic — a production deployment should place tier boundaries outside the running-distribution tail and add hysteresis. By volume the picture is different from the aggregated tier column: the draft/gen ratio during CAL is 0.02 (compare ~0.52 when K = 1 dominates), so at c ≥ 192 the steady state is mostly K = 0 with drafting confined to ~2% of ramp steps. "K = 1 held" is an aggregation illusion — only the K of the steps that actually drafted. As a cross-check, the pure-K=3 window in §8 reads draft/gen = 0.99.

## 7. Adjudicating the high-concurrency K = 0 tax (2026-07-13)

To isolate the anomalous K = 0 cells at c ≥ 192, we ran the 120s canonical comparison at c = 256, referenced against the 0.23 c = 128 ceiling of 3,413 tok/s:

| c = 256, 120s canonical | tok/s | TPOT p50 | TTFT p50 / p99 | completions |
|---|---|---|---|---|
| no-spec @ 0.24 | **3,413** | 73.7 ms | 0.28 s / 4.7 s | 2,048 |
| DSD-K0 @ 0.24 | **2,560** | 78.6 ms | **2.0 s / 10.8 s** | 1,536 |

The no-spec arm matches the 0.23 c = 128 ceiling (3,413) exactly, so there is no 0.24 regression, the ceiling extends flat to c = 256, and the wave-artifact hypothesis is ruled out. The DSD-K0 arm reads −25% against no-spec, with the tax concentrated in TTFT (only +6.7% TPOT) — a prefill / scheduling cost, not a decode cost. The lower 45s readings from §6 were not window contamination; the tax is the actual cause (the 45s cell at 2,605 lines up with the 120s cell at 2,560).

Deployment reading: for an always-on high-batch server, a K = 0 tier is not free. Drop the spec config entirely for that regime. DSD as a whole earns its keep for c < 128.

We did not decompose the tax into (i) PIECEWISE cost at high batch, (ii) drafter prefill obligation, or (iii) tier-decision overhead. The 0.23 static-K3 result at c = 128 (3,413) shows the drafter itself is not the source of the tax — DSD mode is what is specific. A discriminator experiment using batch table `[[1,512,0]]` would isolate the pure infra tax.

This datapoint is the source of the RFC's §Motivation note that K = 0 tiers do not make speculation free, with the penalty concentrating in TTFT — consistent with the input-preparation hotspots reported in #47277.

## 8. PIECEWISE tax verification and short-context boundary probe

DSD forces PIECEWISE at startup (see §6 warning). We compared DSD-PIECEWISE against static FULL+PIECEWISE at c = 30, K = 3:

| | DSD (PIECEWISE) | static (FULL + PIECEWISE) |
|---|---|---|
| tok/s | 2,533 | 2,533 |
| TPOT p50 | 10.3 ms | 10.5 ms |
| completions / AR | 570 / 68.1% | 570 / 68.2% |

The PIECEWISE-only path carries no measurable tax at low concurrency — the K = 0 tax in §7 is not this.

Short-context boundary probe (B = 61 → 128): c = 90 K=0/K=3 = 2,445 / **3,000** (1.23×, TTFT p99 960 ms); c = 110 K=0/K=1/K=3 = 2,892 / 3,185 / **3,300** (K = 3 best, 1.14× over K = 0, TTFT p99 796 ms); c = 128 = crossover (K = 0 = K = 3 = 3,413, TTFT p99 4,044 ms). The tier `[1,110,3]` is supported by this run — c = 110 K = 3 TTFT p99 = 796 ms is healthy, and the feared tail explosion lives between 110 and 128. Even at short context, K = 3 beats K = 1 at c = 110 (3,300 vs. 3,185, position-2 acceptance 53.7%), so the standard `[65,128,1]` middle tier is over-conservative and can be raised to K = 3 up to just before c = 128. TPOT also favors K = 3 across this range (27–30 ms vs. K = 0's 35–38). This is the RFC's secondary observation about coarse batch-only tables — the batch axis is scheduled less finely than the data supports.

At ctx = 4k, K = 3 TPOT breaks from 78 to 97 ms — the flat curve is gone. K = 0 at ctx = 4k measured 137.8 ms, giving 1.42× (TPOT) / 1.36× (throughput) — the amortization knee is confirmed. The current "deployable" basis is c ≤ 100; the high-concurrency tier verdict rests on §7.

## Unmeasured

- Quantitative quality gate. Engine consistency was checked on 2026-07-14: greedy K = 3 vs. K = 0 at concurrency 1, control 30 / 30 deterministic, K-switch 0 / 30 bit-identical — but all divergences were benign reword (FP8 KV argmax flip, no corruption). Bit-identity is the wrong gate; scored equivalence is the deployment pipeline's domain and is not run here.
- LMCache-served × spec combined throughput, after the restore-bottleneck isolation.
- LMCache −63% pool root cause; multi-GPU restore parallelism.
