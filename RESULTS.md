# Gemma-4-31B MTP × Spec × LMCache × DSD — Measurement Corpus (2026-07-08 ~ 07-10)

> Environment: single H100 NVL 96GB, target `prithivMLmods/gemma-4-31B-it-qat-FP8` + drafter `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` (4 layers), KV fp8, max_len 32768, TRITON_ATTN.
> Protocol: temp=0, `ignore_eos` with fixed real-token count (200 output), throughput = `vllm:generation_tokens_total` delta at steady state, 30s warmup / 120s window (§2·3) or 45s (§5·6, short-context / short ramps), AR = spec-counter delta, Prometheus cross-check.
> Harness: `benchmarks/runner.py` + `benchmarks/configs/*.yaml`.

## 1. Crossover — batch axis (vLLM 0.23, short prompt ~150 tok)

| c | K=0 (no-spec) | K=1 | K=2 | K=3 | best/K0 |
|---|---|---|---|---|---|
| 30 | 1,400 | 2,200 | 2,200 | **2,500** | 1.79x |
| 60 | 2,200 | 2,800 | — | **3,000** | 1.36x |
| 128 | 3,413 | 3,413 | — | 3,413 | **1.00x (converged = compute ceiling)** |

- TPOT p50 (ms): c30 22.0/12.5/12.5/10.0 · c60 25.9/19.3/—/17.1 · c128 41.1/31.8/—/32.2
- per-position (K=3): 86/66/50% — invariant from c30 → 128. The K=0 arm ships without a speculative-config (footnote).
- Methodology: without `ignore_eos`, benign divergence contaminates output length and flips the conclusion (two prior misjudgments before this was locked in).

## 2. Dose-response — context axis (0.23, hit 98–99% matched, c=256)

| decode-time ctx (tok) | K=0 | K=3 | K3/K0 | K0 TPOT | K3 TPOT |
|---|---|---|---|---|---|
| ~460 | 3,107 | 3,640 | **1.17x** | 81.0 | 62.9 |
| ~970 | 2,320 | 2,897 | **1.25x** | 109.3 | 77.8 |
| ~1,990 | 2,110 | 2,915 | **1.38x** | 119.9 | **78.0** |
| ~4,096 (7/14) | 1,768 | 2,397 | ~1.36x | 137.8 | 96.9 |

- **Key evidence: K=3 TPOT stays flat (970 → 1990: 77.8 → 78.0 ms) while K=0 keeps climbing (109 → 120 ms)** = KV-read amortization.
- **★ V4 complete (7/14, hit 98% c=256, 3 runs each): amortization has a knee.** K=3 TPOT is **perfectly flat at 78 ms up to ctx = 2k, then breaks to 97 ms at 4k (+24%)**; K=0 keeps climbing (120 → 137). Throughput-ratio curve = 1.17 → 1.25 → 1.38 → **1.36** (rise stalls / slightly retreats at 4k). **The TPOT ratio 137/97 = 1.41× is the more robust estimate** (stable across 3 runs; the throughput ratio nudges 1.28 → 1.32 → 1.36 with APC warming — that's the caveat). Mechanism = spec's own long-context decode cost climbs (SWA window / drafter FLOPs eat into the KV-read saving), capping the relative gain. **Narrative: amortization scales with ctx but not unboundedly — knee around ~2k, then saturates near ~1.4× (honest boundary).** preempt = 0, kv ≤ 48% → pool-fit (the hit 0.0 → 0.98 config fix is the reason this holds).
- Parallel run at c = 192: K0 2,060 / K3 2,978 = **1.45x** (TPOT 92.3 → 60.4). 0 preemptions per cell.
- in ≈ 200 cells excluded: hit mismatch (63%).
- Hit-axis probe (in = 2048, K = 3): hit 90 / 60 / 30 → 1,976 / 910 / 660 tok/s, kv 55 / 84 / 99% — hit's role is capacity (pool fit); low-hit long-context saturates and collapses. ⚠ **The hit = 60/30 cells mix in an LMCache auto-fallback (not pure APC)** — labels required if cited. The conclusion (hit = capacity condition) still stands.

## 3. Drafter lineage (0.23, K=3 c=30)

| drafter | AR | per-position | tok/s |
|---|---|---|---|
| QAT-matched (Cell A) | **67.7%** | 86/65/51 | **2,500** |
| original non-QAT (Cell B) | 51.6% | 78/47/30 | 2,200 |

→ Lineage match alone gives +14% throughput.

## 4. LMCache (0.23 + LMCache 0.5.1 MP connector)

| configuration | GPU KV pool | note |
|---|---|---|
| V1 connector alone | 123,520 (−65%) | hybrid manager force-disabled |
| V1 + MTP | crash | `KeyError: draft_model.layers.0.self_attn.attn` |
| **MP alone** | **364,354** | SupportsHMA active, 6 KV groups (5 sliding + 1 full) |
| MP + MTP | ~331k | 4-way coexistence (+FP8 KV) OK |

- Two wiring requirements: netns sharing (ZMQ fixed at `localhost:5555`) + server GPU visibility (ptr = CUDA IPC).
- Consistency: store → `reset_prefix_cache` → retrieve; external hit +8,704 confirms LMCache is serving. Long-range integrity holds (cited beyond the 1024 window).
- Capacity: pool 250 (working set 500k > 331k) → 45 recompute stampedes → 572 LMCache hits.
- Throughput: APC 2,684 vs. LMCache-served 985 tok/s at c = 64 under nominal load. **Root cause not isolated**: serialization + 36% miss recompute + GPU contention. `max-gpu-workers 16` had no effect — presumed single-GPU affinity serialization.
- Stability: under overload, `scheduler assert req_id in self.requests` → EngineDead (upstream defect).

## 5. vLLM 0.24 migration (7/10)

- Test ①: 0.24 × Gemma4-31B × MTP bring-up completes, argument change = 0, KV pool 330,966 (≈ 0.23), TRITON retained → **the RTX 5090 12B 0.24 failure is isolated to sm_120 specifics**.

## 6. MTP × DSD (0.24, `num_speculative_tokens_per_batch_size:[[1,64,3],[65,128,1],[129,512,0]]`)

Active recognition at startup: `WARNING vllm.py:767 "Dynamic speculative decoding ... Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE"`. The `VLLM_USE_V2_MODEL_RUNNER` env var is not required.

| client c | engine running peak | tier (aggregated) | tok/s | TPOT p50 |
|---|---|---|---|---|
| 30 | — | **K=3** (drafts/step 3.0) | 2,533 | 10.3 |
| 100 | — | **K=1** (runtime switch) | 3,089 | 31.5 |
| 192 | 192 | K=1 mixed (AR 93.3%) | 2,870 | 70.5 |
| 224 | 224 | K=0 | 2,330 | 75.2 |
| 256 | 256 | K=1 mixed (AR 84.1%) | 2,605 | 81.2 |
| 320 | 320 | K=0 | 1,740 | 110.5 |
| 400 | 400 | **K=0** (spec 0) | 1,880 | 129.5 |

- **Verdict**: MTP × DSD fully working (a combination we could not find prior verification for in public docs / issues — the vLLM DSD docs say "tested with Eagle / E3 only").
- **Boundary finding**: some drafting cells still appear at running ≥ 129 = **the DSD index (per-step scheduled request count) fluctuates step-to-step** (ramp regions pass through lower tiers). Behavior near a boundary is probabilistic → place deployment boundaries outside the running-distribution tail + add hysteresis.
- **Prometheus volume refinement**: draft/gen ratio during CAL is 0.02 (≈ 0.52 when K = 1 dominates) → **at c ≥ 192 the steady state is mostly K = 0, with drafting only during ramps (~2%)**. "K=1 held" is an aggregation illusion (only the K of the steps that drafted). By volume, the K = 0 tier is effectively active — the tier mixing is small ramp-time leakage. The V1b window shows d/g = 0.99 = pure K = 3 signature (cross-check).
- ⚠ **Mixed-cell AR (93.3% / 84.1%) must not be cited** — an AR read from a ~2%-volume sample of drafting steps is noise.
- ⚠ **High-concurrency K=0 cells (c = 224 / 320 / 400: 1,740–2,330) are suspected anomalies pending the V2 gate**: 51–68% of the 0.23 ceiling (c128 no-spec 3,413), non-monotonic (c320 < c400), and a derived-value mismatch (c224: 75.2 ms → 2,979 tok/s derived vs. 2,330 measured) = a 45s-window contamination signature. Not isolated among: (a) window contamination (b) DSD-K0 drafter-sync tax (arithmetic estimate ~7–10%, does not explain −30 to −50%) (c) 0.24 high-batch regression. **The production workload profile (sat = 256 / peak = 769) sits in this tier — "deployable" verdict withheld pending isolation.** The V2 gate is the 120s canonical three-way comparison at c = 256 (① DSD-K0 @ 0.24  ② no-spec @ 0.24  ③ reference: 0.23 ceiling 3,413).

## 6b. V2 verdict (7/13) — DSD K=0 tier ≠ no-spec, tax −25%

| c = 256, 120s canonical | tok/s | TPOT p50 | TTFT p50 / p99 | completions |
|---|---|---|---|---|
| ② no-spec @ 0.24 | **3,413** | 73.7 ms | 0.28 s / 4.7 s | 2,048 |
| ① DSD-K0 @ 0.24 | **2,560** | 78.6 ms | **2.0 s / 10.8 s** | 1,536 |

- **② matches ③ (0.23 ceiling) exactly** → no 0.24 regression + ceiling verified flat up to c = 256 + wave-artifact ruled out.
- **① < ② = −25%, tax is concentrated in TTFT** (only +6.7% TPOT) = prefill / scheduling cost. The lower reading from 45s cells was not window contamination — the tax is the actual cause (45s 2,605 ≈ 120s 2,560).
- **Deployment guidance: for always-on high-batch servers, don't ship a K = 0 tier — drop the spec config entirely.** DSD is for the c < 128 regime.
- Not decomposed: (i) PIECEWISE cost at high batch (ii) drafter prefill obligation — but the 0.23 static-K3 at c = 128 hits 3,413, so carrying the drafter is exonerated; **the DSD mode itself is what's specific** (iii) tier-decision overhead. Discriminator: batch table `[[1,512,0]]` isolates the pure infra tax.
- **Upstream finding #4**: "DSD K=0 tier is not free at high batch" — measured, undocumented.

## 7. V1b — PIECEWISE tax (0.24, c=30 K=3 direct comparison)

| | DSD (PIECEWISE) | static (FULL+PIECEWISE) |
|---|---|---|
| tok/s | 2,533 | 2,533 |
| TPOT p50 | 10.3 ms | 10.5 ms |
| completions / AR | 570 / 68.1% | 570 / 68.2% |

→ **Tax ≈ 0**. **★ P4 boundary table complete (7/14, short-context B = 61–128):** c90 K0/K3 = 2,445 / **3,000** (1.23×, p99 960 ms) · c110 K0/K1/K3 = 2,892 / 3,185 / **3,300** (K3 best, 1.14×, **p99 796 ms**) · c128 = crossover (K0 = K3 = 3,413, p99 4,044 ms). **`[1,110,3]` extension approved** — c110 K3 TTFT p99 = 796 ms is healthy (the feared tail explosion lives between 110 and 128). **Even at short context K3 > K1** (c110 3,300 > 3,185, pos2 53.7%) → the existing `[65,128,1]` tier is over-conservative; can be raised to K = 3 up to just before c = 128. TPOT also favors K3 across the board (27–30 vs. K0 35–38). At the 4k cell: K3 TPOT 78 → 97 = **flat curve broken**; K0 @ 4k = 137 measured (7/14) → ratios 1.41× (TPOT) / 1.36× (throughput); amortization knee confirmed. **The current "deployable" basis is c ≤ 100 only — the high-concurrency tier verdict waits on §6 V2.**

## Unmeasured (debt)

- Quantitative quality gate — **(A) engine consistency complete 7/14**: greedy K=3 vs. K=0 at concurrency 1, control 30/30 deterministic, K-switch 0/30 bit-identical but **all cases benign reword (FP8 KV argmax flip, no corruption)**. Bit-identity is the wrong gate → **(B) scored equivalence** is the deployment pipeline's scored-eval domain (not run here).
- ~~4k K=0 (4th point on the SWA-asymptote curve)~~ — **complete 7/14** (§2, knee confirmed).
- Combined LMCache-served × spec throughput (after restore-bottleneck isolation).
- LMCache −63% root-cause isolation / multi-GPU restore parallelism.
