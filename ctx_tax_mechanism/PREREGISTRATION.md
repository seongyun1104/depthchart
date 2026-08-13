# Pre-registration — ctx-scaling of the DSD baseline tax, mechanism attribution

**Status: pre-registered, NOT yet run (rental-gated). Written 2026-08-13.**

## Question

Does the DSD baseline tax (spec arm slower than no-spec at production defaults) **scale with context length**, and if so, **where does the extra time go** as ctx grows?

This is the open problem left on [#49986](https://github.com/vllm-project/vllm/issues/49986). Suppressor72 (dual RTX 5090 / SM120 / MoE / TP2) measured the tax widening with ctx — 27B dense −8.5% (short) → −35% (32k), 35B MoE −19% → −24% — and, by re-measurement, **excluded** the K=0 drafter forward as the cause (5–8% of step time), leaving the 32k scaling **unexplained on their stack**. They excluded target-forward duration alone (17.80→18.77 ms ≪ 48.8 ms wall increase) and FULL-vs-PIECEWISE graph mode (PIECEWISE-only −3.2%). They explicitly invited a Gemma/H100 diagnostic with per-scheduler-step token census, prefix-cache-hit chunk sizes, and cg_mode logging; nsys they have not run.

## What we have already established (do not re-measure)

`tax_decomposition/` (2026-08-05, Gemma-4-31B hybrid + MTP, H100 NVL, ctx 400/4000):

- K=0 tier is **7–17% slower than no-spec on either runner** (V1 +7.29%, V2 +16.64% at ctx4000; effects 5–40× per-cell stdev).
- The V1 PIECEWISE downgrade is **DSD-driven, not hybrid-model-driven**, and the tax **persists after removing it** (V2 both-FULL still +16.64%). Consistent with Suppressor72's "FULL-vs-PIECEWISE not primary."
- Our own P2′ (2026-08-03): K=0 `propose()` forward is sub-noise on our shared-KV single-stream stack (TPOT +0.71%). Consistent with their K=0 exclusion.

**Gap this study fills:** our tax work stopped at ctx 4000 and did not sweep to 32k, did not instrument the mechanism, and did not use a warm/prefix-cache-hit-heavy pattern. The uplift work (K* rises with ctx) was a *different axis* and was conceded WEAK — this is the **tax** axis (`tax ≠ uplift`).

## Arms (fixed runner = V1, production default)

| arm | spec | schedule |
|---|---|---|
| A `no_spec` | off | — |
| B `dsd_k0` | on | `[[1,64,3],[65,128,0],[129,512,0]]` — lands in K=0 tier at the chosen concurrency |

Tax(ctx) = (TPOT_B − TPOT_A) / TPOT_A at each ctx. Primary endpoint: **does Tax(ctx) increase monotonically with ctx**, and by how much between the short anchor and the longest ctx.

Runner is held at V1 so this isolates the *ctx* axis; the V1/V2 graph-mode axis is already decomposed in `tax_decomposition/` and is not re-opened here.

## ctx grid

`ctx ∈ {400, 4000, 16000, 32000, 49400}` (49400 matches Suppressor72's constructed "32k"). `--max-model-len` raised to cover `max(ctx)+suffix+output` (≈ 52k).

**Concurrency is derived on the rental, per ctx, from the measured KV pool** (RUNBOOK §KV). The rule (from `tax_decomposition` pre-reg §5.3): pick the largest concurrency that keeps the workload off the preemption margin AND keeps arm B in the K=0 tier (batch ≥ 129). Applied identically across A and B at each ctx, else the within-ctx tax is invalid. Record the chosen concurrency + measured KV pool per ctx before measuring.

## Mechanism instrumentation (the new part; maps to Suppressor72's invitation)

Per measure rep, capture and retain:

1. **Prefix-cache** — `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total` (hit rate), and per-step scheduled prefill-chunk sizes from DEBUG scheduler logs.
2. **Chunked-prefill vs decode split** — `vllm:iteration_tokens_total` histogram + `vllm:num_requests_running/waiting`; count prefill-heavy vs decode-only iterations. Suppressor72's lead is that the divergence is concentrated in chunked prefill, not steady decode (their decode-worker time was flat: 13.77 vs 14.10 ms).
3. **cg_mode** — verbatim `cudagraph_mode` + any downgrade warning from the server log (as in `tax_decomposition` CUDAGRAPH_MODES.txt).
4. **Scheduler step census** — DEBUG scheduler lines (scheduled tokens, running/waiting) → `parse_scheduler_log.py`.
5. **nsys** — one wrapped run of arm B at the longest ctx (kernel/timeline attribution Suppressor72 lacks).

## Attribution criteria (pre-committed)

- **Primary:** Tax(49400) − Tax(400) in TPOT %. Report as the ctx-scaling magnitude regardless of sign; do not soft-sell if flat.
- **Mechanism verdict** (which of these the extra ctx-time concentrates in), by the instrumentation, ranked before running:
  - (H-sched) chunked-prefill / prefix-cache-hit re-processing bursts — Suppressor72's leading lead (cache-hit KV-block drop → token re-processing). Signature: prefill-iteration count and prefill-token totals grow super-linearly with ctx in arm B vs A; prefix-cache hit rate diverges between arms.
  - (H-decode) steady-decode per-step cost — signature: decode-only iteration time grows with ctx and dominates. Suppressor72's data argues against this.
  - (H-kernel) forward/attention kernel scaling — signature: nsys shows attention/allreduce kernels scaling; only invoked if H-sched/H-decode don't account for the wall delta.
- **Honesty gates:** if Tax(ctx) does not scale on our stack, that is a negative cross-stack result and gets reported as such (our stack = Gemma/H100/shared-KV differs from theirs = Qwen/5090/own-KV/MoE). If the mechanism is ambiguous, report leads, not a cause — same discipline that kept the uplift result at WEAK.

## Cross-stack framing

Positive scaling + a chunked-prefill/prefix-cache attribution on Gemma/H100 would be an **independent second-stack confirmation** that the DSD tax is a ctx-axis phenomenon located in the scheduler, not the kernel — the strongest `tax scales with ctx` proof-point for the ctx-axis thesis, and a direct answer to Suppressor72's invited diagnostic. A flat or kernel-located result on our stack is equally publishable as a boundary on the claim.
