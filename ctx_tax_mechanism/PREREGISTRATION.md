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
| B `dsd_k0` | on | `[[1,512,0]]` — flat K=0, so the arm means the same thing at every ctx |

‼️ **Confound found and removed 2026-08-23, before any measurement.** The arm was
originally `[[1,64,3],[65,128,0],[129,512,0]]`, i.e. K=0 only at batch ≥ 65. But the KV
pool forces concurrency *down* as ctx grows (holding batch ≥ 129 at ctx 16k would need
~2.1M pool tokens — far beyond a single H100), so the planned concurrencies would have
read **K=0 at ctx 400/4000 and K=3 at ctx 16000/32000/49400**. The primary endpoint
`Tax(49400) − Tax(400)` would then have measured a ctx effect *plus* a K 0→3 switch and
reported the sum as ctx-scaling. Noting the regime change in RESULTS (the original
mitigation) would not have rescued the headline number, because the headline itself
straddles the switch.

Flat K=0 keeps arm B constant so ctx is the only thing that varies, and
`assert_arm_is_k0(concurrency)` now fails the run rather than trusting the plan — the
concurrency is chosen from the measured pool on the rental, so the invariant has to be
checked at the value actually used, not the value intended. (Verified: the guard blocks
the old schedule at concurrency 12 and 48, and allows it at 192.) The batch axis is not
abandoned — it is a separate study (cf. #49548), and mixing it into the ctx sweep is
exactly what this fix prevents.

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
4. **Per-step token census** — `vllm:iteration_tokens_total` **histogram bucket deltas** around each measured run, split into decode-only vs prefill-heavy steps at the concurrency ceiling.

   ‼️ **Instrument changed 2026-08-23, before any measurement.** The original plan parsed DEBUG scheduler lines (`parse_scheduler_log.py`). Checked against `vllm-project/vllm@e25c586b90`: **current vLLM emits no per-step scheduler DEBUG line at all** — the only `logger.debug` calls in `v1/core/sched/scheduler.py` concern KV-transfer state and connector reset — so that parser would have matched nothing on the rental. The histogram is also the better instrument here: it needs no DEBUG logging, whose overhead would have taxed the very step timing this study measures. The parser has been removed rather than left to look functional.
5. **nsys** — one wrapped run of arm B at the longest ctx (kernel/timeline attribution Suppressor72 lacks).

## Attribution criteria (pre-committed)

- **Primary:** Tax(49400) − Tax(400) in TPOT %. Report as the ctx-scaling magnitude regardless of sign; do not soft-sell if flat.
- **Mechanism verdict** (which of these the extra ctx-time concentrates in), by the instrumentation, ranked before running:
  - **(H-waste) the K=0 arm is not actually K=0 on MRV2** — added 2026-08-23 from
    [#51510](https://github.com/vllm-project/vllm/issues/51510) /
    [PR #51575](https://github.com/vllm-project/vllm/pull/51575) (Suppressor72, open):
    `AutoRegressiveSpeculator` under `VLLM_USE_V2_MODEL_RUNNER=1` ignores the scheduler's
    per-step K and runs the **full configured draft pipeline**, discarding the output. A
    wasted draft step attends over the whole KV, so its cost **grows with ctx** — which is
    the shape of the −8.5% → −35% scaling that is currently unexplained on their stack.

    ‼️ **This is excluded by construction on our stack, and that is the point.** Verified in
    `vllm-project/vllm@e25c586b90`: the V1 runner threads
    `scheduler_output.num_spec_tokens_to_schedule` into every proposer
    (`gpu_model_runner.py:5133–5377`, e.g. `self.drafter.propose(num_speculative_tokens=
    num_spec_tokens_to_schedule, ...)` at :5225), while **no file under
    `vllm/v1/worker/gpu/` references that field at all**. Our arm B is V1, so it is
    genuinely K=0.

    This turns the study into a **discriminating** measurement rather than a fishing trip:
    - Tax(ctx) **scales on our V1 stack too** → there is a second, structural mechanism
      independent of #51510, and #49986 is not merely that bug.
    - Tax(ctx) **is flat on V1** while theirs scales → the leading answer to #49986 is the
      MRV2 wasted-draft bug, and PR #51575 is the fix. Either way the result is decisive,
      which the original open-ended mechanism hunt was not.

    **Free cross-check on data we already hold (no rental):** `tax_decomposition/` measured
    the same K=0 tier at **V1 +7.29%** vs **V2 +16.64%** (ctx 4000). Under H-waste the V2−V1
    gap *is* the wasted draft work, since V2 is exactly the runner that ignores K=0. That is
    a prediction our existing numbers already satisfy in sign and rough magnitude, and it was
    read out of the runner source, not fitted.

  - **(H-kv) drafter KV reservation shrinks the target's usable pool** — added 2026-08-23
    from LongSpec ([2502.17421](https://arxiv.org/abs/2502.17421)), which identifies "the
    excessive memory demands posed by draft models due to large Key-Value (KV) cache" as
    *the* long-context drafting problem and answers it with a constant-size draft KV cache.
    If the drafter's KV grows with ctx, arm B has less pool than arm A at the same
    concurrency. Signature: `GPU KV cache size` differs A vs B, and/or
    `vllm:num_preemptions_total` rises with ctx in B only. Cheap to read — both are already
    captured.

  - (H-sched) chunked-prefill / prefix-cache-hit re-processing bursts — Suppressor72's leading lead (cache-hit KV-block drop → token re-processing). Signature: `prefill_step_frac` and the high `iteration_tokens` buckets grow super-linearly with ctx in arm B vs A; prefix-cache hit rate diverges between arms.
  - (H-decode) steady-decode per-step cost — signature: decode-only iteration time grows with ctx and dominates. Suppressor72's data argues against this.
  - (H-kernel) forward/attention kernel scaling — signature: nsys shows attention/allreduce kernels scaling; only invoked if H-sched/H-decode don't account for the wall delta.
- **Excluded by construction (do not spend the rental on it):** acceptance collapse at long
  context — OWL ([2510.07535](https://arxiv.org/abs/2510.07535)) shows EAGLE3 falling to an
  acceptance length of 1.28 and running **0.81× slower than standard decoding** on 4K–64K
  inputs. That is a real long-context speculative-decoding tax, but it is an *acceptance*
  phenomenon, and arm B emits no draft tokens to accept or reject. It cannot explain a K=0
  tax on either stack.

- **Honesty gates:** if Tax(ctx) does not scale on our stack, that is a negative cross-stack result and gets reported as such (our stack = Gemma/H100/shared-KV differs from theirs = Qwen/5090/own-KV/MoE). If the mechanism is ambiguous, report leads, not a cause — same discipline that kept the uplift result at WEAK.

## Cross-stack framing

Positive scaling + a chunked-prefill/prefix-cache attribution on Gemma/H100 would be an **independent second-stack confirmation** that the DSD tax is a ctx-axis phenomenon located in the scheduler, not the kernel — the strongest `tax scales with ctx` proof-point for the ctx-axis thesis, and a direct answer to Suppressor72's invited diagnostic. A flat or kernel-located result on our stack is equally publishable as a boundary on the claim.
