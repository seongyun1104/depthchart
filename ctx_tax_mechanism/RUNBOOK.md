# RUNBOOK — ctx-tax mechanism study (rental)

Pre-built 2026-08-13, **not yet run**. Pre-reg: `PREREGISTRATION.md`. Extends the
validated `tax_decomposition/` harness. Estimated wall ~2.5–4 h on 1× H100 NVL
(high-ctx runs dominate); budget a rental accordingly ([[feedback-gpu-time-is-money]]).

## 0. Provision (fold in the 2026-08-13 ctx-rental lessons)

- HF_TOKEN preset **before** any pull (auth 3 min vs unauth 25 min).
- `transformers==5.10.2` pin (5.15.0 breaks Gemma-4 hybrid config — `AmbiguousGlobalPerLayerAttributeError`).
- `mkdir -p /workspace` if absent; pin the vLLM build SHA with `git fetch origin <full-sha>` (head can move).
- Models: target `prithivMLmods/gemma-4-31B-it-qat-FP8` → `/root/models/target`; MTP drafter
  `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` (drafter is in the target's speculative-config).

## 1. §KV — measure the pool, then set concurrency (BLOCKING)

`master_bench_ctxtax.py:CTX_CONCURRENCY` ships with **placeholders**. Before measuring:

1. Launch arm `no_spec` at `MAX_MODEL_LEN` and read the KV pool size from the
   server log (`GPU KV cache size: N tokens`).
2. For each ctx, set concurrency to the largest value that (a) keeps
   `concurrency * (ctx+196) < 0.9 * pool` (off the preemption margin) and, where
   the pool allows, (b) is ≥ 129 so arm `dsd_k0` stays in the K=0 tier. Above the
   ctx where the pool forces concurrency < 129, record that the tax is read in the
   low-batch regime for that ctx — do not silently mix regimes; note it in RESULTS.
3. Apply the **same** concurrency to both arms at each ctx (else within-ctx tax is invalid).

`--enable-prefix-caching` is on (the `prefix_repetition` dataset shares one prefix
across prompts, so radix hits accumulate — the pattern Suppressor72's lead implicates).

## 2. Run

```bash
export MODEL=/root/models/target RESULTS_DIR=/root/results MAX_MODEL_LEN=52224
python master_bench_ctxtax.py no_spec dsd_k0
```

Each arm: cold-start burn (2×c64), then per ctx 1 warmup + 3 measure (32k/49k) or
3+3 (short). Server runs at `VLLM_LOGGING_LEVEL=DEBUG`; the DEBUG log is copied to
`RESULTS_DIR/server_{arm}.log` for the scheduler census.

### nsys (arm B, longest ctx only — kernel/timeline attribution Suppressor72 lacks)

```bash
NSYS=1 python master_bench_ctxtax.py dsd_k0 --only-ctx 49400
```

## 3. Aggregate + attribute

```bash
python aggregate_ctxtax.py                       # Tax(ctx) table + primary endpoint
python parse_scheduler_log.py $RESULTS_DIR/server_no_spec.log $RESULTS_DIR/server_dsd_k0.log \
    --out $RESULTS_DIR/scheduler_census.json     # prefill vs decode steps per arm
```

**Confirm the DEBUG format first:** `grep -i schedul $RESULTS_DIR/server_dsd_k0.log | head`.
If `parse_scheduler_log.py` reports `WARNING: 0 steps parsed`, update `PATTERNS`
to the build's actual line format before trusting the prefill/decode split.

## 4. Read (pre-committed, PREREGISTRATION §Attribution)

- **Primary:** `Tax(49400) − Tax(400)` in TPOT pp. Report sign as measured; no soft-sell if flat.
- **Mechanism:** does the extra ctx-time land in prefill-heavy steps (H-sched) — prefill-step
  count / prefill-token totals growing super-linearly in arm B vs A, prefix-hit-rate diverging —
  or steady decode (H-decode), or kernels (H-kernel, nsys)? Report a lead, not a cause, if ambiguous.
- **Cross-stack:** positive scaling + scheduler attribution on Gemma/H100 = independent
  second-stack confirmation for #49986; flat/kernel-located = a boundary on the claim. Either
  is publishable. Land raw + RESULTS.md in depthchart, reply to Suppressor72 with the diagnostic.

## Discipline

Only rent when pursuing this (Suppressor72's invitation stands but isn't urgent). Raw stdout +
JSON + DEBUG logs to depthchart per [[reference-depthchart-bench-source-of-truth]]. Upstream
`vllm bench serve` only ([[feedback-upstream-bench-only]]); the mechanism instrumentation is
logging/metrics around it, not a custom timing loop.
