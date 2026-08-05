# Graph-mode tax decomposition — results (2026-08-05)

Companion to `PREREGISTRATION.md` (v3), `master_bench_tax.py`, `aggregate_tax.py`,
`RUNBOOK.md`. Raw data under `raw/root/results/` + `raw/tmp/server_*.log`.

## Setup

- Target: `prithivMLmods/gemma-4-31B-it-qat-FP8` (hybrid) + MTP drafter
  `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`.
- HW: 1× H100 NVL 94 GB. `--gpu-memory-utilization 0.90 --max-model-len 8192`,
  no `--kv-cache-dtype fp8`. DSD schedule `[[1,64,3],[65,128,0],[129,512,0]]`.
- Arms A/B/C/D built on PR #49652 @ `fd355781`; spot (S) on upstream main
  `4719a9b8` (#49652 not merged). Single lever between V1/V2 arms:
  `VLLM_USE_V2_MODEL_RUNNER`. n = 3 measure seeds (3 warmup discarded), single
  session, no order-reversal.

## Gate T (ctx=400, pre-reg §3)

`Tax(V1)@400 = +13.47% -> PROCEED (>=10%)`. The tax reproduces on this build;
not a #49986 correction event. (This cell is preemption-contaminated — see
Caveats — so the clean read is ctx4000 below.)

## Gate B (V2 launch, pre-reg §3)

Both V2 arms launched clean, **no #48494 assert**, FULL cudagraph captured:

| arm | READY | v48494_assert_seen | full_cudagraph_seen |
|---|---|---|---|
| C v2_no_spec | 220.2 s | false | true |
| D v2_dsd_k0  | 235.3 s | false | true |

The #49652 branch opens the hybrid + MTP + DSD stack in the V2 model runner with
FULL cudagraph, including with speculative decoding active (arm D).

## Per-arm cudagraph mode (verified from compilation_config in server logs)

| arm | spec | cudagraph_mode | downgrade warning |
|---|---|---|---|
| A v1_no_spec | none | FULL_AND_PIECEWISE | no |
| B v1_dsd_k0  | DSD-MTP | **PIECEWISE** | yes (DSD dynamic verify-len) |
| C v2_no_spec | none | FULL_AND_PIECEWISE | no |
| D v2_dsd_k0  | DSD-MTP | FULL_AND_PIECEWISE | no |

The V1 PIECEWISE downgrade is **DSD-driven, not hybrid-model-driven** — the log
override reason is "Dynamic speculative decoding changes the target verification
length at runtime." See `raw/root/results/CUDAGRAPH_MODES.txt` for the verbatim
warning.

## Decomposition (TPOT p50, ms)

| arm | ctx400 | ctx4000 |
|---|---|---|
| A v1_no_spec | 104.58 ± 1.85 | 127.74 ± 0.39 |
| B v1_dsd_k0  | 118.66 ± 1.05 | 137.05 ± 0.70 |
| C v2_no_spec | 102.28 ± 2.04 | 127.23 ± 0.67 |
| D v2_dsd_k0  | 135.78 ± 0.94 | 148.41 ± 1.46 |

**ctx4000 (clean, preempt = 0, both DSD arms confirmed K=0: median batch 192):**

- `Tax(V1) = (B-A)/A = +7.29%`  (includes B's PIECEWISE-downgrade penalty)
- `Tax(V2) = (D-C)/C = +16.64%` (both arms FULL — no graph-mode asymmetry)
- baseline guard `(C-A)/A = -0.39%`
- effect sizes are 5-40× the per-cell stdev.

## What this does and does not say

**Headline (defensible, answers #49986 directly):** the K=0 tier is **7-17%
slower than no-spec on either runner** (V1 +7.29%, V2 +16.64% at ctx4000). If a
schedule spends most steps at K=0, speculation is a net cost there.

**Within the K=0 tier:** V2's spec path is slower than V1's (D vs B = +8.29%
absolute TPOT at ctx4000), even though V2 removes the PIECEWISE downgrade. So the
K=0 tax is not explained by the V1 graph-mode downgrade. This refutes the
#49986 working hypothesis that the tax is a PIECEWISE penalty V2 would recover.

**What we cannot claim (attribution + scope limits):**

1. K=0 is the least favorable case for FULL cudagraph (verify length is fixed,
   so FULL's structural benefit is smallest here). **K>0 net-effect was not
   measured** (pre-reg §5.3 cells empty). No statement about MRV2 in general or
   about K>0 is supported.
2. V2+DSD cannot run without this branch (#48494), so this data **cannot
   separate the MRV2 runner's spec-path cost from the #49652 patch's cost**.
   Separating them requires a `use_dynamic_decode_query_len` True/False
   comparison on the same branch (not run here).

**Thermal/drift control:** arms ran A→B→C→D over ~3 h. C (no_spec, ~2 h after A)
matches A within -0.39%, so the V2>V1 spec-path gap is not a time/thermal drift
artifact.

## Spot check S (pre-reg §5.2)

`spot (main-V1-dsd)@400 = 118.79 ms` vs `branch-V1-dsd (B)@400 = 118.66 ms`,
**drift = +0.11%**. The branch does not alter V1 behavior (expected: #49652
touches the V2 capture path; V1 DSD downgrades to PIECEWISE on both). The V1
measurement is valid and ties, via Gate T, to the published #49986 point.

## Caveats

- **ctx400 preemption:** the DSD arms saturate the KV pool at concurrency 256
  (B 980 / D 823 preemptions, KV usage 100% vs no_spec 97.5%), so ctx400 Tax
  values are preemption-inflated. ctx4000 (concurrency 192) is preempt-free and
  is the clean read. A clean ctx400 would need a re-run at ~110 concurrency
  (keeps K=0, off the KV margin); not run.
- **drafts asymmetry:** ctx4000 drafts V1 5027 vs V2 5697 (×1.13), ramp-only
  (steady state is K=0, no drafts); does not explain the +8.29% steady-state
  gap, but noted.
- Single stack, K=0 only, 2 ctx, single session, n=3, no order-reversal.

## Next

`use_dynamic_decode_query_len` True/False on the #49652 branch (V2+DSD, K=0) to
separate MRV2 runner cost from patch cost — ~15 min, next rental.
