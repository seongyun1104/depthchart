# Pre-registration — splitting the DSD baseline tax between the graph term and the KV term

Written before the run. Successor to `ctx_tax_mechanism/`, which established
that the tax scales with batch rather than context but left the two candidate
mechanisms bundled in every spec arm.

## The question

`ctx_tax_mechanism` measured a K=0 dynamic-SD arm paying +6.56 % at concurrency
2 and +21.58 % at concurrency 189 against a no-spec baseline, at fixed context
400, with zero draft tokens produced. Two mechanisms were visible and neither
was isolated:

- **graph term** — enabling dynamic SD forces `cudagraph_mode` from
  `FULL_AND_PIECEWISE` down to `PIECEWISE`, so the decode path loses its full
  CUDA graphs.
- **pool term** — the drafter costs ~15 025 tokens, about 10.6 % of the pool,
  while never drafting, which costs concurrency rather than per-token latency.
  The memory profiler attributes ~79 % of that to the activation peak it
  reserves when the drafter is present and only ~21 % to its weights, so this
  is a capacity reservation and not a draft KV allocation.
- **sync-forward term** — v0.27.1 places the K=0 early return *after* the
  drafter's forward, so a forward runs on every step of the spec arm even
  though no draft token is produced. Our own fork measurement bounds it at
  +0.71 % TPOT on this stack; Suppressor72 measures 4–11 % of step time on
  other drafter architectures (#53420, #53426).

Every dsd arm carries both, so the +15.02 pp of batch scaling cannot be
attributed. This study varies them independently.

## Design

Fixed context 400. Five arms, each at concurrency 189 and 2, in that order.

| arm | spec | cudagraph | KV pool |
|---|---|---|---|
| `base_full_high` | off | `FULL_AND_PIECEWISE` | high | 
| `base_piece_high` | off | forced `PIECEWISE` | high |
| `base_full_low` | off | `FULL_AND_PIECEWISE` | low |
| `base_piece_low` | off | forced `PIECEWISE` | low |
| `dsd_k0_low` | K=0 schedule | forced to `PIECEWISE` by vLLM | low |

`high` is the pool a no-spec server profiles; `low` is the pool a
drafter-loaded server profiles. Both are pinned with `--kv-cache-memory-bytes`
(present since before v0.27.1, `vllm/config/cache.py`; CLI at
`arg_utils.py`), which ignores `gpu_memory_utilization` and makes the pool a
controlled variable rather than a profiling outcome.

Derived quantities:

- graph term = `base_piece_high` − `base_full_high`
- pool term = `base_full_low` − `base_full_high`
- total = `dsd_k0_low` − `base_full_high`
- sync-forward + bookkeeping = `dsd_k0_low` − `base_piece_low`

## Predictions, registered in advance

1. **Additivity.** graph term + pool term ≈ total, once the sync-forward term
   is included. Registered as an ordering and
   sign claim, not a magnitude: both terms are positive, and their sum accounts
   for the total to within the run-to-run spread of the baseline cells.
2. **Regime.** The pool term is ≈0 at concurrency 2 and materially positive at
   concurrency 189; the graph term is positive at both. This is the specific
   claim that the batch scaling reported in `ctx_tax_mechanism` belongs mostly
   to the pool term.
### A fourth term found after this was first written

Re-reading the `ctx_tax_mechanism` counters against #53670 showed the two arms
do not share a prefix-cache hit rate: at ctx 400 the spec arm hits 83.81 % where
the baseline hits 96.70 %, a constant deficit of ~512–1 024 tokens that does not
grow with context. The cause is the EAGLE-family last-block drop, which is
active for `method='mtp'` and which `kv_cache_coordinator` applies to every KV
group on a layout without `is_eagle_group`.

That term lands inside P3. Only `dsd_k0_low` carries a `speculative_config`, so
only it takes the drop; `base_piece_low` keeps a full hit rate. Left alone, the
`dsd_k0_low` − `base_piece_low` distance would bundle **three** things: the sync
forward, spec-path bookkeeping, and prefix-cache recompute — and attributing a
non-zero result to any one of them would be guesswork of exactly the kind this
study exists to avoid.

So the design gains a control pair, `base_piece_low_nocache` and
`dsd_k0_low_nocache`, identical to their cached counterparts but served with
`--no-enable-prefix-caching`. With no cache there is no block to drop, so their
distance is the forward plus bookkeeping alone, and

    cache term = (dsd_k0_low − base_piece_low) − (dsd_k0_low_nocache − base_piece_low_nocache)

isolates the recompute. The pair runs at the high-concurrency rung only, where
the total term is large enough for the subtraction to resolve.

3. **Reconstruction, and the sync forward.** `base_piece_low` ≈ `dsd_k0_low`.
   A no-spec server with the drafter's graph mode and the drafter's pool has no
   drafter and therefore runs no sync forward, so whatever separates it from the
   spec arm *is* the forward plus spec-path bookkeeping. This cell is the direct
   measurement of the term §3 of `ctx_tax_mechanism` originally left out. We
   predict it is small here, consistent with the +0.71 % our fork measured when
   skipping that forward, and we register in advance that a large residual would
   mean that earlier measurement does not generalise even within this stack.

### The decision rule, fixed before any data

A residual counts against a prediction only when it clears both bars:

- **resolvable** — larger than twice the standard error of the difference,
  propagated across the cells that enter it;
- **material** — larger than 10 % of the total term at that concurrency.

Both bars are needed. With three measured runs per cell the standard error is
small enough that sampling noise alone flips a verdict: replaying the aggregator
over a synthetic grid whose arms were given *identical* means produced a P3
failure from noise at the low-concurrency cell, where the total term is about
1 ms. A residual that is resolvable but immaterial is reported in those words
rather than as either a pass or a failure.

This rule is written here, before the run, precisely because choosing it after
seeing the residuals would be choosing the answer.

No magnitude is registered for any term. `ctx_tax_mechanism` §5 is the reason:
the tax is a ratio whose denominator is the step cost, so its size is a
property of the model and batch, not something to predict in advance.

## What each outcome means, decided now

| result | reading |
|---|---|
| P1, P2, P3 all hold | the tax is fully explained by graph mode plus the pool reservation; #52087 recovers the graph term only, and the pool term needs the drafter's activation peak to shrink, not a smaller draft cache |
| P1 holds, P2 fails (pool term flat in batch) | the batch scaling is the graph term's, and preserving FULL graphs recovers most of the tax at high batch |
| P3 fails, `dsd_k0_low` slower than `base_piece_low` | the K=0 sync forward costs real time on this stack after all, our +0.71 % bound does not hold at these cells, and #53426 becomes directly relevant here rather than only on other stacks |
| P1 fails, terms sub-additive | the two mechanisms interact; the decomposition framing is wrong and must be retracted rather than reported with a caveat |

A negative result on any of these is reported as measured. The
`ctx_tax_mechanism` cross-stack negative against #49986 is the precedent.

## Guards, asserted by the harness rather than trusted

- **First-launch offset.** The first server launch of each arm profiles about
  0.53 GiB less KV memory than every launch after it — 1 586–1 587 tokens, the
  same offset in both arms, stable to the token thereafter. It is what made the
  first published drafter-KV figure 1 pp low. Each arm therefore burns one
  discarded launch before any measured cell, recorded in `pools.json` with a
  `discarded` flag, and cross-arm pool comparisons refuse to run unless both
  arms have one.
- **Pin held.** Every measured launch re-reads its pool and fails if it differs
  from the previous measured launch of the same arm, which would mean
  `--kv-cache-memory-bytes` did not hold.
- **Graph mode took.** Every launch parses its CUDA-graph capture counts.
  An arm meant to keep full graphs must capture `FULL`; an arm meant to run
  without them must capture none. A forced mode that silently failed would make
  the whole 2×2 meaningless, so the run aborts rather than measuring it.
- **K stays 0.** The dsd arm asserts its schedule yields K=0 at the concurrency
  actually used, since the pool and not the plan decides concurrency.
- **Concurrency fits.** Each cell checks `concurrency × (ctx + 197) < 0.9 ×
  pool` before measuring, the same rule that set the concurrencies in
  `ctx_tax_mechanism`, so no cell silently measures preemption instead.

## The ladder, registered separately

The same harness runs a concurrency ladder at fixed context with two arms,
`base_full_high` and `dsd_k0_low`, over rungs 1 to 189. It answers a different
question from the 2x2 and carries its own prediction.

`ctx_tax_mechanism` has two concurrency points, which cannot tell a threshold
from a smooth ramp. #49548 reports a collapse "at the batch-size threshold", so
the distinction decides whether our data corroborates that report or merely
rhymes with it. This was stated as unresolved when the ctx result was posted to
that issue and is the promise being paid here.

- **Registered prediction.** The tax rises monotonically with concurrency, with
  no rung where it jumps by more than half the total range in one doubling. A
  jump that large is a threshold and would corroborate #49548 directly; smooth
  growth means our result is a different phenomenon from theirs and the two
  should stop being cited together.
- **Pools are natural, not pinned.** The ladder is the production-shaped
  comparison, so each arm profiles its own pool and the drafter's KV footprint
  is allowed to bind at high concurrency. That asymmetry is the effect under
  study, not a confound to remove. The first-launch discard still applies.
- **Rungs run in span order** (189, 1, 32, 8, 64, 128, 16, 4, 2), not climbing
  order, so an interrupted rental still covers the range.
- **Over-cap rungs are measured, not skipped.** Where a rung exceeds
  `concurrency x (ctx + 197) < 0.9 x pool` for an arm, the cell records
  `fits: false` with its preemption count and runs anyway. The concurrency at
  which the spec arm starts preempting while the baseline does not is itself the
  measurement, so aborting there would discard the answer.

## Limits accepted in advance

One GPU, one target, one drafter, one workload shape, one context. This study
answers attribution, not generality; a second target of different arithmetic
intensity is the follow-up, and only the sign and mechanism would carry.

Aggregate goodput at a TPOT SLO is not measured here. The KV term is invisible
to a per-token metric by construction, so its cost appears in this design as
the latency of a smaller pool at fixed concurrency rather than as throughput
lost. That is a narrower reading than #49548's complaint and is stated as such.
