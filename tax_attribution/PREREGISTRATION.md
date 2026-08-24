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
- **KV term** — the drafter holds ~15 025 tokens, about 10.6 % of the pool,
  while never drafting, which costs concurrency rather than per-token latency.

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
- KV term = `base_full_low` − `base_full_high`
- total = `dsd_k0_low` − `base_full_high`
- residual = `dsd_k0_low` − `base_piece_low`

## Predictions, registered in advance

1. **Additivity.** graph term + KV term ≈ total. Registered as an ordering and
   sign claim, not a magnitude: both terms are positive, and their sum accounts
   for the total to within the run-to-run spread of the baseline cells.
2. **Regime.** The KV term is ≈0 at concurrency 2 and materially positive at
   concurrency 189; the graph term is positive at both. This is the specific
   claim that the batch scaling reported in `ctx_tax_mechanism` belongs mostly
   to the KV term.
3. **Reconstruction.** `base_piece_low` ≈ `dsd_k0_low`. A no-spec server with
   the drafter's graph mode and the drafter's pool should reproduce the spec
   arm's cost without any drafter present.

No magnitude is registered for any term. `ctx_tax_mechanism` §5 is the reason:
the tax is a ratio whose denominator is the step cost, so its size is a
property of the model and batch, not something to predict in advance.

## What each outcome means, decided now

| result | reading |
|---|---|
| P1, P2, P3 all hold | the tax is fully explained by graph mode plus drafter KV occupancy; #52087 recovers the graph term only, and the KV term needs a constant-size draft cache |
| P1 holds, P2 fails (KV term flat in batch) | the batch scaling is the graph term's, and preserving FULL graphs recovers most of the tax at high batch |
| P3 fails, `dsd_k0_low` slower than `base_piece_low` | a third cost exists — spec-path bookkeeping that neither mechanism covers — and it, not the two known terms, is what the next study must chase |
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

## Limits accepted in advance

One GPU, one target, one drafter, one workload shape, one context. This study
answers attribution, not generality; a second target of different arithmetic
intensity is the follow-up, and only the sign and mechanism would carry.

Aggregate goodput at a TPOT SLO is not measured here. The KV term is invisible
to a per-token metric by construction, so its cost appears in this design as
the latency of a smaller pool at fixed concurrency rather than as throughput
lost. That is a narrower reading than #49548's complaint and is stated as such.
