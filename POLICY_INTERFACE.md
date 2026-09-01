# Two producers, one schedule — what is built and what the cost model gets wrong

Written 2026-09-01. Companion to [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md).

The schedule table is the policy interface of a K controller, and the question
this note answers is whether more than one producer can write into it today.
Two can. The second one, with the coefficients it currently ships, would not
have found the effect the first one measured — and that is the finding worth
recording, not the plumbing.

---

## 1. The measured effect is one cell

The two schedules A/B'd in the PR #48944 campaign
(`pr48944_replication/master_bench.py`):

```
A'  (batch-only, 3-item)   [[1,64,3], [65,128,1], [129,512,0]]

C'  (2-D, 6-cell)          [1,  64,   1, 768,   3]  [1,  64,   769, 32768, 3]
                           [65, 128,  1, 768,   1]  [65, 128,  769, 32768, 1]
                           [129,512,  1, 768,   0]  [129,512,  769, 32768, 3]
```

Expanded through `build_dynamic_sd_schedule_lookup` at `max_num_seqs=512`,
`num_speculative_tokens=3`:

| batch size | A′ K | C′ K, ctx 1–768 | C′ K, ctx ≥ 769 |
|---:|:-:|:-:|:-:|
| 1 | 3 | 3 | 3 |
| 64 | 3 | 3 | 3 |
| 65 | 1 | 1 | 1 |
| 128 | 1 | 1 | 1 |
| **129** | **0** | 0 | **3** |
| **256** | **0** | 0 | **3** |
| **512** | **0** | 0 | **3** |

**The two schedules differ in exactly one cell**: the high-batch tier at long
context, where the batch-only schedule has turned speculation off and the 2-D
schedule turns it back on. Everything else is identical.

That is what the 1.29× / 1.30× / 1.36× at ctx 900 / 1900 / 4000 buys, and it is
also why ctx 400 ties at 1.01× — there the two schedules select the same K.

**The distinct-K set is unchanged**: `{0, 1, 3}` for both. The context axis costs
no additional CUDA graphs, because graphs are keyed on K and the K values are the
same ones. That is a mechanical property of the two schedules, checked rather
than asserted.

## 2. Two producers already write into that interface

Both on branch `feat/dsd-ctx-layer-a-on-837f835759` (fork, `68b69416c4`),
stacked on PR #48944's head `837f835759`:

| mode | producer | provenance of K |
|---|---|---|
| `spec_schedule_mode="manual"` | the list the operator writes | measured, or hand-tuned |
| `spec_schedule_mode="derive"` | `derive_dynamic_sd_schedule(...)` in `vllm/v1/spec_decode/dynamic/cost_model.py` | argmax of a stated cost model |

The two are mutually exclusive and both emit the same rectangular
`(bs_lo, bs_hi, ctx_lo, ctx_hi, K)` form, which
`validate_and_normalize_dynamic_sd_schedule` accepts unchanged. Verified: both
A′ and C′ pass the validator and expand to a lookup, and the derived schedules
below do too.

The cost model is:

```
E_accept(K) = sum_{i=0..K} acceptance^i
T_step(K)   = F(ctx) + K * M(bs)
K*(bs,ctx)  = argmax_{0<=K<=max_k}  E_accept(K) / T_step(K)

F(ctx) = verify_fixed_base   + verify_fixed_per_ctx   * ctx
M(bs)  = draft_marginal_base + draft_marginal_per_bs  * bs
```

It also coarsens the result to a CUDA-graph budget (`capture_budget`) and drops
context boundaries that no tier distinguishes after coarsening.

This is the same marginal-utility shape two 2026 papers arrive at independently
— LibraSpec ([2608.08721](https://arxiv.org/abs/2608.08721)) optimises
speculative length by marginal gain, and SparseSpec-L
([2607.27735](https://arxiv.org/abs/2607.27735)) states the condition directly:
*"extending the speculation horizon can reduce rather than improve speedup when
the marginal acceptance probability falls below the relative drafting cost."*

## 3. What the cost model gets wrong

Run with **the coefficients it ships** (`verify_fixed_base=1.0`,
`verify_fixed_per_ctx=2.5e-4`, `draft_marginal_base=0.2`,
`draft_marginal_per_bs=0.02`), tiers `(1,64) (65,128) (129,512)`, grid
`400 900 1900 4000 32768`, `max_k=3`, sweeping only the acceptance prior:

| acceptance | K at bs 64 | K at bs 128 | K at bs 256 |
|---:|---|---|---|
| 0.35 | 0 → 1 | 0 → 1 | **0 → 0** |
| 0.45 | 0 → 2 | 0 → 1 | **0 → 0** |
| 0.55 | 0 → 2 | 0 → 1 | **0 → 0** |
| 0.65 | 0 → 3 | 0 → 2 | **0 → 0** |
| 0.75 | 0 → 1 → 3 | 0 → 0 → 2 | **0 → 0 → 0** |

(each cell reads short context → long context; boundaries are the ones the model
emitted, mostly at ctx 32768)

**At the high-batch tier the model selects K=0 at every context, for every
acceptance prior in the measured range** — the campaign observed 44–47 %
acceptance in the K=3 tier. In other words, the cost model with its shipped
coefficients reproduces A′, not C′. **It would not have found the one cell that
the measurement says is worth 29–36 %.**

**This is calibration, not functional form.** The model can produce a
context-driven K rise at high batch once the coefficient ratio changes — the
per-context growth has to be roughly an order of magnitude larger relative to
the per-batch growth than the defaults assume:

| `draft_marginal_per_bs` | `verify_fixed_per_ctx` | K at bs 256, ctx 400 → 4000 |
|---:|---:|---|
| 0.02 (default) | 2.5e-4 (default) | 0 → 0 |
| 0.02 | 2.5e-3 | **0 → 1** |
| 0.005 | 2.5e-4 | **0 → 1** |
| 0.001 | 2.5e-4 | **2 → 3** |

The docstring already says the defaults are "design-time references, not
measurements" that "must be calibrated on the target hardware before any speedup
is claimed." This note is the concrete demonstration of why that sentence is
load-bearing rather than boilerplate.

## 4. An open question this raises, stated as a question

`K*` from this model is **monotonically non-increasing in batch at every
context** — a structural consequence of `M(bs)` growing while `F(ctx)` does not
depend on batch. Checked:

| ctx | K\* over bs = 1, 16, 64, 128, 256, 512 |
|---:|---|
| 400 | 1, 0, 0, 0, 0, 0 |
| 4000 | 2, 1, 0, 0, 0, 0 |
| 32768 | 3, 3, 2, 1, 0, 0 |

The schedule that won the campaign is not monotone in batch at long context: K=3
at bs 1–64, K=1 at bs 65–128, K=3 at bs 129–512.

**We cannot yet call that a refutation.** Only one of those three tiers was A/B'd
— the campaign measured K=3 against K=0 at bs 129–512, ctx ≥ 769. The K=1 at bs
65–128 was authored, not measured against alternatives. So the honest statement
is: *the measured cell is one this functional form reaches only under
recalibration, and whether the full non-monotone shape is real or an artifact of
how we authored the tiers has not been tested.* Testing it means A/B'ing the
middle tier, which is one rung on the concurrency ladder already designed in
`tax_attribution/`.

## 5. What this licenses

- Saying that more than one producer writes into the schedule interface today —
  **yes**, both exist and both are exercised above.
- Saying the context axis is resource-neutral in graph terms for this pair of
  schedules — **yes**, the distinct-K set is identical.
- Saying we have a validated cost model — **no**. We have a stated one whose
  shipped coefficients disagree with our own measurement in the cell that
  matters, and calibration on hardware has not been done.
- Saying the measured 29–36 % generalises — **no**. It is one model pair, one
  concurrency, four context points, and one differing cell.
