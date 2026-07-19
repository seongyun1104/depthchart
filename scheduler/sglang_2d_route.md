# SGLang adaptive spec `_route` — (batch, ctx) 2D extension (paper-diff)

**Status.** Design + patch shape for cross-engine replication of the vLLM
`(B, ctx)` K controller. Not yet applied to SGLang tree. Ready to become a
real PR once V7 (gemma4_mtp smoke on SGLang) lands and a serving env is
available.

**Anchors.** All line references are against SGLang upstream at commit
`73d976e` (main, 2026-07-14 fetch).

## Claim

The mechanism the vLLM RFC extends into `num_speculative_tokens_per_batch_size`
(add a ctx-length range to each entry) has a direct, small-surface analogue in
SGLang's `--speculative-adaptive` machinery: extend the BS-keyed slot dict to a
`(BS, ctx)`-keyed slot dict, and pass a batch ctx representative to `_route`.

CUDA-graph capture set is unchanged (graphs are keyed by step count, not by
config-cell count), so the extension is **resource-neutral** — same argument
as vLLM.

## Current 1D shape (recap, with line anchors)

`python/sglang/srt/speculative/adaptive_spec_params.py`:

- L22-47 — `DEFAULT_ADAPTIVE_CONFIG`: dict keyed by BS integer-string, values
  are per-BS `candidate_steps` + hysteresis params.
- L88-124 — `_load_adaptive_config`: parses the dict, validates each BS entry
  has a non-empty `candidate_steps` list.
- L260-288 — `AdaptiveSpeculativeParams.__init__`: builds
  `self._slots: dict[int, AdaptiveStepSlot]` from the BS dict.
- L327-331 — `_route(batch_size) -> AdaptiveStepSlot`: pad bs to CUDA-graph
  bs, then closest-below-BS lookup.
- L313-325 — `cuda_graph_bs_for_step`: returns bs values whose slot's
  `candidate_steps` contains the target step (used for graph capture pruning).

Call sites:

- `adaptive_runtime_state.py:113` — `activate_step_by_batch(batch_size)`
  called before batch dispatch.
- `adaptive_runtime_state.py:118` — `on_verify_complete(num_correct_drafts_per_req, batch_size)`
  called after CPU sync of `accept_lens`.
- `scheduler_components/batch_result_processor.py:548` — the only production
  invocation of `on_verify_complete_cpu`, passing `batch_size=len(batch.reqs)`.
  `batch.seq_lens_cpu` is already resident at this point (scheduler.py:2575).
- `eagle_worker_v2.py:1246` / `ngram_worker.py:201` — worker-side
  `on_verify_complete_cpu` wrappers.

## 2D config schema (backward-compatible)

Legacy (still valid — interpreted as single ctx bucket `[1, max_context]`):

```json
{
  "1":  {"candidate_steps": [1, 3, 7], "down_hysteresis": -0.25},
  "8":  {"candidate_steps": [0, 1, 3]},
  "32": {"candidate_steps": [0, 1]},
  "64": {"candidate_steps": [0]}
}
```

Extended (ctx sub-buckets under each BS):

```json
{
  "1":  {"candidate_steps": [1, 3, 7], "down_hysteresis": -0.25},
  "8": {
    "ctx_buckets": {
      "1-2047":     {"candidate_steps": [0, 1, 3]},
      "2048-32768": {"candidate_steps": [1, 3]}
    }
  },
  "32": {
    "ctx_buckets": {
      "1-767":      {"candidate_steps": [0]},
      "768-32768":  {"candidate_steps": [1, 3]}
    }
  },
  "64": {"candidate_steps": [0]}
}
```

- Detection rule: BS entry contains key `ctx_buckets` → 2D; otherwise treat
  as legacy single-bucket `[1, max_context]`. Legacy dicts still parse
  unchanged.
- Bucket key format: `"lo-hi"` (inclusive), integer tokens. Must fully cover
  `[1, max_context]` per BS with no gaps (validated at load time). Rule
  mirrors the vLLM RFC's rectangular-grid constraint.
- Per-bucket params inherit BS-level defaults (existing dict-merge in
  `AdaptiveSpeculativeParams.__init__` L279 already does this).

## `_route` extension

Change `_route(batch_size)` → `_route(batch_size, ctx_repr)`:

```python
def _route(
    self, batch_size: int, ctx_repr: int
) -> AdaptiveStepSlot:
    bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
    return self._slot_for_ctx(bs, ctx_repr)

def _slot_for_ctx(self, bs: int, ctx_repr: int) -> AdaptiveStepSlot:
    buckets = self._ctx_buckets_by_bs[bs]
    if len(buckets) == 1:
        return next(iter(buckets.values()))
    idx = bisect.bisect_right(self._ctx_hi_by_bs[bs], ctx_repr)
    return buckets[self._ctx_lo_by_bs[bs][idx]]
```

`_ctx_buckets_by_bs`, `_ctx_lo_by_bs`, `_ctx_hi_by_bs` are built in
`__init__` alongside `_slots` — one per BS. Legacy single-bucket entries
build with `_ctx_lo = [1]`, `_ctx_hi = [max_context]`, so the fast path
(`len(buckets) == 1`) covers all legacy configs at zero routing cost.

## Ctx representative

Same choice as vLLM RFC: **p50 of `batch.seq_lens_cpu`** at both call sites.
Rationale:

- Already CPU-resident (see scheduler.py L2647) — no new sync.
- p50 is robust to outliers (a single long sequence in a mostly-short batch
  doesn't flip the tier).
- SGLang spec v2 convention (`eagle_worker_v2.py:1084`): `batch.seq_lens` is
  the length **before** this iter's tokens. Within one verify cycle,
  `seq_lens_cpu` is not mutated between `activate_step_by_batch`
  (pre-dispatch) and `on_verify_complete_cpu` (post-CPU-sync); both sites
  observe the identical pre-iter snapshot. `seq_lens_cpu` advances only
  once per cycle, at `process_batch_result` (`scheduler.py:3408`),
  becoming next cycle's pre-iter value.
- Physically identical to vLLM RFC's `num_computed_tokens`: both name the
  pre-iter processed length, which is what KV-read cost is proportional to.
  The dose-response curve's independent variable maps 1:1 across engines.

## Callers — signature changes

Three sites, each a one-liner change to pass ctx:

**`adaptive_runtime_state.py`** (~L113, L118):

```python
def activate_step_by_batch(self, batch_size: int, ctx_repr: int) -> None:
    target = self.params.get_steps_for_batch(batch_size, ctx_repr)
    ...

def on_verify_complete(
    self, num_correct_drafts_per_req: list[int],
    batch_size: int, ctx_repr: int,
) -> None:
    new_step = self.params.on_verify_complete(
        num_correct_drafts_per_req, batch_size, ctx_repr,
    )
    ...
```

**`base_spec_worker.py` / `eagle_worker_v2.py` / `ngram_worker.py`** —
`on_verify_complete_cpu` grows a `ctx_repr` kwarg, threaded through to the
adaptive controller.

**`scheduler_components/batch_result_processor.py:548`** — the only real call
site. `batch.seq_lens_cpu` is a torch tensor here (int64, on CPU). Compute
once:

```python
ctx_repr = int(batch.seq_lens_cpu.median().item()) if len(batch.reqs) else 0
self.model_worker.on_verify_complete_cpu(
    result.num_correct_drafts_per_req_cpu,
    batch_size=len(batch.reqs),
    ctx_repr=ctx_repr,
)
```

For `activate_step_by_batch` (called before dispatch, not from this file):
the scheduler already has `batch.seq_lens_cpu` populated at dispatch time
(scheduler.py L2575). Add a symmetric one-liner in the pre-dispatch path.

## CUDA-graph implications — none

`cuda_graph_bs_for_step` returns `cuda_graph_bs` values reaching a target
step. In the 2D schema, a BS slot's reachable steps are the **union** of its
ctx-buckets' `candidate_steps`. Adjust L321-324:

```python
return [
    v for v in self._cuda_graph_bs
    if step in self._steps_reachable_at_bs[self._find_closest_bs(v)]
]
```

where `_steps_reachable_at_bs[bs]` = `sorted(union(bucket.candidate_steps
for bucket in _ctx_buckets_by_bs[bs].values()))`. Captured graphs are the
same as if the config were flattened to 1D with the union palette — no new
graph shapes, no new memory.

## Backward compatibility — enumerated

- Legacy config with no `ctx_buckets` key anywhere → 2D layer degenerates to
  1D (one bucket per BS with range `[1, max_context]`). `_slot_for_ctx` hits
  the `len(buckets) == 1` fast path. Runtime cost delta: one dict lookup +
  one int compare.
- CLI `--speculative-adaptive` with no `--speculative-adaptive-config` → uses
  `DEFAULT_ADAPTIVE_CONFIG` (still 1D). Unchanged.
- Callers of `on_verify_complete_cpu` that don't pass `ctx_repr` → keyword
  default `ctx_repr: int = 0` on the wrapper, treated as "shortest bucket".
  For legacy 1D configs this is inert (single bucket covers all ctx).

## Adaptive-unsupported guard

`adaptive_unsupported_reason` (L50-85) already lists what disqualifies
adaptive spec (non-EAGLE, topk≠1, DP attention, multi-layer eagle, TBO,
PDmux). The 2D extension does not change this set — a 2D config only
becomes reachable once adaptive itself is supported.

## What can be validated today (unit-level, no CUDA)

- `_load_adaptive_config` parses both legacy and 2D schemas without error.
- Rectangular-coverage validation rejects gaps (`{"1-1023", "2048-32768"}`
  under one BS should fail; `{"1-2047", "2048-32768"}` should pass).
- `_route(bs=8, ctx=1000)` and `_route(bs=8, ctx=3000)` return different
  slots for the 2D example above; the same call on a legacy config returns
  the same slot regardless of ctx.
- `cuda_graph_bs_for_step` returns identical output when a 2D config is
  compared to a 1D config with the union `candidate_steps` palette.

## What blocks running-smoke today

1. **Local SGLang install** — source tree is present at `external/sglang`
   but no built `sglang` module (`ModuleNotFoundError` on `import sglang`).
   Need `pip install -e .` in a Python 3.11 venv with SGLang's kernel deps.
2. **V7 gemma4_mtp smoke on SGLang** — status external to DepthChart.
   The RFC's second-engine claim needs the same *mechanism* to reproduce on
   SGLang, not necessarily the same *numbers*. Any EAGLE/EAGLE3 model that
   SGLang serves is sufficient for the 2D routing smoke.
3. **SGLang-native drafter/target pair on H100** — the vLLM Gemma-4 MTP
   head is not a SGLang MTP; SGLang's adaptive path requires EAGLE/EAGLE3
   (adaptive_unsupported_reason L52-56). LLaMA-3-8B + EAGLE-LLaMA-3-8B is
   the standard smoke target.

## Plan

- **Today** (paper-diff, this file). No SGLang tree changes.
- **After V7 lands** — apply the diff on a `feature/2d-route` branch of
  `external/sglang`, run unit tests (parser + routing), post a draft PR on
  SGLang upstream referencing the vLLM RFC as the design source.
- **Cross-engine smoke** — pick an EAGLE model, one-hour sanity: reproduce
  the (B, ctx) shape (does K stay > 0 at long ctx / high batch, unlike a
  batch-only table). No claim about SGLang-specific numbers is needed;
  the mechanism replication is the claim.

## Project rules to consult before landing

Two SGLang `.claude/rules/` items apply:

- `modify-component-must-read.md` → `speculative-naming` skill (naming
  conventions for anything under `python/sglang/srt/speculative/`). Read
  before assigning names to `_ctx_buckets_by_bs`, `_steps_reachable_at_bs`,
  and any new config keys.
- `no-dataclasses.md` → new data containers must be `msgspec.Struct`. The
  existing `AdaptiveStepSlot` is a plain class (grandfathered); no new
  container is strictly needed for this diff (dicts of dicts suffice), but
  if a `CtxBucket` struct is added for clarity, it must be `msgspec.Struct`.
