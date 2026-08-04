# dspark_trace_sim

Offline trace-driven simulator for DSpark's confidence-scheduled verification
scheduler ([vllm-project/vllm#47808](https://github.com/vllm-project/vllm/pull/47808)).
Answers questions like "does EMA smoothing on the budget-side confidence snapshot
matter compared to the raw 2-step-prior value the paper describes?" without
needing a serving stack.

## Scope

**In scope**: offline measurements of acceptance rate, budget efficiency, and
ranking preservation across synthetic batches assembled from single-request
traces.

**Out of scope**: throughput, latency, absolute serving behaviour. Trace replay
assumes request-independent progression — valid for **ranking policies on
confidence signals**, not for **absolute serving behaviour**. Results from this
tool must not be presented as throughput.

## Two-phase pipeline

- **Phase 2A — Collection (GPU required, one-off per checkpoint × dataset subset)**.
  Runs [DeepSpec](https://github.com/deepseek-ai/DeepSpec)'s `eval.py` with a
  DSpark draft checkpoint on the target model. A hook records per-step confidence
  probabilities and per-position accept masks into canonical JSONL traces.

- **Phase 2B — Replay (GPU-free, infinitely re-runnable)**. Loads the collected
  traces, synthesises batches, and replays scheduling policies over an
  `(EMA α × staleness × batch size)` grid, comparing each cell to a post-hoc
  oracle. Adding a new α value or a new policy variant re-runs 2B locally with
  no additional collection.

This scaffold ships the trace format, the hook logger, and the DeepSpec adapter
(Phase 2A machinery). The scheduler / replay / oracle / orchestrator modules
(Phase 2B) land in follow-up commits.

## Canonical trace format

JSONL, one file per sample. The first line is a provenance header; subsequent
lines are per-step records.

```jsonl
{"__provenance__": true, "trace_schema_version": "1.0", "deepspec_commit": "005e03b8...", "checkpoint_id": "deepseek-ai/dspark_gemma4_12b_block7", "checkpoint_revision": "<HF revision SHA>", "target_model": "google/gemma-4-12B-it", "dataset": "gsm8k", "sampling_config": {"temperature": 1.0, "confidence_threshold": 0.0, "seed": 980406}, "collected_at": "2026-XX-XXTXX:XX:XXZ"}
{"sample_id": "gsm8k_042", "step_idx": 0, "confidences": [0.87, 0.62, 0.41, 0.29, 0.18, 0.11, 0.07], "accepts": [1, 1, 0, 0, 0, 0, 0], "prefix_len": 2}
```

### Field semantics

- `confidences` — **post-sigmoid probabilities in [0, 1]**, never logits. The
  conversion happens in `deepspec_adapter.to_plain_arrays` via
  `torch.sigmoid(proposal.confidence_logits)`. Storing logits here would silently
  break Phase 2B replay (the survival-probability cumprod assumes probs).
- `accepts` — 0/1 mask, position-aligned with `confidences`.
- `prefix_len` — leading-1 count of `accepts`. Kept as a duplicate of information
  in `accepts` for downstream convenience; the schema validates the two agree.
- Provenance fields are all mandatory. Their values are cheap at collection time
  and impossible to add retroactively; missing any of them makes traces
  non-replayable in principle.

## Layers

| Module | Purpose |
|---|---|
| `trace_format.py` | Provenance and StepRecord schemas (pydantic v2) + JSONL read/write helpers. No dependency on DeepSpec. |
| `logger.py` | `TraceLogger`, per-run driver. Accepts plain float/int arrays. No dependency on DeepSpec. |
| `deepspec_adapter.py` | `to_plain_arrays(proposal, verification)` — bridges DeepSpec runtime objects to the plain arrays the logger consumes. Interface pinned to a specific DeepSpec commit; re-verify at Phase 2A collection time. |

The DeepSpec dependency is confined to `deepspec_adapter.py` on purpose: if
DeepSpec renames a field between now and Phase 2A collection, the change is
localised to ~30 lines rather than the whole pipeline.

## Running the tests

```bash
pytest tests/dspark_trace_sim -v
```
