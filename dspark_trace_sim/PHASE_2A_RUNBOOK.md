# Phase 2A — Trace Collection Runbook

One-off GPU collection that turns a DSpark eval run into the canonical JSONL
traces the Phase 2B replay (`replay.py` / `orchestrator.py`) consumes. Run once
per `(checkpoint × dataset subset)`; the grid replay is then GPU-free and
infinitely re-runnable.

> **Scope.** These traces measure acceptance rate and budget efficiency
> offline. They do **not** measure throughput or latency. Any artifact derived
> from them must carry the request-independence caveat and must not be framed as
> a throughput result.

## 0. Prerequisites

- **Hardware**: a single H100 (SM90) is enough. Base DSpark is Hopper-compatible
  (field-reported: 2×H100 NVL sm_90, TP=2). The SM100 gate is specific to
  `#47808`'s adaptive-verification FP4 indexer path, **not** to this eval.
- **`HF_TOKEN` exported before any pull.** Authenticated pulls are ~3 min vs
  ~25 min unauthenticated; wall time is billed. Set it in the shell profile on
  the rig, not inline per-command.
- **DeepSpec checkout** at the pinned commit
  `deepseek-ai/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`.
- **Models**: target `google/gemma-4-12B-it`, drafter
  `deepseek-ai/dspark_gemma4_12b_block7`. Record the resolved HF revision SHA of
  each — it goes in the provenance header and cannot be added later.

## 1. RE-VERIFY the DeepSpec adapter (blocking, do first)

`deepspec_adapter.py` is pinned to the commit above and reads exactly three
fields. Before collecting, diff the pin against the live checkout:

- `deepspec/eval/dspark/draft_ops.py::DSparkDraftProposal.confidence_logits`
- `deepspec/eval/base_evaluator.py::VerificationResult.accept_prefix_mask`
- `deepspec/eval/base_evaluator.py::VerificationResult.accepted_draft_tokens`

If any name drifted, update `deepspec_adapter.py` **only** — the logger and the
whole 2B pipeline consume plain arrays and are shielded from DeepSpec renames.
Do not start a collection run against an unverified adapter; a silent field
rename corrupts every trace and is invisible until 2B replay.

## 2. Dataset subset

Collect three datasets whose acceptance profiles diverge, so the policy
comparison sees a range rather than one regime:

| dataset | regime | acceptance profile |
|---|---|---|
| `gsm8k` | reasoning | high, long accepted runs |
| `mt-bench` | dialogue | mid |
| `humaneval` | code | distinct (structured tokens) |

Start with a small fixed sample count per dataset (e.g. 64–128). Statistical
power in 2B comes from batch resampling (`num_batches`), so 2A only needs enough
distinct request traces to synthesise diverse batches. This is a one-off fixed
cost and can be extended later without touching 2B.

## 3. Hook wiring

Collection reuses the existing `confidence_head_recorder` call site; add the
`TraceLogger` alongside it, no changes to DeepSpec's own recorder.

```python
from pathlib import Path
from dspark_trace_sim.deepspec_adapter import to_plain_arrays
from dspark_trace_sim.logger import TraceLogger
from dspark_trace_sim.trace_format import Provenance

# In Gemma4DSparkEvaluator.__init__:
self.trace_logger = TraceLogger(
    out_dir=Path("~/traces/dspark_gemma4_12b_block7").expanduser(),
    provenance=Provenance(
        deepspec_commit="005e03b81cec38b7da6399833d609ee89a2587f2",
        checkpoint_id="deepseek-ai/dspark_gemma4_12b_block7",
        checkpoint_revision="<resolved HF revision SHA>",
        target_model="google/gemma-4-12B-it",
        dataset="<gsm8k|mt-bench|humaneval>",   # one logger per dataset run
        sampling_config={"temperature": 1.0, "confidence_threshold": 0.0,
                         "seed": 980406},
        collected_at="<ISO-8601 UTC>",
    ),
)

# In generate_one_sample, around the per-sample loop:
self.trace_logger.start_sample(sample_id, dataset)
...
self.trace_logger.end_sample()

# In _post_verify, after super()._post_verify(...):
confidences, accepts, prefix_len = to_plain_arrays(proposal, verification)
self.trace_logger.observe(confidences, accepts, prefix_len)
```

The provenance header is written once per sample file automatically by
`start_sample`. All seven provenance fields are mandatory and validated on read.

## 4. Run

```bash
export HF_TOKEN=...            # already in profile per §0
python eval.py \
  --target=google/gemma-4-12B-it \
  --draft=deepseek-ai/dspark_gemma4_12b_block7 \
  --dataset=gsm8k \
  --confidence-threshold=0
# repeat for mt-bench, humaneval
```

`--confidence-threshold=0` records every step's confidence (no admission
filtering at collection time; filtering is a 2B replay policy).

Output layout: `~/traces/dspark_gemma4_12b_block7/{dataset}/{sample_id}.jsonl`,
first line a provenance header, subsequent lines per-step records.

## 5. Validate before egress

```bash
python -c "
from pathlib import Path
from dspark_trace_sim.replay import load_pool
prov, traces = load_pool(Path('~/traces/dspark_gemma4_12b_block7').expanduser())
print(len(traces), 'traces;', len(prov), 'provenance headers')
print('total steps:', sum(len(t.steps) for t in traces))
"
```

`load_pool` runs the full pydantic schema validation (probs in [0,1], accepts
binary, `prefix_len` == leading-1 count, provenance complete). A malformed trace
raises here rather than silently in 2B.

Then run a credential scan over the trace directory before copying anything off
the rig; traces are payload-free (probabilities and accept masks only), but the
scan is a hard gate on any artifact leaving the box.

## 6. Measure the SPS curve (primary budget-path input)

The budget path's throughput-optimal K depends on the steps-per-second curve.
The synthetic `power_law_sps` family keeps replay non-degenerate locally, but
**every relative comparison then stands on an arbitrary curve shape** — a verdict
that flips between exponents is a curve artifact, not a finding. Measure the real
curve on the rig while it is up (a few-minute probe; `orchestrator` already takes
`sps` as a knob):

- For a handful of batch sizes `b` (e.g. 1, 2, 4, 8, 16, 32), run the target a
  fixed number of decode steps and record steps/sec.
- Fit a monotone-decreasing curve (or interpolate) and expose it as an
  `Sps = Callable[[int], float]`.

Feed the **measured** curve as the primary `sps` into `run_grid`. Demote the
synthetic family to a sensitivity check.

**Verdict protocol (pre-registered).** A verdict is confirmed only if it is
stable across the SPS family — the measured curve plus synthetic exponents
`{0.3, 0.5, 0.7}` (`power_law_sps(exp)`). If the verdict flips within the family,
report it as **SPS-dependent** rather than picking one curve.

## 7. Handoff to Phase 2B (local, GPU-free)

```python
from pathlib import Path
from dspark_trace_sim.replay import load_pool, power_law_sps
from dspark_trace_sim.orchestrator import run_grid, to_markdown_table

provenances, pool = load_pool(Path("~/traces/dspark_gemma4_12b_block7").expanduser())

measured_sps = ...                            # from §6, primary
report = run_grid(provenances, pool, sps=measured_sps)
print(to_markdown_table(report))              # PR-convention table + verdict

# SPS sensitivity: confirm the verdict is stable across the family.
family = {exp: run_grid(provenances, pool, sps=power_law_sps(exp)).verdict
          for exp in (0.3, 0.5, 0.7)}
```

Grid axes and report labels are in the PR convention (`confidence_ema_alpha`);
the internal EMA-retention alpha stays inside the replay modules. Confirm the
grid α values against the live `#47808` default before pasting any table into a
public comment.

## Provenance fields (all mandatory)

| field | source | why |
|---|---|---|
| `deepspec_commit` | `git rev-parse HEAD` in the DeepSpec checkout | reproducibility of the eval harness |
| `checkpoint_id` | drafter model id | which draft head produced the confidences |
| `checkpoint_revision` | resolved HF revision SHA | model cards move; pin the exact weights |
| `target_model` | target model id | verifier identity |
| `dataset` | one of the §2 subset | acceptance regime label |
| `sampling_config` | temperature / threshold / seed | run determinism |
| `collected_at` | ISO-8601 UTC | ordering multiple collection runs |
