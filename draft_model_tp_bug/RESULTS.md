# draft_model speculative decoding: FlashInfer allreduce-fusion workspace crash under TP>1

Measured 2026-08-11 on 2×H100 PCIe (80GB each), vLLM 0.27.1, greedy, `draft_model` method,
`num_speculative_tokens=1`. Raw server logs: `repro45669_logs.tgz`.

## Summary

`draft_model` speculative decoding crashes at engine init under tensor parallelism (TP>1)
when the **draft model's hidden_size is larger than the target model's**. The FlashInfer
TRT-LLM fused allreduce+RMSNorm workspace is sized from the **target** model's hidden_size
only; the draft model's forward then overflows it and `check_trtllm_allreduce_fusion_workspace_metadata`
raises to prevent an illegal memory access.

## Arms

| Arm | target (TP) | draft | draft_hidden vs target_hidden | result |
|-----|-------------|-------|-------------------------------|--------|
| A | Qwen3-30B-A3B (TP=1) | Qwen3-4B | 2560 > 2048 | **acceptance 145/173 = 83.8%** (normal) |
| B | Qwen3-30B-A3B (TP=2) | Qwen3-4B | 2560 > 2048 | **CRASH** (workspace validation) |
| C | Qwen3-30B-A3B (TP=2) | Qwen3-0.6B | 1024 < 2048 | **acceptance 133/182 = 73.1%**, coherent output |
| B-patched | Qwen3-30B-A3B (TP=2) | Qwen3-4B | 2560 > 2048 | **CRASH persists** (one-line fix insufficient) |

Baseline (A) and the draft<target case (C) both work. The crash is specific to
**draft_hidden > target_hidden AND TP>1**.

## Crash (arm B)

```
ValueError: Workspace validation failed:
  - token_num (8192) * hidden_dim (2560) exceeds workspace max_token_num (8192)
    * hidden_dim (2048). This may cause Illegal Memory Access.
```
Call path: draft `dummy_run` -> qwen3 forward -> `flashinfer_trtllm_fused_allreduce_norm`
(from the `allreduce_rms_fusion` compilation pass) -> `flashinfer/comm/trtllm_ar.py:1026
check_trtllm_allreduce_fusion_workspace_metadata`.

## Root cause

`vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1008`
```python
self.hidden_dim = config.model_config.get_hidden_size()   # target model only
```
`self.hidden_dim` drives the fused-allreduce workspace token budget
(`max_token_num = max_size // (hidden_dim * element_size)`). It never accounts for the
draft model, so when a `draft_model` with a larger hidden_size runs its own forward under
TP>1, `token_num * draft_hidden` exceeds the target-sized workspace.

## Fix is non-trivial (one-liner insufficient)

Patching line 1008 to `max(target_hidden, draft_hidden)` in the pass `__init__` did **not**
fix the crash (arm B-patched): the patched run still reports `hidden_dim (2048)` in the
validation error, i.e. the workspace metadata that gets checked is created elsewhere (not
from `self.hidden_dim` in this pass instance). A correct fix has to reach the workspace
**creation** site so its stored hidden_dim covers the draft model.

## Relationship to existing work

Sibling of the in-flight FlashInfer-allreduce-under-draft-model cluster, but a **distinct
code path**:
- #50877 / PR #50932 — the **MNNVL** backend (`trtllm_mnnvl_ar`, vocab-embedding allreduce,
  Lamport buffer rotation). PR #50932 touches `flashinfer_all_reduce.py` /
  `fused_allreduce_gemma_rms_norm.py`.
- This report — the **TRT-LLM fused allreduce+RMSNorm** backend (`allreduce_rms_fusion.py` /
  `trtllm_ar.py`), hidden_dim sized from target only. Not covered by #50932.

Common theme across both: vLLM's FlashInfer allreduce workspace/buffer sizing derives from
the target model and does not account for the draft model's extra allreduce traffic.

## Note on issue #45669

This work started as an attempt to reproduce #45669 (draft_model produces degenerate
`token_id=0` output with Qwen3-235B target + Qwen3-4B draft, TP=4). That degenerate-output
symptom did **not** reproduce here (arm C, draft<target, TP=2 gives normal 73% acceptance
and coherent output). #45669 appears specific to TP=4 and/or 235B scale, which is beyond
2×H100. The crash documented here is a different, `draft_hidden > target_hidden` failure.
