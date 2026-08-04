# DSpark H100 E2E Smoke — Result (2026-08-04)

Rental: Vast H100 PCIe (id 46096710), $1.868/hr, ~85 min ≈ $2.6. Instance destroyed.
GPU: H100 PCIe, driver 580.159.04, **compute_cap 9.0 (SM90/Hopper)**, 80GB.
vLLM built from **PR #47808 head `e399e1c7`** (`neuralmagic` `codex/dspark-capacity-realloc`,
via `git fetch origin pull/47808/head`; SHA matches the one cited in memory — mergify has
not moved it). Editable install, `VLLM_USE_PRECOMPILED=1`, torch 2.13.0, Python 3.10.

## Headline: SM90/Hopper is NOT the blocker. The Gemma DSpark path has PR-side model bugs.

Everything hardware/architecture-related on Hopper initialised cleanly. The two smoke
attempts failed inside the **Gemma4 DSpark draft-model class**, on ANY hardware.

### Confirmed on H100/SM90 (the enablement hypotheses)
- **H1 (FA3):** `vllm.v1.attention.backends.fa_utils.get_flash_attn_version()` → **3**.
  → `flash_attn.py` `_cudagraph_support = AttentionCGSupport.ALWAYS` (ragged qlen>1 spec
  decode + full CG supported on SM90). *(Note: correct import path is
  `vllm.v1.attention.backends.fa_utils`, not the `vllm.attention.utils.fa_utils` the
  runbook first guessed — attention lives under `vllm/v1/attention/` now.)*
- **H0 (V2 runner):** `vllm/config/vllm.py:593` `speculative_config.method == "dspark"`
  forces V2. No env var needed. Confirmed present on e399e1c7.
- **FA backend admits SM90:** `flash_attn.py:201-202` `capability >= (8,0)`. No SM100 gate.
- Target `google/gemma-4-12B-it` (23.9GB, not gated for this token) + draft
  `dspark_gemma4_12b_block7` (`Gemma4DSparkModel`, 6.86GB) both load through model init.

### The two failures (both Gemma DSpark model-impl gaps, not hardware)
1. **adaptive verification ON** (`enable_adaptive_verification=True`, the default):
   `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:108` reads
   `model.model.confidence_head`, but `Gemma4DSparkModel` (inherits `DFlashQwen3Model`)
   never defines it — and `gemma4_dspark.py:297` **explicitly skips `confidence_head`
   weights** on load, though the checkpoint ships `confidence_head.proj.{weight,bias}`.
   → `AttributeError: 'Gemma4DSparkModel' object has no attribute 'confidence_head'`.
   Contrast: `Qwen3DSparkModel` (`qwen3_dspark.py:131-139`) builds `self.confidence_head`
   conditionally on `config.enable_confidence_head`.
2. **adaptive verification OFF:** KV-cache init →
   `dflash/speculator.py:212 set_attn` → `qwen3_dflash.py:720 get_draft_attn_causal`
   iterates draft attn layers expecting `.causal`, but `Gemma4DSparkAttention`
   (inherits `Gemma4MTPAttention`) has no `.causal`.
   → `AttributeError: 'Gemma4DSparkAttention' object has no attribute 'causal'`.

Full tracebacks: `smoke_adaptive_on.log`, `smoke_adaptive_off.log`.

## Interpretation
- The old "DSpark = SM100-only" premise is now **triply falsified**: static source read (#6),
  runtime FA3=3 on SM90, and the fact that the failures are model-implementation, not
  hardware. Nothing SM-gated blocked the run.
- The released **Gemma4 DSpark checkpoint does not run end-to-end** in #47808 @ e399e1c7,
  on any GPU, because `Gemma4DSparkModel` / `Gemma4DSparkAttention` are under-implemented
  relative to the Qwen3 path.
- To positively confirm **adaptive verification runs on H100**, the vehicle is the **Qwen3
  DSpark path** (`Qwen3DSparkModel` has `confidence_head`): `dspark_qwen3_8b_block7` +
  a Qwen3-8B target (dense, FA3, SM90). Not yet run.

## Gate before any public comment
Do NOT post yet: our first #47808 comment is <1 day old with no human reply
([[feedback-oss-pr-comment-timing]] 3-check). If reported later, the bug is a precise,
reproducible contribution (file:line + checkpoint/model mismatch), gated by
[[feedback-public-artifact-verification-gate]] (re-verify on live branch head first).
