# DSpark adaptive verification on H100 — Qwen3 vehicle (2026-08-04, session 2)

Follow-up to the gemma-4-12b smoke (which hit `Gemma4DSparkModel` PR bugs). The
Qwen3 dspark path (`Qwen3DSparkModel` builds `confidence_head`) is the working
vehicle. Same H100 PCIe (SM90), same #47808 build `e399e1c7`.

## Checkpoint (the difference from gemma)
`deepseek-ai/dspark_qwen3_8b_block7`: arch `Qwen3DSparkModel`,
**`enable_confidence_head: True`**, weights include `confidence_head.proj.{weight,bias}`
→ the model class actually builds the confidence head (unlike Gemma4DSparkModel).
Target `Qwen/Qwen3-8B`.

## Result: adaptive verification RUNS on H100/SM90 (with cudagraphs)

Config = the PR's own E2E test shape: `method=dspark`, `num_speculative_tokens=7`,
`draft_sample_method=probabilistic`.

- **`enforce_eager=True` run** → boots, coherent output, AR 0.317 / AL 3.22 —
  **but** log: `DSpark adaptive verification disabled: no cudagraph cost profile
  is available. Falling back to fixed-length verification` (model_runner.py:1340).
  So eager exercises dspark *drafting* but **not adaptive verification**.
- **`enforce_eager=False` run** (`cudagraph_mode=FULL_AND_PIECEWISE`) → boots,
  coherent output, **no fallback warning → adaptive verification ACTIVE**.
  AR 0.173 / AL 2.21, num_drafts 203 / accepted 246.

## Findings
1. **DSpark adaptive verification is functional on H100/SM90** — confirmed at
   runtime with the Qwen3 vehicle under full cudagraphs. Combined with FA3==3 and
   the model-side attention path, the "SM100-only" premise is fully retired.
2. **Adaptive verification requires cudagraphs.** With `enforce_eager=True` there
   is no cudagraph cost profile, so it disables itself and falls back to fixed-K
   (model_runner.py:1340). The PR's own E2E test runs `enforce_eager=True`, so it
   does not itself exercise the adaptive path — worth noting.

Logs: `smoke_qwen3_adaptive_SUCCESS.log` (eager, fixed-K fallback),
`smoke_qwen3_cudagraph.log` (non-eager, adaptive active). Credential-scanned clean.

## B enablement comment
Draft only; publish gated (live re-fetch + comment-budget: our 1st #47808 comment
still has 0 human reaction). The headline material: adaptive verification runs on
Hopper (Qwen3, cudagraphs), and it requires cudagraph mode.
