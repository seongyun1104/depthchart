# MoE Gate 0 — PASSED on real gemma-4-26b-a4b (2026-08-04)

Ran Gate 0 directly on `google/gemma-4-26b-a4b-it` (OLMoE proxy dropped), H100 PCIe,
#47808 build `e399e1c7` (which carries `enable_return_routed_experts`).

## Mechanism (as designed)
`enable_return_routed_experts=True` → `CompletionOutput.routed_experts` numpy array
of shape **`[seq_len, num_layers, top_k]`** (outputs.py:45). Fed each token's
`[num_layers, top_k]` slice into `ExpertUnionCollector.record_step_routing`.

## Result

```
RE_SHAPE = (51, 30, 8)          # tokens x layers x top-k — matches the confirmed spec
ROUTED_NONE = 0                 # capturer fired for every sequence
STEPS = 214
UNION_MIN=100  UNION_MAX=121  MEAN=111.14   # per token, across 30 layers; in [8,128]
COUNTER_OK=True  RANGE_OK=True  PASSED=True
```

- **Gate 0 conditions 1 + 4 pass.** The routing shape is exactly `(tokens, 30, 8)`;
  union per step in `[100, 121]` (well inside `[8, 128]`).
- **compile-swallow trap does NOT occur.** This ran `enforce_eager=False`
  (`cudagraph_mode=FULL_AND_PIECEWISE`) and `routed_experts` was still populated on
  every sequence (`ROUTED_NONE=0`). The built-in capturer is compile-safe — so the
  design doc's eager 2-pass is unnecessary. Confirmed on the real model.

## Caveat / finding (config workaround)
`enable_return_routed_experts` crashes on gemma-4 out of the box:
`RoutedExpertsManager.__init__` (routed_experts_capturer.py:304) reads
`hf_config.num_experts_per_tok` **directly**, while the sibling `RoutedExpertsCapturer`
uses the `_get_num_experts_per_tok` helper that falls back to `top_k_experts`. gemma-4
has `top_k_experts`, not `num_experts_per_tok`, so the Manager throws
`AttributeError`. Workaround: add `num_experts_per_tok = top_k_experts (8)` to the
config. Proper fix (reportable): the Manager should use the same helper.

## Implication
The union mechanism works on the real gemma-4-26b-a4b under compilation → the
MoE-dedicated measurement session is now a **data-driven go/no-go**. Note the observed
per-token cross-layer union is already near-saturation (~111/128) at tiny batch; the
campaign's H-A/H-B measurement (per-layer union vs batch × K, design §4.2) is the
follow-up. Log: `gate0_gemma4_26b_PASS.log`.
