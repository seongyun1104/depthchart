# moe_ctx_union — Gate 0 expert-union tooling (MoE ctx-axis campaign)

Local, GPU-free half of Gate 0 (design doc §5-6). Aggregates the active-expert
union per decode step from a router's `topk_ids`, with a per-call counter that
detects the torch.compile trace-once trap. Duck-typed (no torch / no vLLM) so it
runs and tests in a bare env — same discipline as `dspark_trace_sim`.

## Confirmed on the actual codepath (2026-08-04, upstream `origin/main` + HF)

**Routing path (Gate 0 cond. 2 — desk half).** `gemma-4-26b-a4b` does **not** use
`grouped_topk` nor plain `topk`: `gemma4.py:366` builds its FusedMoE with a
`custom_routing_function` (Gemma4 `per_expert_scale` routing,
`gemma4_routing_function_torch` / `gemma4_fused_routing_kernel_triton`). Union
capture is at the `select_experts` **output** level, so it is agnostic to the
routing method — the custom router still emits `topk_ids`.

**Wiring coordinate.** `select_experts` was refactored out of `FusedMoE`
(`layer.py`, the design doc's stale coordinate) into
`vllm/model_executor/layers/fused_moe/router/fused_moe_router.py::FusedMoERouter.select_experts`.
Preferred rental mechanism is the **built-in** `RoutedExpertsCapturer`
(`enable_return_routed_experts=True`), which writes `topk_ids` to a device buffer
inside the forward — compiled-path safe, so the trace-once trap does not apply.
`ExpertUnionCollector.wrap()` + the per-call counter are the trace-once sanity
check and a monkeypatch fallback; the collector consumes `topk_ids` either way.

**Capturer output shape (confirmed, v1/outputs.py:140).** `RoutedExpertsTensors`
carries `routing_data` of shape
**`(num_scheduled_tokens, num_layers, num_experts_per_tok)`** (step-level, across
all requests) + `slot_mapping` `(num_scheduled_tokens,)`; the scheduler ingests it
via `routed_experts_mgr.store_batch(routing_data, slot_mapping)` (scheduler.py:1716).
Feed `routing_data` per step into `ExpertUnionCollector.record_step_routing()` — it
folds every `(token, layer, k)` into that step's union in one call. The per-call
counter / trace-once guard applies only to the `wrap()` monkeypatch path, since the
capturer already delivers all layers per step. (This interface conformance is the
free, source-anchored substitute for the local "CPU vLLM" test — no macOS vLLM
wheel exists, so the collector is pinned to the real tensor shape instead.)

**Spec (Gate 0 cond. 3 / design §4.1, resolves the "25.2B/26B" ambiguity).**
`google/gemma-4-26b-a4b-it` `text_config`: **num_experts = 128, top_k_experts = 8,
num_hidden_layers = 30**, hidden_size 2816. So the Gate 0 union range is
`8 <= union <= 128` per step (top_k .. num_experts).

## Gate 0 conditions and where they land

| cond. | check | here |
|---|---|---|
| 1 hook fires per step | `gate0_check().counter_ok` (call_count == steps x layers) | ✅ tested |
| 2 routing path | custom_routing at `gemma4.py:366`, capture at router output | ✅ above |
| 3 spec | 128 experts / top-k 8 / 30 layers | ✅ above |
| 4 union in range | `gate0_check().range_ok` (`8 <= union <= 128`) | ✅ tested |

Conditions 3-eager-overhead and the actual union numbers are the **rental** half
(run with `enable_return_routed_experts=True`, feed `RoutedExpertsLists` into
`ExpertUnionCollector`). Do not run measurement before Gate 0 passes (design §5).

## Tests

```bash
pytest tests/moe_ctx_union -v
```
