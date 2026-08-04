# MoE Gate 0 Runbook — real gemma-4-26b-a4b (go/no-go)

Gate 0 for the MoE ctx-axis campaign (design doc §5-6), run **directly on the real
target** `google/gemma-4-26b-a4b-it`. The OLMoE / proxy-model bypass tier is
**dropped**: verifying Gate 0 on the real model removes the proxy's generalisation
assumption entirely. Do not run any campaign measurement before Gate 0 passes.

Spec (confirmed): num_experts **128**, top_k_experts **8**, num_hidden_layers **30**,
text-only workload (vision path idle — state it in the report).

## Mechanism (supersedes the design doc's monkeypatch 2-pass)

Use the **built-in** `RoutedExpertsCapturer`: serve with
`enable_return_routed_experts=True`. It writes `topk_ids` to a device buffer inside
the forward and surfaces `RoutedExpertsTensors.routing_data` of shape
`(num_scheduled_tokens, num_layers=30, num_experts_per_tok=8)` per step
(v1/outputs.py:140; scheduler ingests via `store_batch`, scheduler.py:1716). This is
**compiled-path safe**, so the torch.compile trace-once trap does not apply and the
enforce-eager instrument pass is not required for correctness. Feed each step's
`routing_data` into `ExpertUnionCollector.record_step_routing()`.

```python
from moe_ctx_union.union_collector import ExpertUnionCollector, gate0_check
c = ExpertUnionCollector(num_experts=128, top_k=8)
# for each decode step, from the returned RoutedExpertsTensors.routing_data:
c.record_step_routing(routing_data)   # (tokens, 30, 8) -> folds into step union
...
report = gate0_check(c, expected_calls=num_decode_steps)  # capturer: 1 call/step
```

If the capturer path is unavailable for our arm, fall back to the `wrap()`
monkeypatch on `FusedMoERouter.select_experts` under `enforce_eager=True` (the
per-call counter then expects `num_decode_steps × 30 layers`; a compiled run would
show `call_count << that` = trace-once trap).

## Gate 0 pass conditions (design §5, verbatim intent)

1. **Hook fires every step.** `report.counter_ok` — capturer: `call_count ==
   num_decode_steps`; monkeypatch: `== steps × 30`. A gross shortfall = trace-once
   trap / capturer not wired.
2. **Routing path located.** `gemma-4-26b-a4b` uses `custom_routing_function`
   (`gemma4.py:366`), captured at the `FusedMoERouter.select_experts` output — desk
   half done; the run confirms `topk_ids` actually arrive.
3. **Instrumentation overhead sane.** capture overhead within TPOT +10% (the
   capture pass need not be perf-clean, but must not distort union dynamics).
4. **Union in range.** `report.range_ok` — every step's union in `[8, 128]`, over
   ≥100 decode steps (union converges).

`report.passed` = conditions 1 and 4 green. **Fail →** re-estimate with nsys; if the
estimate blows the session budget, **park the campaign** (union-less perf can't
attribute H-A vs H-B — design §5).

## After Gate 0 (same session, if green)

Proceed to design §6 sanity 1-5 (drafter compat, DSD×MoE stability, NVFP4, text-only,
spec confirm) then the §4.2-4.4 measurement grid. Verdict follows the §7 pre-registered
decision matrix (∂union/∂K regime map + TPOT). Raw JSON → depthchart; report =
regime-map figure + P1-P5.

## Session placement

This session's H100 rental runs (design order): DSpark H100 smoke → Phase 2A collect →
**this Gate 0 on real gemma-4-26b-a4b**. Green here means the MoE-dedicated measurement
session becomes a data-driven go/no-go. `#49652` gates only the separate V2 3-step
check, not this.
