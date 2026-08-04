# Gate 0 expert-union collector for the MoE ctx-axis campaign.
#
# Consumes per-call `topk_ids` from a router's `select_experts` and aggregates
# the active-expert union per decode step. Two things Gate 0 needs (design doc
# §5): (1) a per-call counter that detects the torch.compile trace-once trap —
# if select_experts is monkeypatched inside a compiled region the wrapper fires
# only at trace time, so call_count << decode_steps x moe_layers; (2) a union
# size in [top_k, num_experts].
#
# Duck-typed on purpose: no torch / no vLLM import, so it runs and tests in a
# bare env (same discipline as dspark_trace_sim). topk_ids may be nested lists
# or any object exposing .tolist() (e.g. a torch tensor). Mechanism-agnostic:
# feed it from a select_experts monkeypatch or from the built-in
# RoutedExpertsCapturer (enable_return_routed_experts) — it only sees topk_ids.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def to_expert_rows(topk_ids) -> list[list[int]]:
    """Normalise topk_ids to ``[token][k]`` ints without importing torch.

    Accepts a tensor-like (``.tolist()``), a nested sequence, or a flat
    sequence of ints (treated as one expert per token).
    """
    obj = topk_ids.tolist() if hasattr(topk_ids, "tolist") else topk_ids
    rows: list[list[int]] = []
    for tok in obj:
        if hasattr(tok, "tolist"):
            tok = tok.tolist()
        if isinstance(tok, int):
            rows.append([tok])
        else:
            rows.append([int(e) for e in tok])
    return rows


@dataclass
class ExpertUnionCollector:
    num_experts: int = 128
    top_k: int = 8
    _call_count: int = 0
    _step_experts: set[int] = field(default_factory=set)
    _union_per_step: list[int] = field(default_factory=list)

    def record(self, topk_ids) -> None:
        """Record one ``select_experts`` call's expert ids into the current step."""
        self._call_count += 1
        for tok in to_expert_rows(topk_ids):
            self._step_experts.update(tok)

    def end_step(self) -> int:
        """Close the current decode step; returns and logs its union size."""
        union = len(self._step_experts)
        self._union_per_step.append(union)
        self._step_experts.clear()
        return union

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def union_per_step(self) -> list[int]:
        return list(self._union_per_step)

    def mean_union(self) -> float:
        u = self._union_per_step
        return sum(u) / len(u) if u else 0.0

    def wrap(self, select_experts: Callable) -> Callable:
        """Return a wrapper around ``router.select_experts`` that records
        ``topk_ids`` on every call. ``select_experts`` returns
        ``(topk_weights, topk_ids)``; a bare-tensor return is also accepted."""

        def wrapped(*args, **kwargs):
            out = select_experts(*args, **kwargs)
            topk_ids = out[1] if isinstance(out, tuple) else out
            self.record(topk_ids)
            return out

        return wrapped


@dataclass(frozen=True)
class Gate0Report:
    call_count: int
    expected_calls: int
    counter_ok: bool
    union_min: int
    union_max: int
    range_ok: bool
    passed: bool


def gate0_check(collector: ExpertUnionCollector, expected_calls: int) -> Gate0Report:
    """Gate 0 conditions 1 and 4 (design doc §5).

    - counter_ok: the wrapper fired ``expected_calls`` times (= decode_steps x
      moe_layers). A trace-once trap shows up as call_count << expected.
    - range_ok: every recorded step's union is in ``[top_k, num_experts]``.
    """
    counter_ok = collector.call_count == expected_calls
    unions = collector.union_per_step
    union_min = min(unions) if unions else 0
    union_max = max(unions) if unions else 0
    range_ok = bool(unions) and all(
        collector.top_k <= u <= collector.num_experts for u in unions
    )
    return Gate0Report(
        call_count=collector.call_count,
        expected_calls=expected_calls,
        counter_ok=counter_ok,
        union_min=union_min,
        union_max=union_max,
        range_ok=range_ok,
        passed=counter_ok and range_ok,
    )
