from __future__ import annotations

import random

import pytest

from moe_ctx_union.union_collector import (
    ExpertUnionCollector,
    gate0_check,
    to_expert_rows,
)


class _FakeTensor:
    """Duck-types a torch tensor for the collector (only .tolist() is used)."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeRouter:
    """Scripted stand-in for FusedMoERouter; returns (topk_weights, topk_ids)."""

    def __init__(self, script: list[list[list[int]]]):
        self.script = script
        self.calls = 0

    def select_experts(self, hidden_states, router_logits):
        ids = self.script[self.calls % len(self.script)]
        self.calls += 1
        weights = [[1.0] * len(tok) for tok in ids]
        return weights, ids


# ----- to_expert_rows duck-typing -----

def test_to_expert_rows_nested_list():
    assert to_expert_rows([[0, 3, 7], [1, 3, 5]]) == [[0, 3, 7], [1, 3, 5]]


def test_to_expert_rows_tensor_like():
    assert to_expert_rows(_FakeTensor([[2, 4], [4, 6]])) == [[2, 4], [4, 6]]


def test_to_expert_rows_flat_ints():
    assert to_expert_rows([1, 2, 3]) == [[1], [2], [3]]


def test_to_expert_rows_rows_are_tensor_like():
    assert to_expert_rows([_FakeTensor([1, 2]), _FakeTensor([2, 3])]) == [[1, 2], [2, 3]]


# ----- union aggregation -----

def test_union_is_distinct_experts_across_tokens():
    c = ExpertUnionCollector(num_experts=128, top_k=2)
    c.record([[0, 1], [1, 2], [2, 3]])  # distinct = {0,1,2,3}
    assert c.end_step() == 4


def test_union_saturates_at_identical_routing():
    c = ExpertUnionCollector(num_experts=128, top_k=3)
    c.record([[5, 6, 7]] * 10)  # all tokens same 3 experts
    assert c.end_step() == 3


def test_union_accumulates_across_calls_within_a_step():
    """Multiple select_experts calls (layers) fold into one step's union."""
    c = ExpertUnionCollector()
    c.record([[0, 1]])
    c.record([[1, 2]])
    c.record([[9]])
    assert c.end_step() == 4  # {0,1,2,9}


def test_end_step_resets_and_logs():
    c = ExpertUnionCollector()
    c.record([[0, 1]])
    c.end_step()
    c.record([[5, 6, 7]])
    c.end_step()
    assert c.union_per_step == [2, 3]
    assert c.mean_union() == pytest.approx(2.5)


# ----- per-call counter (trace-once trap detector) -----

def test_call_count_increments_per_record():
    c = ExpertUnionCollector()
    for _ in range(5):
        c.record([[0, 1]])
    assert c.call_count == 5


def test_wrap_records_and_passes_through():
    c = ExpertUnionCollector()
    router = _FakeRouter([[[0, 1, 2]], [[3, 4, 5]]])
    wrapped = c.wrap(router.select_experts)

    w, ids = wrapped("h", "logits")
    assert ids == [[0, 1, 2]]      # passthrough intact
    assert w == [[1.0, 1.0, 1.0]]
    assert c.call_count == 1


def test_wrap_over_many_steps_matches_call_count():
    c = ExpertUnionCollector()
    router = _FakeRouter([[[0, 1]], [[2, 3]]])
    wrapped = c.wrap(router.select_experts)
    n_layers = 4
    n_steps = 6
    for _ in range(n_steps):
        for _ in range(n_layers):
            wrapped("h", "logits")
        c.end_step()
    assert c.call_count == n_steps * n_layers
    assert len(c.union_per_step) == n_steps


# ----- Gate 0 check -----

def _fill(c: ExpertUnionCollector, steps: int, layers: int, experts_per_call):
    for _ in range(steps):
        for _ in range(layers):
            c.record(experts_per_call)
        c.end_step()


def test_gate0_passes_when_counter_and_range_ok():
    c = ExpertUnionCollector(num_experts=128, top_k=8)
    # each call routes 8 distinct experts, union per step in [8,128]
    call = [[list(range(8))]]  # one token, 8 experts
    _fill(c, steps=10, layers=3, experts_per_call=call[0])
    rep = gate0_check(c, expected_calls=10 * 3)
    assert rep.counter_ok
    assert rep.range_ok
    assert rep.passed
    assert rep.union_min >= 8 and rep.union_max <= 128


def test_gate0_detects_trace_once_trap():
    """If the wrapper only fired at trace time, call_count << expected."""
    c = ExpertUnionCollector(num_experts=128, top_k=8)
    c.record([list(range(8))])  # fired once (trace) instead of steps*layers
    c.end_step()
    rep = gate0_check(c, expected_calls=10 * 3)
    assert not rep.counter_ok
    assert not rep.passed


def test_gate0_range_fails_below_top_k():
    c = ExpertUnionCollector(num_experts=128, top_k=8)
    c.record([[0, 1]])  # union 2 < top_k 8
    c.end_step()
    rep = gate0_check(c, expected_calls=1)
    assert rep.counter_ok
    assert not rep.range_ok
    assert not rep.passed


def test_gate0_range_fails_above_num_experts():
    c = ExpertUnionCollector(num_experts=16, top_k=2)
    c.record([[i] for i in range(20)])  # 20 distinct > 16 experts (impossible → guard)
    c.end_step()
    rep = gate0_check(c, expected_calls=1)
    assert not rep.range_ok


def test_gate0_empty_is_not_passing():
    c = ExpertUnionCollector()
    rep = gate0_check(c, expected_calls=0)
    assert not rep.range_ok
    assert not rep.passed


# ----- capturer path: RoutedExpertsTensors.routing_data 3D shape -----
# routing_data shape = (num_scheduled_tokens, num_layers, num_experts_per_tok)

def test_record_step_routing_folds_all_tokens_layers_k():
    c = ExpertUnionCollector(num_experts=128, top_k=2)
    # 2 tokens x 2 layers x 2 experts; distinct = {0,1,2,3,4,5,9}
    routing_data = [
        [[0, 1], [2, 3]],   # token 0: layer0={0,1}, layer1={2,3}
        [[1, 4], [5, 9]],   # token 1
    ]
    union = c.record_step_routing(routing_data)
    assert union == 7
    assert c.union_per_step == [7]


def test_record_step_routing_tensor_like():
    c = ExpertUnionCollector(top_k=2)
    routing_data = _FakeTensor([[[0, 1], [1, 2]], [[3, 4], [4, 0]]])
    assert c.record_step_routing(routing_data) == 5  # {0,1,2,3,4}


def test_record_step_routing_gemma_shape_passes_gate0():
    """Realistic gemma-4-26b-a4b step: (tokens, 30 layers, top-8), union in [8,128]."""
    rng = random.Random(0)
    c = ExpertUnionCollector(num_experts=128, top_k=8)
    for _ in range(20):  # 20 decode steps
        routing_data = [
            [rng.sample(range(128), 8) for _ in range(30)]  # 30 layers x top-8
            for _ in range(4)  # 4 scheduled tokens
        ]
        c.record_step_routing(routing_data)
    rep = gate0_check(c, expected_calls=20)
    assert rep.passed
    assert rep.union_min >= 8 and rep.union_max <= 128


def test_record_step_routing_counts_one_call_per_step():
    c = ExpertUnionCollector(top_k=1)
    c.record_step_routing([[[0]], [[1]]])
    c.record_step_routing([[[2]], [[3]]])
    assert c.call_count == 2
    assert len(c.union_per_step) == 2
