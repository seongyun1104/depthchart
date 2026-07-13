from __future__ import annotations

from speculative.controller import SpecControllerCore


def test_decide_returns_k_from_lut(simple_config):
    ctrl = SpecControllerCore(simple_config)
    k = ctrl.decide(batch_size=30, ctx_repr=100)
    assert k == 3


def test_overload_forces_k_zero(simple_config):
    ctrl = SpecControllerCore(simple_config)
    ctrl.decide(batch_size=30, ctx_repr=100)
    ctrl.observe_pressure(kv_usage=95.0, preempts_delta=0, queue_depth=0)
    assert ctrl.decide(batch_size=30, ctx_repr=100) == 0


def test_dwell_prevents_tier_flap(simple_config):
    ctrl = SpecControllerCore(simple_config)
    for _ in range(20):
        ctrl.decide(batch_size=30, ctx_repr=100)
    tier_low = ctrl.current_tier
    ctrl.decide(batch_size=200, ctx_repr=100)
    assert ctrl.current_tier == tier_low


def test_dwell_allows_sustained_shift(simple_config):
    ctrl = SpecControllerCore(simple_config)
    for _ in range(20):
        ctrl.decide(batch_size=30, ctx_repr=100)
    tier_low = ctrl.current_tier
    for _ in range(30):
        ctrl.decide(batch_size=200, ctx_repr=100)
    assert ctrl.current_tier != tier_low


def test_observe_verify_feeds_ema(simple_config):
    ctrl = SpecControllerCore(simple_config)
    for _ in range(15):
        ctrl.observe_verify([1, 1, 1], k_applied=3)
    for _ in range(10):
        ctrl.decide(batch_size=30, ctx_repr=100)
    k = ctrl.decide(batch_size=30, ctx_repr=100)
    assert k == 1


def test_smoother_low_alpha_smooths_step(gemma_h100_config):
    ctrl = SpecControllerCore(gemma_h100_config)
    for bs in [30, 60, 90, 60, 30, 60, 90, 60, 30, 60]:
        ctrl.decide(batch_size=bs, ctx_repr=100)
    tier = ctrl.current_tier
    assert tier is not None
    bs_range, _ = ctrl.lut.tier_ranges(tier)
    assert bs_range[0] <= 60 <= bs_range[1] or bs_range[1] >= 60
