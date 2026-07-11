from __future__ import annotations

from speculative.controller.overload import OverloadDetector
from speculative.controller.schema import OverloadConfig


def test_kv_usage_triggers_overload():
    od = OverloadDetector(OverloadConfig(kv_usage_pct=90.0))
    od.observe(kv_usage_pct=85.0, preempts=0, queue_depth=0)
    assert not od.overloaded
    od.observe(kv_usage_pct=91.0, preempts=0, queue_depth=0)
    assert od.overloaded


def test_preempt_streak_triggers_overload():
    od = OverloadDetector(OverloadConfig(preempt_sustained=3))
    for _ in range(2):
        od.observe(kv_usage_pct=50.0, preempts=1, queue_depth=0)
    assert not od.overloaded
    od.observe(kv_usage_pct=50.0, preempts=1, queue_depth=0)
    assert od.overloaded


def test_preempt_streak_resets_on_zero():
    od = OverloadDetector(OverloadConfig(preempt_sustained=3))
    od.observe(kv_usage_pct=50.0, preempts=1, queue_depth=0)
    od.observe(kv_usage_pct=50.0, preempts=1, queue_depth=0)
    od.observe(kv_usage_pct=50.0, preempts=0, queue_depth=0)  # reset
    od.observe(kv_usage_pct=50.0, preempts=1, queue_depth=0)
    assert not od.overloaded


def test_queue_depth_threshold_optional():
    od = OverloadDetector(OverloadConfig(queue_depth=100))
    od.observe(kv_usage_pct=50.0, preempts=0, queue_depth=50)
    assert not od.overloaded
    od.observe(kv_usage_pct=50.0, preempts=0, queue_depth=200)
    assert od.overloaded
