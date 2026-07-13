from __future__ import annotations

import pytest

from speculative.controller.ema import EMAAccept
from speculative.controller.schema import EMAConfig


def _cfg(**overrides) -> EMAConfig:
    return EMAConfig(**overrides)


def test_no_adjust_before_warmup():
    ema = EMAAccept(_cfg(warmup_batches=10))
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3
    for _ in range(5):
        ema.observe([2, 2, 2], k_applied=3)
    assert not ema.is_warm
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_no_adjust_when_k_lut_zero():
    ema = EMAAccept(_cfg(warmup_batches=1))
    ema.observe([0], k_applied=1)
    assert ema.adjust(k_lut=0, k_palette=[0, 1, 3]) == 0


def test_downward_correction_when_rate_low():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([1, 1, 1], k_applied=3)
    assert ema.value == pytest.approx(1 / 3, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_no_downgrade_when_rate_high():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([3, 2, 2, 3], k_applied=3)
    assert ema.value == pytest.approx(10 / 12, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_never_upgrades():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([1, 1, 1], k_applied=1)
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 1


def test_downgrade_to_zero_when_palette_bottom():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([0, 0, 0], k_applied=1)
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 0


def test_empty_batch_ignored():
    ema = EMAAccept(_cfg(warmup_batches=1))
    ema.observe([], k_applied=3)
    assert not ema.is_warm
    assert ema.value is None


def test_k_applied_zero_skipped_no_signal():
    ema = EMAAccept(_cfg(warmup_batches=1))
    ema.observe([2, 2, 2], k_applied=0)
    assert not ema.is_warm
    assert ema.value is None


def test_update_interval_gates_ema():
    cfg = _cfg(warmup_batches=1, update_interval=5, ema_alpha=0.5)
    ema = EMAAccept(cfg)
    ema.observe([2, 2], k_applied=3)
    assert ema.value == pytest.approx(2 / 3, abs=1e-9)
    for _ in range(4):
        ema.observe([0, 0], k_applied=3)
    assert ema.value == pytest.approx(2 / 3, abs=1e-9)
    ema.observe([0, 0], k_applied=3)
    assert ema.value == pytest.approx(1 / 3, abs=1e-9)


def test_return_to_k3_via_lut_holds_when_k1_rate_healthy():
    """Ratchet regression. v0.1 raw-count EMA: after K=3→K=1 demotion the EMA
    was capped by K=1 observations and could not clear a K=3-scaled threshold
    when L0 raised K back. v0.1.1 rate normalization makes the signal K-invariant."""
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    ema.observe([1] * 80 + [0] * 20, k_applied=1)
    assert ema.value == pytest.approx(0.8, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


@pytest.mark.xfail(
    reason="v0.1.1 known limit: sharp per-pos decay drafter flaps K=1↔K=3 near the "
    "0.55 rate floor (§3.3 ①). Fix = _demoted state bit + hold/recover threshold "
    "split; deferred to v0.2. Our production drafter (85/68) has enough margin.",
    strict=True,
)
def test_flap_prevention_via_demoted_bit_not_yet_implemented():
    """Hypothetical per-pos [60, 30, 15] drafter. K=3 rate 0.35 → demote. K=1
    rate 0.60 → clears 0.55 floor and would let L0 return K=3, whose rate reverts
    to 0.35 → flap. A demoted-state recover-threshold ~0.75 would hold at K=1."""
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    ema.observe([2] * 5 + [1] * 95, k_applied=3)
    assert ema.value == pytest.approx(0.35, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1
    ema.observe([1] * 60 + [0] * 40, k_applied=1)
    assert ema.value == pytest.approx(0.60, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1
