from __future__ import annotations

from speculative.controller.ema import EMAAccept
from speculative.controller.schema import EMAConfig


def _cfg(**overrides) -> EMAConfig:
    return EMAConfig(**overrides)


def test_no_adjust_before_warmup():
    ema = EMAAccept(_cfg(warmup_batches=10))
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3
    for _ in range(5):
        ema.observe([2, 2, 2])
    assert not ema.is_warm
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_no_adjust_when_k_lut_zero():
    ema = EMAAccept(_cfg(warmup_batches=1))
    ema.observe([0])
    assert ema.adjust(k_lut=0, k_palette=[0, 1, 3]) == 0


def test_downward_correction_when_acceptance_low():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([1, 1, 1])
    assert ema.value == 1.0
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_no_downgrade_when_acceptance_high():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([3, 2, 2, 3])
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_never_upgrades():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([3, 3, 3])
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 1


def test_downgrade_to_zero_when_palette_bottom():
    cfg = _cfg(warmup_batches=1, update_interval=1, ema_alpha=1.0)
    ema = EMAAccept(cfg)
    for _ in range(3):
        ema.observe([0, 0, 0])
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 0


def test_empty_batch_ignored():
    ema = EMAAccept(_cfg(warmup_batches=1))
    ema.observe([])
    assert not ema.is_warm
    assert ema.value is None


def test_update_interval_gates_ema():
    cfg = _cfg(warmup_batches=1, update_interval=5, ema_alpha=0.5)
    ema = EMAAccept(cfg)
    ema.observe([2, 2])
    assert ema.value == 2.0
    for _ in range(4):
        ema.observe([0, 0])
    assert ema.value == 2.0
    ema.observe([0, 0])
    assert ema.value == 1.0
