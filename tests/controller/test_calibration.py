from __future__ import annotations

from speculative.controller.ema import EMAAccept
from speculative.controller.schema import EMAConfig

HEALTHY_MTP_86_66_50 = [3] * 50 + [2] * 16 + [1] * 20 + [0] * 14
CELL_B_MAL_1_55 = [3] * 25 + [2] * 25 + [1] * 30 + [0] * 20
SEVERE_MAL_0_80 = [1] * 80 + [0] * 20


def _instant_ema() -> EMAAccept:
    return EMAAccept(EMAConfig(warmup_batches=1, update_interval=1, ema_alpha=1.0))


def test_healthy_mtp_signature_holds_k3():
    ema = _instant_ema()
    ema.observe(HEALTHY_MTP_86_66_50)
    assert abs(ema.value - 2.02) < 1e-9
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_cell_b_degraded_demotes_k3_to_k1():
    ema = _instant_ema()
    ema.observe(CELL_B_MAL_1_55)
    assert abs(ema.value - 1.55) < 1e-9
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_severe_degradation_demotes_k3_to_k1():
    ema = _instant_ema()
    ema.observe(SEVERE_MAL_0_80)
    assert abs(ema.value - 0.80) < 1e-9
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_severe_degradation_holds_k1_by_threshold():
    ema = _instant_ema()
    ema.observe(SEVERE_MAL_0_80)
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 1
