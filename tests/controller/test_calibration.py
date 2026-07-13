from __future__ import annotations

import pytest

from speculative.controller.ema import EMAAccept
from speculative.controller.schema import EMAConfig

HEALTHY_MTP_86_66_50 = [3] * 50 + [2] * 16 + [1] * 20 + [0] * 14
CELL_B_MAL_1_55 = [3] * 25 + [2] * 25 + [1] * 30 + [0] * 20
DEGRADED_AT_K3_MAL_0_80 = [1] * 80 + [0] * 20


def _instant_ema() -> EMAAccept:
    return EMAAccept(EMAConfig(warmup_batches=1, update_interval=1, ema_alpha=1.0))


def test_healthy_mtp_signature_holds_k3():
    ema = _instant_ema()
    ema.observe(HEALTHY_MTP_86_66_50, k_applied=3)
    assert ema.value == pytest.approx(2.02 / 3, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 3


def test_cell_b_degraded_demotes_k3_to_k1():
    ema = _instant_ema()
    ema.observe(CELL_B_MAL_1_55, k_applied=3)
    assert ema.value == pytest.approx(1.55 / 3, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_degraded_at_k3_demotes_k3_to_k1():
    ema = _instant_ema()
    ema.observe(DEGRADED_AT_K3_MAL_0_80, k_applied=3)
    assert ema.value == pytest.approx(0.80 / 3, abs=1e-9)
    assert ema.adjust(k_lut=3, k_palette=[0, 1, 3]) == 1


def test_same_shape_healthy_at_k1_holds_k1():
    """Same raw shape as DEGRADED_AT_K3 but drafted at K=1: rate 0.80 → healthy.
    Rate normalization makes the semantic difference explicit."""
    ema = _instant_ema()
    ema.observe(DEGRADED_AT_K3_MAL_0_80, k_applied=1)
    assert ema.value == pytest.approx(0.80, abs=1e-9)
    assert ema.adjust(k_lut=1, k_palette=[0, 1, 3]) == 1
