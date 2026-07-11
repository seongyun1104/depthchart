from __future__ import annotations

import pytest
from pydantic import ValidationError

from speculative.controller import ControllerConfig, ScheduleRow


def r(bs_lo, bs_hi, ctx_lo, ctx_hi, k):
    return ScheduleRow(bs_lo=bs_lo, bs_hi=bs_hi, ctx_lo=ctx_lo, ctx_hi=ctx_hi, k=k)


def test_valid_config(simple_config):
    assert simple_config.k_palette == [0, 1, 3]
    assert len(simple_config.schedule_2d) == 4


def test_empty_schedule_rejected():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(k_palette=[0, 3], schedule_2d=[])


def test_bs_axis_must_start_at_1():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 3],
            schedule_2d=[r(2, 60, 0, 512, 3), r(2, 60, 513, 999, 3)],
        )


def test_bs_ranges_non_contiguous_rejected():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 3],
            schedule_2d=[
                r(1, 60, 0, 999, 3),
                r(100, 200, 0, 999, 0),
            ],
        )


def test_k_must_be_in_palette():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 1],
            schedule_2d=[r(1, 60, 0, 999, 3)],
        )


def test_palette_must_include_zero():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[1, 3],
            schedule_2d=[r(1, 60, 0, 999, 3)],
        )


def test_ctx_boundaries_must_be_uniform():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 3],
            schedule_2d=[
                r(1, 60, 0, 512, 3),
                r(1, 60, 513, 999, 3),
                r(61, 128, 0, 999, 0),
            ],
        )


def test_ctx_must_start_at_0():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 3],
            schedule_2d=[r(1, 60, 100, 999, 3)],
        )


def test_all_cells_required():
    with pytest.raises((ValueError, ValidationError)):
        ControllerConfig(
            k_palette=[0, 3],
            schedule_2d=[
                r(1, 60, 0, 512, 3),
                r(1, 60, 513, 999, 3),
                r(61, 128, 0, 512, 0),
            ],
        )


def test_gemma_full_table_validates(gemma_h100_config):
    assert len(gemma_h100_config.schedule_2d) == 16
