from __future__ import annotations

import pytest

from speculative.controller import ControllerConfig, ScheduleRow


def _row(bs_lo, bs_hi, ctx_lo, ctx_hi, k) -> ScheduleRow:
    return ScheduleRow(bs_lo=bs_lo, bs_hi=bs_hi, ctx_lo=ctx_lo, ctx_hi=ctx_hi, k=k)


@pytest.fixture
def simple_config() -> ControllerConfig:
    return ControllerConfig(
        k_palette=[0, 1, 3],
        schedule_2d=[
            _row(1, 60, 0, 512, 3),
            _row(1, 60, 513, 100_000, 3),
            _row(61, 256, 0, 512, 0),
            _row(61, 256, 513, 100_000, 3),
        ],
    )


@pytest.fixture
def gemma_h100_config() -> ControllerConfig:
    # §3.2 confirmed dose-response cells; other cells tentative pending V4/§7
    palette = [0, 1, 3]
    ctx_bands = [(0, 256), (257, 768), (769, 1536), (1537, 100_000)]
    bs_bands = [(1, 60), (61, 128), (129, 256), (257, 1024)]
    grid = {
        (1, 60):     [3, 3, 3, 3],
        (61, 128):   [1, 3, 3, 3],
        (129, 256):  [0, 3, 3, 3],
        (257, 1024): [0, 0, 3, 3],
    }
    rows: list[ScheduleRow] = []
    for bs in bs_bands:
        for j, ctx in enumerate(ctx_bands):
            rows.append(_row(bs[0], bs[1], ctx[0], ctx[1], grid[bs][j]))
    return ControllerConfig(k_palette=palette, schedule_2d=rows)
