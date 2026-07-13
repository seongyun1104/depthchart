from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class EMAConfig(BaseModel):
    ema_alpha: float = 0.2
    update_interval: int = 5
    warmup_batches: int = 10
    down_hysteresis: float = Field(
        default=-0.45,
        description="Accept-rate floor = 1.0 + down_hysteresis; EMA below this demotes K.",
    )
    up_hysteresis: float = 0.0


class OverloadConfig(BaseModel):
    kv_usage_pct: float = 90.0
    preempt_sustained: int = 3
    queue_depth: int | None = None
    action: str = "K=0"


class SmoothingConfig(BaseModel):
    b_ema_alpha: float = 0.3
    tier_dwell_steps: int = 5


class ScheduleRow(BaseModel):
    bs_lo: int = Field(ge=1)
    bs_hi: int = Field(ge=1)
    ctx_lo: int = Field(ge=0)
    ctx_hi: int = Field(ge=0)
    k: int = Field(ge=0)


class ControllerConfig(BaseModel):
    k_palette: list[int]
    schedule_2d: list[ScheduleRow]
    alpha_correction: EMAConfig = Field(default_factory=EMAConfig)
    overload_override: OverloadConfig = Field(default_factory=OverloadConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)

    @model_validator(mode="after")
    def _validate(self) -> ControllerConfig:
        rows = self.schedule_2d
        if not rows:
            raise ValueError("schedule_2d must not be empty")
        palette = set(self.k_palette)
        if 0 not in palette:
            raise ValueError("k_palette must include 0 (overload fallback target)")
        for r in rows:
            if r.k not in palette:
                raise ValueError(f"K={r.k} not in palette {sorted(palette)}")
            if r.bs_lo > r.bs_hi:
                raise ValueError(f"bs_lo>bs_hi in row {r}")
            if r.ctx_lo > r.ctx_hi:
                raise ValueError(f"ctx_lo>ctx_hi in row {r}")

        bs_pairs = sorted({(r.bs_lo, r.bs_hi) for r in rows})
        if bs_pairs[0][0] != 1:
            raise ValueError("bs axis must start at 1")
        for (_, hi), (lo, _) in zip(bs_pairs, bs_pairs[1:], strict=False):
            if hi + 1 != lo:
                raise ValueError(f"bs ranges non-contiguous after bs_hi={hi}")

        by_bs: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r in rows:
            by_bs.setdefault((r.bs_lo, r.bs_hi), []).append((r.ctx_lo, r.ctx_hi))
        boundary_sets = {tuple(sorted(v)) for v in by_bs.values()}
        if len(boundary_sets) != 1:
            raise ValueError("ctx boundaries must be identical across all bs groups")
        ctx_pairs = sorted(next(iter(boundary_sets)))
        if ctx_pairs[0][0] != 0:
            raise ValueError("ctx axis must start at 0")
        for (_, hi), (lo, _) in zip(ctx_pairs, ctx_pairs[1:], strict=False):
            if hi + 1 != lo:
                raise ValueError(f"ctx ranges non-contiguous after ctx_hi={hi}")

        expected_cells = len(bs_pairs) * len(ctx_pairs)
        if len(rows) != expected_cells:
            raise ValueError(
                f"schedule_2d has {len(rows)} rows, expected "
                f"{expected_cells} = {len(bs_pairs)} bs × {len(ctx_pairs)} ctx cells"
            )
        return self
