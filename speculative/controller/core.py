from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .ema import EMAAccept
from .lut import LUT2D, Tier
from .overload import OverloadDetector
from .schema import ControllerConfig


@runtime_checkable
class EngineAdapter(Protocol):
    def build_runtime_state(self, K: int) -> object: ...
    def apply_runtime_state(self, state: object) -> None: ...


class BSmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._ema: float | None = None

    def observe(self, bs: int) -> int:
        if self._ema is None:
            self._ema = float(bs)
        else:
            self._ema = self.alpha * bs + (1 - self.alpha) * self._ema
        return max(1, int(round(self._ema)))


class SpecControllerCore:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.lut = LUT2D(config)
        self.ema = EMAAccept(config.alpha_correction)
        self.overload = OverloadDetector(config.overload_override)
        self._smoother = BSmoother(config.smoothing.b_ema_alpha)
        self._current_tier: Tier | None = None
        self._pending_tier: Tier | None = None
        self._pending_dwell: int = 0
        self._dwell_target: int = config.smoothing.tier_dwell_steps

    def decide(self, batch_size: int, ctx_repr: int) -> int:
        bs_smoothed = self._smoother.observe(batch_size)
        proposed = self.lut.tier_of(bs_smoothed, ctx_repr)
        stable = self._apply_dwell(proposed)
        k_lut = self.lut.k_at(stable)
        k_adj = self.ema.adjust(k_lut, self.config.k_palette)
        if self.overload.overloaded:
            return 0
        return k_adj

    def observe_verify(self, accepted_per_req: Sequence[int]) -> None:
        self.ema.observe(accepted_per_req)

    def observe_pressure(
        self, kv_usage: float, preempts: int, queue_depth: int
    ) -> None:
        self.overload.observe(kv_usage, preempts, queue_depth)

    @property
    def current_tier(self) -> Tier | None:
        return self._current_tier

    def _apply_dwell(self, proposed: Tier) -> Tier:
        if self._current_tier is None:
            self._current_tier = proposed
            self._pending_tier = None
            self._pending_dwell = 0
            return proposed
        if proposed == self._current_tier:
            self._pending_tier = None
            self._pending_dwell = 0
            return self._current_tier
        if proposed == self._pending_tier:
            self._pending_dwell += 1
        else:
            self._pending_tier = proposed
            self._pending_dwell = 1
        if self._pending_dwell >= self._dwell_target:
            self._current_tier = proposed
            self._pending_tier = None
            self._pending_dwell = 0
            return proposed
        return self._current_tier
