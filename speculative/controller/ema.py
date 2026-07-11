from __future__ import annotations

from collections.abc import Sequence

from .schema import EMAConfig


class EMAAccept:
    def __init__(self, config: EMAConfig):
        self.config = config
        self._ema: float | None = None
        self._batches_seen: int = 0
        self._batches_since_update: int = 0

    def observe(self, accepted_per_req: Sequence[int]) -> None:
        if not accepted_per_req:
            return
        self._batches_seen += 1
        self._batches_since_update += 1
        batch_mean = sum(accepted_per_req) / len(accepted_per_req)
        if self._ema is None:
            self._ema = batch_mean
            self._batches_since_update = 0
            return
        if self._batches_since_update >= self.config.update_interval:
            a = self.config.ema_alpha
            self._ema = a * batch_mean + (1 - a) * self._ema
            self._batches_since_update = 0

    @property
    def is_warm(self) -> bool:
        return self._batches_seen >= self.config.warmup_batches

    @property
    def value(self) -> float | None:
        return self._ema

    def adjust(self, k_lut: int, k_palette: Sequence[int]) -> int:
        if k_lut <= 0 or not self.is_warm or self._ema is None:
            return k_lut
        threshold = k_lut * (1.0 + self.config.down_hysteresis)
        if self._ema >= threshold:
            return k_lut
        smaller = sorted((k for k in k_palette if k < k_lut), reverse=True)
        return smaller[0] if smaller else 0
