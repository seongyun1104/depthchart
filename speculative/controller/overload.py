from __future__ import annotations

from .schema import OverloadConfig


class OverloadDetector:
    def __init__(self, config: OverloadConfig):
        self.config = config
        self._preempt_streak: int = 0
        self._last_kv: float = 0.0
        self._last_queue: int = 0

    def observe(self, kv_usage_pct: float, preempts: int, queue_depth: int) -> None:
        self._preempt_streak = self._preempt_streak + 1 if preempts > 0 else 0
        self._last_kv = kv_usage_pct
        self._last_queue = queue_depth

    @property
    def overloaded(self) -> bool:
        if self._last_kv > self.config.kv_usage_pct:
            return True
        if self._preempt_streak >= self.config.preempt_sustained:
            return True
        if (
            self.config.queue_depth is not None
            and self._last_queue > self.config.queue_depth
        ):
            return True
        return False
