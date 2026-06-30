from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AcceptanceTrace:
    """Per-request rolling counters of drafted vs accepted speculative tokens,
    bucketed by draft step index (0..K-1).
    """
    k: int
    drafted_per_step: list[int]
    accepted_per_step: list[int]

    @classmethod
    def empty(cls, k: int) -> AcceptanceTrace:
        return cls(k=k, drafted_per_step=[0] * k, accepted_per_step=[0] * k)

    def record(self, drafted: list[int], accepted: list[int]) -> None:
        assert len(drafted) == self.k and len(accepted) == self.k
        for i in range(self.k):
            self.drafted_per_step[i] += drafted[i]
            self.accepted_per_step[i] += accepted[i]

    def rate_per_step(self) -> list[float]:
        return [
            (a / d) if d else 0.0
            for a, d in zip(self.accepted_per_step, self.drafted_per_step, strict=True)
        ]

    def average_accepted_length(self) -> float:
        total_drafted = sum(self.drafted_per_step)
        total_accepted = sum(self.accepted_per_step)
        return (total_accepted / total_drafted * self.k) if total_drafted else 0.0
