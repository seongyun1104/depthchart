from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AcceptanceTrace:
    """Rolling counters of drafted vs accepted speculative tokens, bucketed
    by draft step index (0..K-1).

    `attempts` = number of spec attempts recorded. Each attempt drafts up
    to K tokens; per-step counters track how many attempts reached step i
    (drafted_per_step[i]) and how many had step i accepted (accepted_per_step[i]).
    """
    k: int
    attempts: int = 0
    drafted_per_step: list[int] = None
    accepted_per_step: list[int] = None

    @classmethod
    def empty(cls, k: int) -> AcceptanceTrace:
        return cls(k=k, attempts=0, drafted_per_step=[0] * k, accepted_per_step=[0] * k)

    def record(self, drafted: list[int], accepted: list[int]) -> None:
        assert len(drafted) == self.k and len(accepted) == self.k
        self.attempts += 1
        for i in range(self.k):
            self.drafted_per_step[i] += drafted[i]
            self.accepted_per_step[i] += accepted[i]

    def rate_per_step(self) -> list[float]:
        return [
            (a / d) if d else 0.0
            for a, d in zip(self.accepted_per_step, self.drafted_per_step, strict=True)
        ]

    def mean_accepted_length(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(self.accepted_per_step) / self.attempts
