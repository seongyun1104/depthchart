# Policy configuration and signal transformation for Phase 2B replay.
#
# Defines the contract that `replay.py` (future) must satisfy: given a
# per-request confidence history and a policy config, produce the confidence
# signal the scheduler consumes for the current step.
#
# Knobs (pre-registered for Phase 2B grid):
# - alpha: EMA *history retention* in [0, 1]. alpha=0.0 -> raw (no smoothing,
#   signal = readable snapshot). alpha=1.0 -> frozen at warm_up_prior.
#     ema_new = alpha * ema_old + (1 - alpha) * observation
#   NOTE: this sign is OPPOSITE the PR code's `confidence_ema_alpha`
#   (neuralmagic/vllm@codex/dspark-capacity-realloc adaptive_verification.py
#    L263-268) where alpha weights the NEW observation. Translation:
#     PR_alpha = 1 - this_module_alpha
#   PR default confidence_ema_alpha=0.8 corresponds to alpha=0.2 here.
# - staleness: how many steps behind the current step the observation is
#   drawn from. Paper §5.2 says budget uses 2-step-prior; per-request uses 0.
# - warm_up_prior: signal value used before the first observation is
#   available (either before step 0 or when staleness exceeds the history).
#   Parameterised at reviewer's direction because "1.0" front-loads budget
#   optimistically; sensitivity {1.0, 0.5} enters the Phase 2B grid.
#
# Identity invariant (encoded in test_policy.py):
#     Policy(alpha=0.0, staleness=0) applied to any history returns the
#     current step's observation unchanged. Fixes the replay boundary
#     before `replay.py` lands so the scheduler pipeline can be validated
#     independently.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    alpha: float = 0.0
    staleness: int = 0
    warm_up_prior: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha={self.alpha} outside [0, 1].")
        if self.staleness < 0:
            raise ValueError(f"staleness={self.staleness} must be >= 0.")
        if not 0.0 <= self.warm_up_prior <= 1.0:
            raise ValueError(
                f"warm_up_prior={self.warm_up_prior} outside [0, 1]."
            )


def signal_from_history(
    history: Sequence[Sequence[Sequence[float]]],
    current_step: int,
    policy: Policy,
) -> list[list[float]]:
    """Compute the scheduler-facing signal at `current_step` under `policy`.

    Args:
        history: [request][step][position] post-sigmoid confidences observed
            so far. Each request may have a different step count; the value
            at index (current_step - staleness) is the "readable snapshot".
        current_step: 0-indexed current step. The observation at index
            (current_step - staleness) is the freshest observation the
            scheduler is allowed to see under the staleness constraint.
        policy: EMA alpha (history retention), staleness, warm-up prior.

    Returns:
        [request][position] confidences ready to feed algorithm_1 or
        admit_top_k.

    EMA recurrence (per request, per position, applied forward through steps
    up to `readable_idx = current_step - staleness`):

        ema_{-1} = warm_up_prior
        ema_t    = alpha * ema_{t-1} + (1 - alpha) * observation_t

    Then signal = ema_{readable_idx}. If readable_idx < 0 (history hasn't
    caught up with the staleness requirement), the signal is uniform
    warm_up_prior. If a request's history is shorter than readable_idx + 1,
    that request's remaining EMA steps freeze at the last available step.

    Identity: with policy=Policy(alpha=0.0, staleness=0), the returned
    signal equals `history[r][current_step]` for every request r that has
    an observation at current_step.
    """
    if current_step < 0:
        raise ValueError(f"current_step={current_step} must be >= 0.")

    readable_idx = current_step - policy.staleness

    signals: list[list[float]] = []
    for req_history in history:
        if not req_history:
            signals.append([])
            continue

        gamma = len(req_history[0])
        ema = [policy.warm_up_prior] * gamma

        if readable_idx < 0:
            signals.append(ema)
            continue

        last_step = min(readable_idx, len(req_history) - 1)
        for step_idx in range(last_step + 1):
            observation = req_history[step_idx]
            for pos in range(gamma):
                ema[pos] = (
                    policy.alpha * ema[pos]
                    + (1.0 - policy.alpha) * observation[pos]
                )
        signals.append(ema)
    return signals
