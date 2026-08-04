# Post-hoc oracle and 2-component metrics for scheduler policies.
#
# Oracle knows realized accept/reject and picks per-request length = leading-1
# count of `accepts`. It's the throughput-maximal choice given full future
# information (any extra length = wasted verification; any less = missed
# acceptance).
#
# 2-component metrics decompose "policy tax" into two orthogonal costs:
# - wasted_verification_ratio: over-allocation (admitted slot that gets
#   rejected). Detects EMA-smoothed / stale-optimistic policies.
# - missed_acceptance_ratio: under-allocation (leading-1 slot the policy did
#   not admit). Detects raw-signal policies that respond too conservatively.
# Oracle is 0 on both by construction.
#
# Sum threshold (wasted_ratio + missed_ratio) is the pre-registered judgement
# axis for the (α × staleness) grid in Phase 2B; see README.md for the 3-branch
# result → action mapping.

from __future__ import annotations

from collections.abc import Sequence


def _leading_ones(accepts_r: Sequence[int]) -> int:
    n = 0
    for a in accepts_r:
        if a == 1:
            n += 1
        else:
            break
    return n


def oracle_lengths(accepts: Sequence[Sequence[int]]) -> list[int]:
    """Per-request oracle length = leading-1 count of the accept mask."""
    return [_leading_ones(row) for row in accepts]


def wasted_verification_ratio(
    policy_lengths: Sequence[int],
    accepts: Sequence[Sequence[int]],
) -> float:
    """Fraction of policy-admitted slots that would be rejected by the target.

    Returns 0.0 if the policy admitted nothing.
    """
    total_admitted = 0
    total_wasted = 0
    for ell, row in zip(policy_lengths, accepts, strict=True):
        oracle_len = _leading_ones(row)
        total_admitted += ell
        total_wasted += max(0, ell - oracle_len)
    if total_admitted == 0:
        return 0.0
    return total_wasted / total_admitted


def missed_acceptance_ratio(
    policy_lengths: Sequence[int],
    accepts: Sequence[Sequence[int]],
) -> float:
    """Fraction of oracle-accepted slots the policy failed to admit.

    Returns 0.0 if the oracle would accept nothing (denominator zero).
    """
    total_oracle_accepted = 0
    total_missed = 0
    for ell, row in zip(policy_lengths, accepts, strict=True):
        oracle_len = _leading_ones(row)
        total_oracle_accepted += oracle_len
        total_missed += max(0, oracle_len - ell)
    if total_oracle_accepted == 0:
        return 0.0
    return total_missed / total_oracle_accepted
