from __future__ import annotations

import itertools
import random

import pytest

from dspark_trace_sim.oracle import (
    _leading_ones,
    missed_acceptance_ratio,
    oracle_lengths,
    wasted_verification_ratio,
)


# ----- oracle_lengths -----

def test_oracle_length_equals_leading_ones():
    accepts = [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],  # trailing 1s after a 0 don't count
        [0, 1, 1, 1, 1],  # first-position 0 -> length 0
        [1, 1, 1, 1, 1],
    ]
    assert oracle_lengths(accepts) == [2, 1, 0, 5]


def test_oracle_length_empty_batch():
    assert oracle_lengths([]) == []


def test_leading_ones_helper():
    assert _leading_ones([1, 1, 1]) == 3
    assert _leading_ones([0, 1, 1]) == 0
    assert _leading_ones([1, 0, 1]) == 1
    assert _leading_ones([]) == 0


# ----- 2-component metrics -----

def test_oracle_choice_yields_zero_wasted_and_zero_missed():
    """Oracle picks length = leading-1 count. Both metrics = 0 by construction."""
    accepts = [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    oracle = oracle_lengths(accepts)

    assert wasted_verification_ratio(oracle, accepts) == 0.0
    assert missed_acceptance_ratio(oracle, accepts) == 0.0


def test_over_allocating_policy_has_wasted_but_no_missed():
    """Admitting past the leading-1 tail is pure waste."""
    accepts = [[1, 1, 0, 0, 0]]
    policy = [5]  # admitted all 5 positions; oracle would admit 2.

    assert wasted_verification_ratio(policy, accepts) == pytest.approx(3 / 5)
    assert missed_acceptance_ratio(policy, accepts) == 0.0


def test_under_allocating_policy_has_missed_but_no_wasted():
    """Admitting fewer than leading-1 misses accepts, wastes nothing."""
    accepts = [[1, 1, 1, 1, 0]]
    policy = [2]  # admitted 2; oracle would admit 4.

    assert wasted_verification_ratio(policy, accepts) == 0.0
    assert missed_acceptance_ratio(policy, accepts) == pytest.approx(2 / 4)


def test_mixed_policy_shows_both_costs_in_batch():
    """Batch aggregates over-allocation and under-allocation independently."""
    accepts = [
        [1, 1, 0, 0, 0],  # oracle=2, policy=4 -> wasted=2, missed=0
        [1, 1, 1, 1, 0],  # oracle=4, policy=2 -> wasted=0, missed=2
    ]
    policy = [4, 2]

    # Denominators: total_admitted = 4 + 2 = 6, total_oracle = 2 + 4 = 6.
    assert wasted_verification_ratio(policy, accepts) == pytest.approx(2 / 6)
    assert missed_acceptance_ratio(policy, accepts) == pytest.approx(2 / 6)


def test_zero_admission_defines_wasted_ratio_as_zero():
    """No admissions -> no waste (denominator is 0)."""
    accepts = [[1, 1, 0]]
    assert wasted_verification_ratio([0], accepts) == 0.0


def test_zero_oracle_defines_missed_ratio_as_zero():
    """Oracle would accept nothing -> no missed (denominator is 0)."""
    accepts = [[0, 0, 0]]
    assert missed_acceptance_ratio([0], accepts) == 0.0
    assert missed_acceptance_ratio([2], accepts) == 0.0


# ----- Property tests (user-mandated) -----

@pytest.mark.parametrize("seed", range(20))
def test_no_policy_beats_oracle_on_both_components(seed):
    """Invariant: any non-oracle policy has wasted > 0 OR missed > 0.

    Oracle sets both to 0 by construction; any deviation from oracle_lengths
    incurs at least one of the two costs. If this ever fails, the oracle
    implementation or the metric definition has drifted.
    """
    rng = random.Random(seed)
    num_requests = rng.randint(1, 6)
    gamma = rng.randint(1, 7)

    accepts = []
    for _ in range(num_requests):
        row = [rng.choice([0, 1]) for _ in range(gamma)]
        accepts.append(row)

    oracle = oracle_lengths(accepts)

    for candidate_lengths in itertools.product(range(gamma + 1), repeat=num_requests):
        policy = list(candidate_lengths)
        wasted = wasted_verification_ratio(policy, accepts)
        missed = missed_acceptance_ratio(policy, accepts)

        if policy == oracle:
            assert wasted == 0.0
            assert missed == 0.0
        else:
            # At least one cost must be > 0 for any deviation from oracle.
            # (Both may be 0 only when the whole batch has zero oracle_len
            #  and zero admissions, which is exactly the oracle case there.)
            has_oracle_accepts = any(_leading_ones(row) > 0 for row in accepts)
            has_any_admission = any(l > 0 for l in policy)
            deviates_by_over = any(
                l > _leading_ones(row) for l, row in zip(policy, accepts)
            )
            deviates_by_under = any(
                l < _leading_ones(row) for l, row in zip(policy, accepts)
            )

            if deviates_by_over:
                assert wasted > 0.0
            if deviates_by_under and has_oracle_accepts:
                assert missed > 0.0
            assert has_any_admission or has_oracle_accepts
