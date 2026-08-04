from __future__ import annotations

import pytest

from dspark_trace_sim.scheduler import (
    _survival_candidates,
    admit_top_k,
    algorithm_1,
)


def _flat_sps(_: int) -> float:
    return 1000.0


def _linear_sps(b: int) -> float:
    return max(1.0, 1000.0 - 5.0 * b)


def _cliff_sps(threshold: int, high: float, low: float):
    def sps(b: int) -> float:
        return high if b <= threshold else low
    return sps


# ----- Survival candidate enumeration -----

def test_survival_candidates_orders_descending_by_survival():
    confidences = [[0.9, 0.5, 0.1], [0.8, 0.6, 0.2]]
    survivals = _survival_candidates(confidences)

    probs = [s[0] for s in survivals]
    assert probs == sorted(probs, reverse=True)


def test_survival_candidates_skips_zero_probability():
    confidences = [[0.0, 0.9, 0.5]]
    survivals = _survival_candidates(confidences)

    assert survivals == []


def test_survival_candidates_within_request_are_monotonic():
    """(r, j-1) survival >= (r, j) survival; must precede it in sorted order."""
    confidences = [[0.9, 0.8, 0.7]]
    survivals = _survival_candidates(confidences)
    positions = [j for _, _, j in survivals]

    assert positions == [0, 1, 2]


# ----- Algorithm 1: correctness -----

def test_algorithm_1_empty_batch():
    lengths, theta = algorithm_1([], _flat_sps)
    assert lengths == []
    assert theta == 0.0


def test_algorithm_1_zero_confidence_admits_nothing():
    lengths, theta = algorithm_1([[0.0, 0.0]], _flat_sps)
    assert lengths == [0]
    assert theta == pytest.approx(1.0 * 1000.0)


def test_algorithm_1_flat_sps_admits_everything():
    """With constant SPS, throughput grows monotonically with admissions."""
    confidences = [[0.9, 0.8, 0.7]]
    lengths, _theta = algorithm_1(confidences, _flat_sps, unconstrained=True)
    assert lengths == [3]


def test_algorithm_1_break_at_first_throughput_drop():
    """Early-stop should quit as soon as throughput stops improving."""
    # SPS drops sharply after batch size 2: R=1, bonus=1. Admitting any draft
    # crosses the cliff and reduces throughput.
    confidences = [[0.9, 0.9, 0.9]]
    lengths, _theta = algorithm_1(
        confidences, _cliff_sps(threshold=1, high=1000.0, low=1.0)
    )
    assert lengths == [0]


def test_algorithm_1_unconstrained_beats_early_stop_under_recovery():
    """If throughput dips then recovers, unconstrained catches the later peak."""
    # SPS: 100 at B=1, 1 at B=2, 100 at B>=3. Early-stop breaks after the dip;
    # unconstrained keeps scanning and finds the higher plateau.
    def bumpy_sps(b: int) -> float:
        if b == 1:
            return 100.0
        if b == 2:
            return 1.0
        return 100.0

    confidences = [[0.9, 0.9, 0.9]]

    stop_lengths, stop_theta = algorithm_1(confidences, bumpy_sps)
    global_lengths, global_theta = algorithm_1(
        confidences, bumpy_sps, unconstrained=True
    )

    # Early-stop bails at the first drop.
    assert stop_lengths == [0]
    # Unconstrained recovers.
    assert global_lengths[0] > 0
    assert global_theta >= stop_theta


def test_algorithm_1_matches_paper_bonus_baseline():
    """Baseline throughput = R * SPS(R) (paper line 6)."""
    lengths, theta = algorithm_1(
        [[0.01, 0.01], [0.01, 0.01]],
        _cliff_sps(threshold=2, high=500.0, low=0.001),
    )
    # Neither request has confidence high enough to overcome the SPS cliff.
    assert lengths == [0, 0]
    assert theta == pytest.approx(2 * 500.0)


# ----- admit_top_k: PR-equivalent admission -----

def test_admit_top_k_matches_algorithm_1_when_budget_is_optimal_size():
    """PR's split (budget determined separately, then greedy admit) equals
    algorithm_1's joint choice when the budget matches the optimal admission
    count. Locks the semantic parity with `_assign_draft_token_budget`.
    """
    confidences = [[0.9, 0.7, 0.5], [0.8, 0.4, 0.2]]
    joint_lengths, _theta = algorithm_1(confidences, _flat_sps, unconstrained=True)
    joint_admitted = sum(joint_lengths)

    split_lengths = admit_top_k(confidences, budget=joint_admitted)

    assert split_lengths == joint_lengths


def test_admit_top_k_zero_budget_admits_nothing():
    confidences = [[0.9, 0.7], [0.8, 0.4]]
    assert admit_top_k(confidences, budget=0) == [0, 0]


def test_admit_top_k_budget_larger_than_valid_slots():
    """Budget above the number of positive-survival slots caps at that count."""
    confidences = [[0.9, 0.0, 0.5], [0.8, 0.7, 0.0]]
    lengths = admit_top_k(confidences, budget=100)
    total_admitted = sum(lengths)
    # 2 positive slots in r=0 (only position 0 survives the zero), 2 in r=1.
    # Position after a zero has cum=0 and is skipped.
    assert total_admitted == 3


def test_admit_top_k_preserves_within_request_contiguity():
    """Once (r, j) is admitted, (r, 0..j-1) must already be admitted."""
    confidences = [[0.99, 0.98, 0.97, 0.96], [0.10, 0.05, 0.01, 0.005]]
    lengths = admit_top_k(confidences, budget=3)
    # The top-3 survivals should be r=0's first 3 positions (r=1 all < 0.10).
    assert lengths[0] == 3
    assert lengths[1] == 0


def test_admit_top_k_empty_batch():
    assert admit_top_k([], budget=5) == []
