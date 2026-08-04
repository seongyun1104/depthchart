from __future__ import annotations

import math

import pytest

from dspark_trace_sim.policy import Policy, signal_from_history
from dspark_trace_sim.scheduler import algorithm_1


def _flat_sps(_: int) -> float:
    return 1000.0


# ----- Policy validation -----

def test_policy_defaults_are_raw_and_fresh():
    p = Policy()
    assert p.alpha == 0.0
    assert p.staleness == 0
    assert p.warm_up_prior == 1.0


def test_policy_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError, match="alpha"):
        Policy(alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        Policy(alpha=1.1)


def test_policy_rejects_negative_staleness():
    with pytest.raises(ValueError, match="staleness"):
        Policy(staleness=-1)


def test_policy_rejects_warm_up_prior_outside_unit_interval():
    with pytest.raises(ValueError, match="warm_up_prior"):
        Policy(warm_up_prior=-0.01)
    with pytest.raises(ValueError, match="warm_up_prior"):
        Policy(warm_up_prior=1.5)


# ----- User-mandated identity: staleness=0 + alpha=0 == fresh observation -----

def test_identity_alpha_zero_staleness_zero_returns_current_step():
    """Boundary contract for replay: no smoothing + no stale-shift = raw."""
    history = [
        [[0.9, 0.6, 0.3], [0.85, 0.55, 0.25]],  # request 0: two steps
        [[0.8, 0.4, 0.1], [0.7, 0.35, 0.05]],   # request 1: two steps
    ]
    policy = Policy(alpha=0.0, staleness=0)

    signal = signal_from_history(history, current_step=1, policy=policy)

    assert signal == [[0.85, 0.55, 0.25], [0.7, 0.35, 0.05]]


def test_identity_holds_across_scheduler_pipeline():
    """The scheduler's decision must be identical whether it's fed the raw
    current-step observation directly or the same observation routed through
    signal_from_history with the identity policy.
    """
    history = [[[0.9, 0.5, 0.2]], [[0.7, 0.4, 0.1]]]
    current = [row[0] for row in history]
    policy = Policy(alpha=0.0, staleness=0)

    routed = signal_from_history(history, current_step=0, policy=policy)
    lengths_direct, theta_direct = algorithm_1(current, _flat_sps, unconstrained=True)
    lengths_routed, theta_routed = algorithm_1(routed, _flat_sps, unconstrained=True)

    assert lengths_direct == lengths_routed
    assert theta_direct == theta_routed


# ----- Staleness -----

def test_staleness_shifts_readable_snapshot_back():
    history = [[[0.9, 0.6], [0.5, 0.3], [0.1, 0.05]]]
    policy = Policy(alpha=0.0, staleness=2)

    signal = signal_from_history(history, current_step=2, policy=policy)

    # readable_idx = 2 - 2 = 0, raw signal = history[0][0]
    assert signal == [[0.9, 0.6]]


def test_staleness_beyond_history_returns_warm_up_prior():
    history = [[[0.9, 0.6]]]
    policy = Policy(alpha=0.0, staleness=5, warm_up_prior=0.5)

    signal = signal_from_history(history, current_step=1, policy=policy)

    assert signal == [[0.5, 0.5]]


def test_warm_up_prior_parameterizable():
    """warm_up_prior is a knob (per Phase 2B pre-registration, sensitivity
    {0.5, 1.0} enters the grid); the module accepts either without change.
    """
    history = [[[0.9]]]

    for prior in (0.0, 0.5, 1.0):
        policy = Policy(alpha=0.0, staleness=2, warm_up_prior=prior)
        signal = signal_from_history(history, current_step=1, policy=policy)
        assert signal == [[prior]]


# ----- EMA (alpha > 0) -----

def test_alpha_one_freezes_signal_at_warm_up_prior():
    """alpha=1 -> ema_new = 1 * ema_old + 0 * observation; never updates."""
    history = [[[0.9, 0.6], [0.7, 0.5], [0.5, 0.3]]]
    policy = Policy(alpha=1.0, staleness=0, warm_up_prior=0.42)

    signal = signal_from_history(history, current_step=2, policy=policy)

    assert signal == [[0.42, 0.42]]


def test_alpha_half_averages_history_toward_recent():
    """alpha=0.5 -> equal weight on old EMA and new observation each step."""
    history = [[[1.0], [0.0]]]
    policy = Policy(alpha=0.5, staleness=0, warm_up_prior=0.0)

    # step 0: ema = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    # step 1: ema = 0.5 * 0.5 + 0.5 * 0.0 = 0.25
    signal = signal_from_history(history, current_step=1, policy=policy)

    assert signal[0][0] == pytest.approx(0.25)


def test_ema_matches_documented_recurrence():
    """Full recurrence check across multiple steps and positions."""
    history = [[[0.8, 0.4], [0.6, 0.2], [0.4, 0.1]]]
    policy = Policy(alpha=0.3, staleness=0, warm_up_prior=1.0)

    expected = [1.0, 1.0]
    for step in range(3):
        obs = history[0][step]
        for pos in range(2):
            expected[pos] = 0.3 * expected[pos] + 0.7 * obs[pos]

    signal = signal_from_history(history, current_step=2, policy=policy)

    for got, want in zip(signal[0], expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)
