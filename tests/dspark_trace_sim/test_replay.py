from __future__ import annotations

import random

import pytest

from dspark_trace_sim.oracle import (
    missed_acceptance_ratio,
    oracle_lengths,
    wasted_verification_ratio,
)
from dspark_trace_sim.policy import Policy
from dspark_trace_sim.replay import (
    BatchMember,
    SynthesizedBatch,
    _spearman,
    default_sps,
    load_pool,
    replay_batch,
    replay_cell,
    synthesize_batch,
)
from dspark_trace_sim.scheduler import algorithm_1
from dspark_trace_sim.trace_format import (
    Provenance,
    SampleTrace,
    StepRecord,
    write_trace,
)


def _leading_ones(accepts: list[int]) -> int:
    n = 0
    for a in accepts:
        if a == 1:
            n += 1
        else:
            break
    return n


def _step(sample_id: str, step_idx: int, confidences: list[float], accepts: list[int]):
    return StepRecord(
        sample_id=sample_id,
        step_idx=step_idx,
        confidences=confidences,
        accepts=accepts,
        prefix_len=_leading_ones(accepts),
    )


def _trace(sample_id: str, n_steps: int, gamma: int = 3) -> SampleTrace:
    steps = []
    for i in range(n_steps):
        base = 0.9 - 0.05 * i
        confidences = [max(0.01, base - 0.1 * p) for p in range(gamma)]
        accepts = [1] * (i % (gamma + 1)) + [0] * (gamma - (i % (gamma + 1)))
        accepts = accepts[:gamma]
        steps.append(_step(sample_id, i, confidences, accepts))
    return SampleTrace(sample_id=sample_id, dataset="gsm8k", steps=steps)


def _pool(n: int = 6, n_steps: int = 10) -> list[SampleTrace]:
    return [_trace(f"gsm8k_{i:03d}", n_steps) for i in range(n)]


# ----- synthesize_batch -----

def test_synthesize_batch_is_deterministic_in_seed():
    pool = _pool()
    b1 = synthesize_batch(pool, batch_size=4, horizon=5, rng=random.Random(7))
    b2 = synthesize_batch(pool, batch_size=4, horizon=5, rng=random.Random(7))
    assert [m.sample_id for m in b1.members] == [m.sample_id for m in b2.members]
    assert [m.confidences for m in b1.members] == [m.confidences for m in b2.members]


def test_synthesize_batch_different_seed_diverges():
    pool = _pool(n=12, n_steps=12)
    b1 = synthesize_batch(pool, batch_size=6, horizon=4, rng=random.Random(1))
    b2 = synthesize_batch(pool, batch_size=6, horizon=4, rng=random.Random(2))
    assert [m.sample_id for m in b1.members] != [m.sample_id for m in b2.members]


def test_synthesize_batch_windows_have_horizon_length():
    pool = _pool()
    batch = synthesize_batch(pool, batch_size=3, horizon=5, rng=random.Random(0))
    assert batch.horizon == 5
    for m in batch.members:
        assert len(m.confidences) == 5
        assert len(m.accepts) == 5


def test_synthesize_batch_rejects_horizon_beyond_all_traces():
    pool = _pool(n=3, n_steps=4)
    with pytest.raises(ValueError, match="horizon"):
        synthesize_batch(pool, batch_size=2, horizon=99, rng=random.Random(0))


# ----- Identity cell: replay routes the raw current step (extends test_policy) -----

def test_identity_policy_reproduces_online_optimal_admission():
    """With Policy(alpha=0, staleness=0) the budget signal equals the fresh
    signal, so replay's admitted lengths must equal algorithm_1's own choice on
    the current-step confidence — the boundary that test_policy locks, now
    exercised through the full replay loop."""
    pool = _pool()
    batch = synthesize_batch(pool, batch_size=4, horizon=6, rng=random.Random(3))
    outcomes = replay_batch(batch, Policy(alpha=0.0, staleness=0), default_sps)

    for t, outcome in enumerate(outcomes):
        fresh = [m.confidences[t] for m in batch.members]
        expected, _theta = algorithm_1(fresh, default_sps)
        assert outcome.policy_lengths == expected


# ----- End-to-end accounting consistency -----

def test_replay_metrics_are_consistent_with_returned_lengths():
    members = [
        BatchMember(
            sample_id="a",
            confidences=[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],
            accepts=[[1, 1, 0], [1, 0, 0]],
        ),
        BatchMember(
            sample_id="b",
            confidences=[[0.5, 0.4, 0.3], [0.9, 0.9, 0.9]],
            accepts=[[0, 0, 0], [1, 1, 1]],
        ),
    ]
    batch = SynthesizedBatch(members=members, horizon=2)
    outcomes = replay_batch(batch, Policy(alpha=0.2, staleness=1), default_sps)

    for t, outcome in enumerate(outcomes):
        accepts_now = [m.accepts[t] for m in members]
        assert outcome.oracle_lengths == oracle_lengths(accepts_now)
        assert outcome.wasted == wasted_verification_ratio(
            outcome.policy_lengths, accepts_now
        )
        assert outcome.missed == missed_acceptance_ratio(
            outcome.policy_lengths, accepts_now
        )
        assert 0.0 <= outcome.wasted <= 1.0
        assert 0.0 <= outcome.missed <= 1.0


def test_replay_cell_returns_bounded_metrics():
    pool = _pool()
    result = replay_cell(
        pool,
        Policy(alpha=0.2, staleness=2, warm_up_prior=1.0),
        batch_size=4,
        horizon=5,
        num_batches=8,
        seed=11,
    )
    assert result.n_steps == 8 * 5
    assert 0.0 <= result.mean_wasted <= 1.0
    assert 0.0 <= result.mean_missed <= 1.0
    assert result.tax == pytest.approx(result.mean_wasted + result.mean_missed)
    assert -1.0 <= result.rank_preservation <= 1.0


def test_replay_cell_is_deterministic_in_seed():
    pool = _pool()
    policy = Policy(alpha=0.5, staleness=1)
    a = replay_cell(pool, policy, batch_size=4, horizon=5, num_batches=8, seed=99)
    b = replay_cell(pool, policy, batch_size=4, horizon=5, num_batches=8, seed=99)
    assert a == b


# ----- Rank correlation helper -----

def test_spearman_identical_is_one():
    assert _spearman([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_spearman_reversed_is_minus_one():
    assert _spearman([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)


def test_spearman_constant_against_varying_is_zero():
    assert _spearman([2, 2, 2, 2], [0, 1, 2, 3]) == 0.0


def test_spearman_singleton_is_one():
    assert _spearman([1], [0]) == 1.0


# ----- Round trip through disk -----

def test_load_pool_reads_written_traces(tmp_path):
    provenance = Provenance(
        deepspec_commit="005e03b8aaaa",
        checkpoint_id="deepseek-ai/dspark_gemma4_12b_block7",
        checkpoint_revision="abc1234",
        target_model="google/gemma-4-12B-it",
        dataset="gsm8k",
        collected_at="2026-08-04T00:00:00Z",
    )
    trace = _trace("gsm8k_000", n_steps=4)
    write_trace(tmp_path / "gsm8k_000.jsonl", provenance, trace.steps)

    provenances, traces = load_pool(tmp_path)

    assert len(provenances) == 1
    assert len(traces) == 1
    assert traces[0].sample_id == "gsm8k_000"
    assert len(traces[0].steps) == 4


def test_load_pool_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No trace files"):
        load_pool(tmp_path)
