# Trace-driven replay of scheduling policies (Phase 2B).
#
# Loads collected single-request traces, synthesises batches, and steps each
# batch forward. At every synthetic step the split from the PR is reproduced:
# - budget path: how many total draft slots to admit, computed from the
#   staleness-shifted, EMA-smoothed signal (policy.signal_from_history) fed to
#   scheduler.algorithm_1.
# - admission path: which slots, computed from the fresh current-step
#   confidence fed to scheduler.admit_top_k under that budget.
# Each step is scored against the post-hoc oracle (oracle.py) into the two
# orthogonal tax components. See README.md and dspark_pr47808_analysis.md.
#
# Scope: request-independent progression (a synthetic batch never feeds budget
# decisions back into which request advances). Valid for ranking policies on
# confidence signals, not for absolute serving behaviour — never report as
# throughput.

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .oracle import missed_acceptance_ratio, oracle_lengths, wasted_verification_ratio
from .policy import Policy, signal_from_history
from .scheduler import admit_top_k, algorithm_1
from .trace_format import Provenance, SampleTrace, StepRecord, read_trace

Sps = Callable[[int], float]


def power_law_sps(exponent: float = 0.5) -> Sps:
    """Synthetic step-curve family ``sps(b) = b**-exponent`` for the budget path.

    The real steps-per-second curve is a hardware property measured separately at
    collection time (see PHASE_2A_RUNBOOK.md) and, once measured, is the primary
    curve. This synthetic family exists to test how sensitive a verdict is to the
    curve shape: a verdict is only confirmed if it survives the pre-registered
    family (measured ± exponents {0.3, 0.5, 0.7}), otherwise it is reported as
    SPS-dependent.

    The exponent is sub-linear on purpose: a flat curve admits every slot and a
    1/b curve admits none, so algorithm_1's throughput-optimal K would not depend
    on the confidence signal in either extreme — both wash out the comparison.
    """

    def sps(batch_size: int) -> float:
        return 1.0 / (max(1, batch_size) ** exponent)

    return sps


default_sps: Sps = power_law_sps(0.5)


def load_pool(trace_dir: Path) -> tuple[list[Provenance], list[SampleTrace]]:
    """Load every ``*.jsonl`` trace under ``trace_dir`` into SampleTraces.

    Returns the de-duplicated provenance headers (passed through untouched to
    the final report) alongside the traces. Raises if the directory holds no
    trace files.
    """
    trace_dir = Path(trace_dir)
    paths = sorted(trace_dir.rglob("*.jsonl"))
    if not paths:
        raise ValueError(f"No trace files under {trace_dir}")

    provenances: list[Provenance] = []
    seen: set[str] = set()
    traces: list[SampleTrace] = []
    for path in paths:
        provenance, records = read_trace(path)
        key = provenance.to_jsonl_line()
        if key not in seen:
            seen.add(key)
            provenances.append(provenance)
        traces.append(_sample_trace_from_records(provenance, records))
    return provenances, traces


def _sample_trace_from_records(
    provenance: Provenance, records: Sequence[StepRecord]
) -> SampleTrace:
    sample_id = records[0].sample_id if records else ""
    return SampleTrace(sample_id=sample_id, dataset=provenance.dataset, steps=list(records))


@dataclass(frozen=True)
class BatchMember:
    """One request's horizon-length window, aligned so local step 0 is the
    moment it enters the synthetic batch."""

    sample_id: str
    confidences: list[list[float]]
    accepts: list[list[int]]


@dataclass(frozen=True)
class SynthesizedBatch:
    members: list[BatchMember]
    horizon: int


@dataclass(frozen=True)
class StepOutcome:
    policy_lengths: list[int]
    oracle_lengths: list[int]
    wasted: float
    missed: float


@dataclass(frozen=True)
class CellResult:
    mean_wasted: float
    mean_missed: float
    tax: float
    rank_preservation: float
    n_batches: int
    n_steps: int


def synthesize_batch(
    pool: Sequence[SampleTrace],
    batch_size: int,
    horizon: int,
    rng: random.Random,
) -> SynthesizedBatch:
    """Assemble a batch of ``batch_size`` requests, each a random ``horizon``
    window drawn from a random trace with enough steps.

    Deterministic in ``rng``: the same seeded Random yields the same batch.
    """
    if horizon < 1:
        raise ValueError(f"horizon={horizon} must be >= 1")
    if batch_size < 1:
        raise ValueError(f"batch_size={batch_size} must be >= 1")
    eligible = [t for t in pool if len(t.steps) >= horizon]
    if not eligible:
        raise ValueError(
            f"No trace in the pool has >= {horizon} steps (horizon too long)."
        )

    members: list[BatchMember] = []
    for _ in range(batch_size):
        trace = eligible[rng.randrange(len(eligible))]
        max_offset = len(trace.steps) - horizon
        offset = rng.randint(0, max_offset)
        window = trace.steps[offset : offset + horizon]
        members.append(
            BatchMember(
                sample_id=trace.sample_id,
                confidences=[list(s.confidences) for s in window],
                accepts=[list(s.accepts) for s in window],
            )
        )
    return SynthesizedBatch(members=members, horizon=horizon)


def replay_batch(
    batch: SynthesizedBatch,
    policy: Policy,
    sps: Sps = default_sps,
) -> list[StepOutcome]:
    """Step the batch forward, scoring each synthetic step against the oracle."""
    outcomes: list[StepOutcome] = []
    for t in range(batch.horizon):
        budget_signal: list[list[float]] = []
        fresh_signal: list[list[float]] = []
        accepts_now: list[list[int]] = []
        for member in batch.members:
            window = member.confidences[: t + 1]
            budget_signal.append(signal_from_history([window], t, policy)[0])
            fresh_signal.append(member.confidences[t])
            accepts_now.append(member.accepts[t])

        budget = sum(algorithm_1(budget_signal, sps)[0])
        policy_lengths = admit_top_k(fresh_signal, budget)

        outcomes.append(
            StepOutcome(
                policy_lengths=policy_lengths,
                oracle_lengths=oracle_lengths(accepts_now),
                wasted=wasted_verification_ratio(policy_lengths, accepts_now),
                missed=missed_acceptance_ratio(policy_lengths, accepts_now),
            )
        )
    return outcomes


def replay_cell(
    pool: Sequence[SampleTrace],
    policy: Policy,
    *,
    sps: Sps = default_sps,
    batch_size: int = 8,
    horizon: int = 8,
    num_batches: int = 64,
    seed: int = 980406,
) -> CellResult:
    """Aggregate one grid cell: mean tax components and rank preservation over
    ``num_batches`` synthesised batches × ``horizon`` steps."""
    rng = random.Random(seed)
    wasted_sum = 0.0
    missed_sum = 0.0
    rank_sum = 0.0
    n_steps = 0
    for _ in range(num_batches):
        batch = synthesize_batch(pool, batch_size, horizon, rng)
        for outcome in replay_batch(batch, policy, sps):
            wasted_sum += outcome.wasted
            missed_sum += outcome.missed
            rank_sum += _spearman(outcome.policy_lengths, outcome.oracle_lengths)
            n_steps += 1

    if n_steps == 0:
        return CellResult(0.0, 0.0, 0.0, 0.0, num_batches, 0)
    mean_wasted = wasted_sum / n_steps
    mean_missed = missed_sum / n_steps
    return CellResult(
        mean_wasted=mean_wasted,
        mean_missed=mean_missed,
        tax=mean_wasted + mean_missed,
        rank_preservation=rank_sum / n_steps,
        n_batches=num_batches,
        n_steps=n_steps,
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(a: Sequence[int], b: Sequence[int]) -> float:
    """Spearman rank correlation in [-1, 1].

    Zero-variance guard: identical vectors preserve ranking perfectly (1.0);
    a constant vector against a varying one has no monotone agreement (0.0).
    """
    if len(a) != len(b):
        raise ValueError("rank-correlation inputs must be equal length")
    n = len(a)
    if n < 2:
        return 1.0
    if list(a) == list(b):
        return 1.0
    ra = _average_ranks([float(x) for x in a])
    rb = _average_ranks([float(x) for x in b])
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0.0 or var_b == 0.0:
        return 0.0
    return cov / (var_a**0.5 * var_b**0.5)
