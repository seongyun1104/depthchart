from __future__ import annotations

import random

import pytest

from dspark_trace_sim.orchestrator import (
    DEFAULT_CONFIDENCE_EMA_ALPHAS,
    DEFAULT_STALENESSES,
    DEFAULT_WARMUP_PRIORS,
    PR_DEFAULT_ALPHA,
    RAW_ALPHA,
    REF_STALENESS,
    REF_WARMUP_PRIOR,
    CellReport,
    classify,
    internal_alpha_to_pr,
    pr_alpha_to_internal,
    run_grid,
    to_markdown_table,
)
from dspark_trace_sim.replay import power_law_sps
from dspark_trace_sim.trace_format import Provenance, SampleTrace, StepRecord


def _leading_ones(accepts: list[int]) -> int:
    n = 0
    for a in accepts:
        if a == 1:
            n += 1
        else:
            break
    return n


def _trace(sample_id: str, n_steps: int = 8, gamma: int = 3) -> SampleTrace:
    steps = []
    for i in range(n_steps):
        confidences = [max(0.01, 0.9 - 0.05 * i - 0.1 * p) for p in range(gamma)]
        accepts = ([1] * (i % (gamma + 1)) + [0] * gamma)[:gamma]
        steps.append(
            StepRecord(
                sample_id=sample_id,
                step_idx=i,
                confidences=confidences,
                accepts=accepts,
                prefix_len=_leading_ones(accepts),
            )
        )
    return SampleTrace(sample_id=sample_id, dataset="gsm8k", steps=steps)


def _provenance() -> Provenance:
    return Provenance(
        deepspec_commit="005e03b8aaaa",
        checkpoint_id="deepseek-ai/dspark_gemma4_12b_block7",
        checkpoint_revision="abc1234",
        target_model="google/gemma-4-12B-it",
        dataset="gsm8k",
        collected_at="2026-08-04T00:00:00Z",
    )


def _ref_cell(pr_alpha: float, tax: float, rank_pres: float = 1.0) -> CellReport:
    return CellReport(
        confidence_ema_alpha=pr_alpha,
        staleness=REF_STALENESS,
        warm_up_prior=REF_WARMUP_PRIOR,
        internal_alpha=pr_alpha_to_internal(pr_alpha),
        wasted=tax / 2,
        missed=tax / 2,
        tax=tax,
        rank_preservation=rank_pres,
    )


# ----- alpha convention translation -----

def test_pr_default_maps_to_internal_point_two():
    assert pr_alpha_to_internal(0.8) == pytest.approx(0.2)


def test_alpha_translation_round_trips():
    for pr in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert internal_alpha_to_pr(pr_alpha_to_internal(pr)) == pytest.approx(pr)


def test_pr_alpha_out_of_range_rejected():
    with pytest.raises(ValueError, match="confidence_ema_alpha"):
        pr_alpha_to_internal(1.5)


# ----- Grid completeness and no internal-alpha leak into labels -----

def test_run_grid_covers_full_cartesian_product():
    provenances = [_provenance()]
    pool = [_trace(f"gsm8k_{i:03d}") for i in range(4)]
    report = run_grid(
        provenances, pool, batch_size=3, horizon=4, num_batches=4, seed=5
    )
    expected = (
        len(DEFAULT_CONFIDENCE_EMA_ALPHAS)
        * len(DEFAULT_STALENESSES)
        * len(DEFAULT_WARMUP_PRIORS)
    )
    assert len(report.cells) == expected


def test_report_labels_speak_pr_convention_with_internal_alpha_recorded():
    provenances = [_provenance()]
    pool = [_trace(f"gsm8k_{i:03d}") for i in range(4)]
    report = run_grid(
        provenances, pool, batch_size=3, horizon=4, num_batches=4, seed=5
    )
    for c in report.cells:
        assert c.confidence_ema_alpha in DEFAULT_CONFIDENCE_EMA_ALPHAS
        assert c.internal_alpha == pytest.approx(1.0 - c.confidence_ema_alpha)


def test_default_grid_includes_raw_and_pr_default():
    assert RAW_ALPHA in DEFAULT_CONFIDENCE_EMA_ALPHAS
    assert PR_DEFAULT_ALPHA in DEFAULT_CONFIDENCE_EMA_ALPHAS


# ----- 3-branch classifier -----

def test_classify_ema_harmless_when_tax_within_equivalence_band():
    cells = [
        _ref_cell(RAW_ALPHA, tax=0.30, rank_pres=0.99),
        _ref_cell(PR_DEFAULT_ALPHA, tax=0.31, rank_pres=0.99),
    ]
    assert classify(cells) == "ema_harmless"


def test_classify_raw_wins_when_ema_costs_materially_more():
    cells = [
        _ref_cell(RAW_ALPHA, tax=0.20),
        _ref_cell(PR_DEFAULT_ALPHA, tax=0.30),
    ]
    assert classify(cells) == "raw_wins"


def test_classify_ema_wins_when_ema_costs_materially_less():
    cells = [
        _ref_cell(RAW_ALPHA, tax=0.30),
        _ref_cell(PR_DEFAULT_ALPHA, tax=0.20),
    ]
    assert classify(cells) == "ema_wins"


def test_classify_inconclusive_between_bands():
    cells = [
        _ref_cell(RAW_ALPHA, tax=0.30),
        _ref_cell(PR_DEFAULT_ALPHA, tax=0.33),
    ]
    assert classify(cells) == "inconclusive"


def test_classify_not_harmless_when_ranking_degrades():
    cells = [
        _ref_cell(RAW_ALPHA, tax=0.30, rank_pres=0.99),
        _ref_cell(PR_DEFAULT_ALPHA, tax=0.31, rank_pres=0.80),
    ]
    assert classify(cells) == "inconclusive"


def test_classify_missing_reference_cell_is_inconclusive():
    assert classify([]) == "inconclusive"


# ----- Reporting -----

def test_markdown_table_carries_provenance_and_pr_columns():
    provenances = [_provenance()]
    pool = [_trace(f"gsm8k_{i:03d}") for i in range(4)]
    report = run_grid(
        provenances, pool, batch_size=3, horizon=4, num_batches=4, seed=5
    )
    md = to_markdown_table(report)

    assert "confidence_ema_alpha" in md
    assert "internal_alpha" not in md
    assert "deepseek-ai/dspark_gemma4_12b_block7" in md
    assert md.count("\n|") >= len(report.cells)
    assert report.verdict in md


def test_report_json_embeds_provenance_and_cells():
    provenances = [_provenance()]
    pool = [_trace(f"gsm8k_{i:03d}") for i in range(4)]
    report = run_grid(
        provenances, pool, batch_size=3, horizon=4, num_batches=4, seed=5
    )
    payload = report.to_json()

    assert payload["provenance"][0]["checkpoint_id"] == (
        "deepseek-ai/dspark_gemma4_12b_block7"
    )
    assert len(payload["cells"]) == len(report.cells)
    assert payload["config"]["seed"] == 5
    assert payload["verdict"] == report.verdict


# ----- Non-degeneracy guard + SPS sensitivity -----

def _varied_pool(n: int, seed: int, n_steps: int = 12, gamma: int = 4) -> list[SampleTrace]:
    rng = random.Random(seed)
    pool = []
    for i in range(n):
        steps = []
        for step_idx in range(n_steps):
            confidences = [max(0.02, rng.random()) for _ in range(gamma)]
            k = rng.randint(0, gamma)
            accepts = ([1] * k + [0] * gamma)[:gamma]
            steps.append(
                StepRecord(
                    sample_id=f"gsm8k_{i:03d}",
                    step_idx=step_idx,
                    confidences=confidences,
                    accepts=accepts,
                    prefix_len=_leading_ones(accepts),
                )
            )
        pool.append(SampleTrace(sample_id=f"gsm8k_{i:03d}", dataset="gsm8k", steps=steps))
    return pool


def test_grid_is_non_degenerate():
    """Regression guard for the SPS budget path: a curve that admits nothing (or
    everything) collapses every cell to identical tax. A healthy grid keeps a
    meaningful spread of distinct cells."""
    report = run_grid(
        [_provenance()],
        _varied_pool(16, seed=3),
        batch_size=8,
        horizon=8,
        num_batches=24,
        seed=5,
    )
    distinct = {round(c.tax, 6) for c in report.cells}
    assert len(distinct) > len(report.cells) // 2


def test_grid_runs_across_sps_family():
    """The pre-registered SPS sensitivity family {0.3, 0.5, 0.7} plugs into the
    same knob and produces a verdict per curve."""
    pool = _varied_pool(16, seed=3)
    verdicts = {
        exp: run_grid(
            [_provenance()],
            pool,
            sps=power_law_sps(exp),
            batch_size=8,
            horizon=8,
            num_batches=16,
            seed=5,
        ).verdict
        for exp in (0.3, 0.5, 0.7)
    }
    assert set(verdicts) == {0.3, 0.5, 0.7}
