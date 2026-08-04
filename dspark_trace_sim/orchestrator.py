# Grid orchestration + reporting for Phase 2B replay.
#
# Sweeps the (confidence_ema_alpha × staleness × warm_up_prior) grid, running
# replay_cell per cell, and emits a provenance-carrying JSON report plus a
# markdown table for the follow-up PR comment.
#
# Convention (policy.py L11-15): the module runs on the internal EMA-retention
# alpha, but every grid axis and report label speaks the PR's convention,
# `confidence_ema_alpha = 1 - internal_alpha` (PR default 0.8 == internal 0.2).
# The internal alpha never leaves this module — leaking it into the report
# would invert the sign the PR reviewers read.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .policy import Policy
from .replay import CellResult, Sps, default_sps, replay_cell
from .trace_format import Provenance, SampleTrace

# Grid in PR convention. 1.0 = raw (no smoothing, paper §5.2), 0.8 = PR default,
# 0.5 = heavier smoothing. Chosen so both the raw and PR-default points are on
# the grid; the pre-registration's "{0, 0.5, 0.8}" mixed the two conventions
# (see project_dspark_phase1.md). Override before any public comment.
DEFAULT_CONFIDENCE_EMA_ALPHAS: tuple[float, ...] = (1.0, 0.8, 0.5)
DEFAULT_STALENESSES: tuple[int, ...] = (0, 1, 2)
DEFAULT_WARMUP_PRIORS: tuple[float, ...] = (0.5, 1.0)

# Pre-registered judgement thresholds (project_dspark_phase1.md §Phase 2B).
EQUIV_EPS = 0.02
MATERIAL_DELTA = 0.05
RANK_PRES_MIN = 0.95

# Reference cell for the raw-vs-default verdict: the paper/PR spec is a 2-step
# stale snapshot with the optimistic prior.
REF_STALENESS = 2
REF_WARMUP_PRIOR = 1.0
RAW_ALPHA = 1.0
PR_DEFAULT_ALPHA = 0.8

Verdict = Literal["ema_harmless", "raw_wins", "ema_wins", "inconclusive"]

_VERDICT_ACTION: dict[Verdict, str] = {
    "ema_harmless": (
        "EMA default (confidence_ema_alpha=0.8) is harmless: tax within "
        f"{EQUIV_EPS} of raw and ranking preserved > {RANK_PRES_MIN}. "
        "Data-point supporting the current default."
    ),
    "raw_wins": (
        "Raw 2-step-prior beats the EMA default by >= "
        f"{MATERIAL_DELTA} tax. Candidate for revisiting the "
        "confidence_ema_alpha default under the offending regime."
    ),
    "ema_wins": (
        "EMA default beats raw by >= "
        f"{MATERIAL_DELTA} tax. The smoothing improves on the paper's raw "
        "2-step-prior — reinforces the design choice."
    ),
    "inconclusive": (
        "Difference between raw and EMA default is below the material "
        f"threshold ({MATERIAL_DELTA}) but outside the equivalence band "
        f"({EQUIV_EPS}); no pre-registered branch fires."
    ),
}


def pr_alpha_to_internal(pr_alpha: float) -> float:
    """confidence_ema_alpha (new-observation weight) -> internal retention alpha."""
    if not 0.0 <= pr_alpha <= 1.0:
        raise ValueError(f"confidence_ema_alpha={pr_alpha} outside [0, 1].")
    return 1.0 - pr_alpha


def internal_alpha_to_pr(internal_alpha: float) -> float:
    if not 0.0 <= internal_alpha <= 1.0:
        raise ValueError(f"internal alpha={internal_alpha} outside [0, 1].")
    return 1.0 - internal_alpha


@dataclass(frozen=True)
class CellReport:
    confidence_ema_alpha: float
    staleness: int
    warm_up_prior: float
    internal_alpha: float
    wasted: float
    missed: float
    tax: float
    rank_preservation: float


@dataclass(frozen=True)
class GridReport:
    provenances: list[Provenance]
    batch_size: int
    horizon: int
    num_batches: int
    seed: int
    cells: list[CellReport]
    verdict: Verdict
    action: str

    def to_json(self) -> dict:
        return {
            "provenance": [p.model_dump() for p in self.provenances],
            "config": {
                "batch_size": self.batch_size,
                "horizon": self.horizon,
                "num_batches": self.num_batches,
                "seed": self.seed,
            },
            "cells": [asdict(c) for c in self.cells],
            "verdict": self.verdict,
            "action": self.action,
        }


def run_grid(
    provenances: list[Provenance],
    pool: list[SampleTrace],
    *,
    sps: Sps = default_sps,
    batch_size: int = 8,
    horizon: int = 8,
    num_batches: int = 64,
    seed: int = 980406,
    confidence_ema_alphas: tuple[float, ...] = DEFAULT_CONFIDENCE_EMA_ALPHAS,
    stalenesses: tuple[int, ...] = DEFAULT_STALENESSES,
    warmup_priors: tuple[float, ...] = DEFAULT_WARMUP_PRIORS,
) -> GridReport:
    """Run every grid cell and assemble the provenance-carrying report."""
    cells: list[CellReport] = []
    for pr_alpha in confidence_ema_alphas:
        internal_alpha = pr_alpha_to_internal(pr_alpha)
        for staleness in stalenesses:
            for warm_up_prior in warmup_priors:
                policy = Policy(
                    alpha=internal_alpha,
                    staleness=staleness,
                    warm_up_prior=warm_up_prior,
                )
                result = replay_cell(
                    pool,
                    policy,
                    sps=sps,
                    batch_size=batch_size,
                    horizon=horizon,
                    num_batches=num_batches,
                    seed=seed,
                )
                cells.append(_cell_report(pr_alpha, staleness, warm_up_prior, result))

    verdict = classify(cells)
    return GridReport(
        provenances=provenances,
        batch_size=batch_size,
        horizon=horizon,
        num_batches=num_batches,
        seed=seed,
        cells=cells,
        verdict=verdict,
        action=_VERDICT_ACTION[verdict],
    )


def _cell_report(
    pr_alpha: float, staleness: int, warm_up_prior: float, result: CellResult
) -> CellReport:
    return CellReport(
        confidence_ema_alpha=pr_alpha,
        staleness=staleness,
        warm_up_prior=warm_up_prior,
        internal_alpha=pr_alpha_to_internal(pr_alpha),
        wasted=result.mean_wasted,
        missed=result.mean_missed,
        tax=result.tax,
        rank_preservation=result.rank_preservation,
    )


def _find_cell(
    cells: list[CellReport],
    confidence_ema_alpha: float,
    staleness: int,
    warm_up_prior: float,
) -> CellReport | None:
    for c in cells:
        if (
            c.confidence_ema_alpha == confidence_ema_alpha
            and c.staleness == staleness
            and c.warm_up_prior == warm_up_prior
        ):
            return c
    return None


def classify(cells: list[CellReport]) -> Verdict:
    """Pre-registered 3-branch verdict comparing the raw and PR-default cells at
    the reference (2-step stale, optimistic prior) operating point."""
    raw = _find_cell(cells, RAW_ALPHA, REF_STALENESS, REF_WARMUP_PRIOR)
    default = _find_cell(cells, PR_DEFAULT_ALPHA, REF_STALENESS, REF_WARMUP_PRIOR)
    if raw is None or default is None:
        return "inconclusive"

    delta = default.tax - raw.tax  # > 0 means EMA default carries more tax
    ranked = min(raw.rank_preservation, default.rank_preservation)
    if abs(delta) < EQUIV_EPS and ranked > RANK_PRES_MIN:
        return "ema_harmless"
    if delta >= MATERIAL_DELTA:
        return "raw_wins"
    if delta <= -MATERIAL_DELTA:
        return "ema_wins"
    return "inconclusive"


def to_markdown_table(report: GridReport) -> str:
    """Copy-pasteable table for the PR comment. Columns in PR convention."""
    lines: list[str] = []
    for p in report.provenances:
        lines.append(
            f"<!-- trace: checkpoint={p.checkpoint_id}@{p.checkpoint_revision} "
            f"target={p.target_model} dataset={p.dataset} "
            f"deepspec={p.deepspec_commit} collected={p.collected_at} -->"
        )
    lines.append(
        f"<!-- replay: batch_size={report.batch_size} horizon={report.horizon} "
        f"num_batches={report.num_batches} seed={report.seed} -->"
    )
    lines.append(
        "| confidence_ema_alpha | staleness | warm_up_prior | wasted | missed "
        "| tax | rank_pres |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for c in sorted(
        report.cells,
        key=lambda c: (-c.confidence_ema_alpha, c.staleness, c.warm_up_prior),
    ):
        lines.append(
            f"| {c.confidence_ema_alpha:g} | {c.staleness} | {c.warm_up_prior:g} "
            f"| {c.wasted:.4f} | {c.missed:.4f} | {c.tax:.4f} "
            f"| {c.rank_preservation:.4f} |"
        )
    lines.append("")
    lines.append(f"**Verdict:** `{report.verdict}` — {report.action}")
    return "\n".join(lines)
