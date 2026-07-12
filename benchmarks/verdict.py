from __future__ import annotations

import argparse
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GateVerdict = Literal["survive", "ambiguous", "falsify"]


@dataclass(frozen=True)
class CellVerdict:
    batch_size: int
    ctx_tokens: int
    tax_mean_pct: float
    tax_ci_low_pct: float
    tax_ci_high_pct: float
    verdict: GateVerdict


@dataclass(frozen=True)
class GateReport:
    cells: tuple[CellVerdict, ...]
    verdict: GateVerdict
    tax_survive_pct: float
    tax_falsify_pct: float
    n_bootstrap: int
    ci_level: float


def bootstrap_tax_pct(
    dsd_k0: Sequence[float],
    no_spec: Sequence[float],
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if not dsd_k0 or not no_spec:
        raise ValueError("both arms need at least one seed observation")
    rng = random.Random(seed)
    taxes: list[float] = []
    for _ in range(n_bootstrap):
        d = _mean(rng.choices(dsd_k0, k=len(dsd_k0)))
        n = _mean(rng.choices(no_spec, k=len(no_spec)))
        if n <= 0:
            continue
        taxes.append((n - d) / n * 100.0)
    if not taxes:
        raise ValueError("bootstrap produced no valid samples (no_spec throughput <= 0)")
    taxes.sort()
    alpha = (1.0 - ci_level) / 2.0
    lo_idx = max(0, int(alpha * len(taxes)))
    hi_idx = min(len(taxes) - 1, int((1.0 - alpha) * len(taxes)))
    mean_tax = sum(taxes) / len(taxes)
    return mean_tax, taxes[lo_idx], taxes[hi_idx]


def cell_verdict(
    batch_size: int,
    ctx_tokens: int,
    dsd_k0: Sequence[float],
    no_spec: Sequence[float],
    tax_survive_pct: float = 5.0,
    tax_falsify_pct: float = 10.0,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> CellVerdict:
    mean, lo, hi = bootstrap_tax_pct(
        dsd_k0, no_spec, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed,
    )
    if hi < tax_survive_pct:
        v: GateVerdict = "survive"
    elif lo > tax_falsify_pct:
        v = "falsify"
    else:
        v = "ambiguous"
    return CellVerdict(
        batch_size=batch_size,
        ctx_tokens=ctx_tokens,
        tax_mean_pct=mean,
        tax_ci_low_pct=lo,
        tax_ci_high_pct=hi,
        verdict=v,
    )


def gate_report(
    cells: Iterable[CellVerdict],
    tax_survive_pct: float = 5.0,
    tax_falsify_pct: float = 10.0,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
) -> GateReport:
    cells_tuple = tuple(cells)
    if not cells_tuple:
        raise ValueError("gate_report needs at least one cell")
    if any(c.verdict == "falsify" for c in cells_tuple):
        overall: GateVerdict = "falsify"
    elif any(c.verdict == "ambiguous" for c in cells_tuple):
        overall = "ambiguous"
    else:
        overall = "survive"
    return GateReport(
        cells=cells_tuple,
        verdict=overall,
        tax_survive_pct=tax_survive_pct,
        tax_falsify_pct=tax_falsify_pct,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _cells_from_parquet(
    parquet_path: Path,
    tax_survive_pct: float,
    tax_falsify_pct: float,
    n_bootstrap: int,
    ci_level: float,
) -> list[CellVerdict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "spec_arm" not in df.columns:
        raise ValueError(
            f"{parquet_path} has no spec_arm column — expected a V2 arm-comparison "
            f"sweep result. Are you sure this is v2_deployment_gate output?"
        )
    df = df[df["spec_arm"].isin(["dsd_k0", "no_spec"])]
    if df.empty:
        raise ValueError(
            f"{parquet_path} has no dsd_k0/no_spec rows — is spec_arm populated?"
        )

    cells: list[CellVerdict] = []
    for (b, ctx), sub in df.groupby(["batch_size", "ctx_tokens"]):
        dsd_per_seed = (
            sub[sub["spec_arm"] == "dsd_k0"]
            .groupby("seed_idx")["throughput_tok_s_counter"].first().tolist()
        )
        nsp_per_seed = (
            sub[sub["spec_arm"] == "no_spec"]
            .groupby("seed_idx")["throughput_tok_s_counter"].first().tolist()
        )
        if not dsd_per_seed or not nsp_per_seed:
            continue
        cells.append(cell_verdict(
            batch_size=int(b),
            ctx_tokens=int(ctx),
            dsd_k0=dsd_per_seed,
            no_spec=nsp_per_seed,
            tax_survive_pct=tax_survive_pct,
            tax_falsify_pct=tax_falsify_pct,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
        ))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 deployment gate verdict from a sweep parquet."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--tax-survive-pct", type=float, default=5.0)
    parser.add_argument("--tax-falsify-pct", type=float, default=10.0)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    cells = _cells_from_parquet(
        args.results,
        args.tax_survive_pct,
        args.tax_falsify_pct,
        args.n_bootstrap,
        args.ci_level,
    )
    if not cells:
        print(f"no eligible (batch, ctx) cells found in {args.results}")
        raise SystemExit(1)
    report = gate_report(
        cells,
        tax_survive_pct=args.tax_survive_pct,
        tax_falsify_pct=args.tax_falsify_pct,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
    )
    for c in report.cells:
        print(
            f"cell b={c.batch_size:>4} ctx={c.ctx_tokens:>5} "
            f"tax_mean={c.tax_mean_pct:6.2f}% "
            f"CI=[{c.tax_ci_low_pct:6.2f}, {c.tax_ci_high_pct:6.2f}] "
            f"→ {c.verdict}"
        )
    print(
        f"\nGate verdict: {report.verdict.upper()}   "
        f"(survive < {report.tax_survive_pct}%, falsify > {report.tax_falsify_pct}%, "
        f"{int(report.ci_level * 100)}% CI on {report.n_bootstrap} bootstrap iters)"
    )


if __name__ == "__main__":
    main()
