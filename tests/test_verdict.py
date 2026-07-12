from __future__ import annotations

from benchmarks.verdict import (
    bootstrap_tax_pct,
    cell_verdict,
    gate_report,
)


def _rep(x: float, n: int = 3) -> list[float]:
    return [x] * n


def test_survive_when_tax_ci_upper_below_5pct():
    v = cell_verdict(
        batch_size=256,
        ctx_tokens=1990,
        dsd_k0=_rep(2900.0),
        no_spec=_rep(2950.0),
    )
    assert v.verdict == "survive"
    assert v.tax_ci_high_pct < 5.0


def test_falsify_when_tax_ci_lower_above_10pct():
    v = cell_verdict(
        batch_size=256,
        ctx_tokens=1990,
        dsd_k0=_rep(2500.0),
        no_spec=_rep(3000.0),
    )
    assert v.verdict == "falsify"
    assert v.tax_ci_low_pct > 10.0


def test_ambiguous_when_tax_spans_5_to_10pct():
    v = cell_verdict(
        batch_size=256,
        ctx_tokens=1990,
        dsd_k0=[2800.0, 2820.0, 2790.0],
        no_spec=[2990.0, 3020.0, 2985.0],
    )
    assert 5.0 <= v.tax_mean_pct <= 10.0
    assert v.verdict == "ambiguous"


def test_gate_falsify_dominates_when_any_cell_falsifies():
    surv = cell_verdict(256, 460, _rep(2900.0), _rep(2950.0))
    fal = cell_verdict(256, 1990, _rep(2500.0), _rep(3000.0))
    report = gate_report([surv, fal])
    assert report.verdict == "falsify"


def test_gate_ambiguous_when_no_falsify_but_ambiguous_present():
    surv = cell_verdict(128, 460, _rep(2900.0), _rep(2950.0))
    amb = cell_verdict(
        192, 970, [2800.0, 2820.0, 2790.0], [2990.0, 3020.0, 2985.0]
    )
    report = gate_report([surv, amb])
    assert report.verdict == "ambiguous"


def test_gate_survive_when_all_cells_survive():
    cells = [
        cell_verdict(b, c, _rep(2900.0), _rep(2950.0))
        for b in (128, 192, 256)
        for c in (460, 970, 1990)
    ]
    report = gate_report(cells)
    assert report.verdict == "survive"
    assert len(report.cells) == 9


def test_bootstrap_rejects_nonpositive_no_spec():
    import pytest
    with pytest.raises(ValueError):
        bootstrap_tax_pct([2900.0], [0.0])


def test_bootstrap_rejects_empty_inputs():
    import pytest
    with pytest.raises(ValueError):
        bootstrap_tax_pct([], [3000.0])
