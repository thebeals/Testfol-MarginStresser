"""Regression tests for Fresh Start return windows."""

import pandas as pd
import pytest

from app.core import backtest_orchestrator as orchestrator
from app.ui import results as results_ui


def test_fresh_start_ignores_anchor_only_year(monkeypatch):
    calls = []

    def _run_single_backtest(**kwargs):
        calls.append((kwargs["name"], kwargs["start_date"]))
        start = pd.Timestamp(kwargs["start_date"])
        anchor = start - pd.offsets.BDay(1)
        end = pd.Timestamp(f"{start.year}-12-31")
        series = pd.Series([100000.0, 110000.0], index=[anchor, end])
        return {"series": series}

    monkeypatch.setattr(orchestrator, "run_single_backtest", _run_single_backtest)

    fresh, stitched = results_ui._compute_fresh_yearly_returns.__wrapped__(
        allocation={"AAA": 100.0},
        maint_pcts={"AAA": 25.0},
        rebalance={"mode": "Standard", "freq": "Yearly"},
        series_start="2004-01-01",
        series_end="2004-12-31",
        series_start_val=100000.0,
        year_list=(2003, 2004),
        cache_policy="anchor-year-regression-test",
    )

    assert calls == [("Fresh_2004", "2004-01-01")]
    assert set(fresh) == {2004}
    assert fresh[2004] == pytest.approx(0.10)
    assert stitched is not None
    assert stitched.index.min() == pd.Timestamp("2003-12-31")
