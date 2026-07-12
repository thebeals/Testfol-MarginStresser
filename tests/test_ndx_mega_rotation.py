import pandas as pd
import pytest

from app.core.shadow_backtest import run_shadow_backtest
from app.services.ndx_mega_rotation import (
    _ndx_top_rows_for_range,
    _top_unique_company_tickers,
    build_dynamic_allocation_plan,
    expand_dynamic_ticker_universe,
    is_dynamic_rotation_ticker,
    is_dynamic_ndxmega_ticker,
)


def test_ndxmega_top8_annual_plan_uses_actual_annual_components():
    plan = build_dynamic_allocation_plan(
        allocation={
            "NDXMEGASIM?L=2": 30.0,
            "NDXMEGA_TOP8_ANN?L=2": 30.0,
            "GLDSIM": 20.0,
            "VXUSSIM": 15.0,
            "QQQSIM?L=3": 5.0,
        },
        maint_pcts={
            "NDXMEGASIM": 50.0,
            "NDXMEGA_TOP8_ANN": 50.0,
            "GLDSIM": 25.0,
            "VXUSSIM": 25.0,
            "QQQSIM": 75.0,
        },
        pm_maint_pcts={"NDXMEGA_TOP8_ANN": 30.0},
        start_date="2024-01-01",
        end_date="2025-12-31",
    )

    assert plan.has_dynamic
    assert is_dynamic_ndxmega_ticker("NDXMEGA_TOP8_ANN?L=2")

    schedule_2024 = plan.dynamic_schedule[pd.Timestamp("2024-12-23")]
    dynamic_names = {
        ticker
        for ticker in schedule_2024
        if ticker.endswith("?L=2") and not ticker.startswith("NDXMEGASIM")
    }

    assert dynamic_names == {
        "AAPL?L=2",
        "AMZN?L=2",
        "AVGO?L=2",
        "GOOGL?L=2",
        "META?L=2",
        "MSFT?L=2",
        "NVDA?L=2",
        "TSLA?L=2",
    }
    assert sum(schedule_2024[ticker] for ticker in dynamic_names) == pytest.approx(30.0)
    assert schedule_2024["AAPL?L=2"] == pytest.approx(3.75)
    assert plan.maint_pcts["AAPL"] == 50.0
    assert plan.pm_maint_pcts["AAPL"] == 30.0


def test_expand_dynamic_ticker_universe_replaces_pseudo_ticker():
    expanded = expand_dynamic_ticker_universe(
        ["NDXMEGA_TOP8_ANN?L=2", "GLDSIM"],
        "2024-01-01",
        "2024-12-31",
    )

    assert "NDXMEGA_TOP8_ANN?L=2" not in expanded
    assert "GLDSIM" in expanded
    assert "AAPL?L=2" in expanded
    assert "MSFT?L=2" in expanded


def test_ndx_top8_annual_plan_uses_ndx_weights_and_auto_expenses():
    plan = build_dynamic_allocation_plan(
        allocation={
            "NDXMEGASIM?L=2&E=0.95": 30.0,
            "GLDSIM?E=0.40": 20.0,
            "VXUSSIM?E=0.05": 15.0,
            "QQQSIM?L=3&E=0.82": 5.0,
            "NDX_TOP8_ANN?L=2&E=AUTO": 30.0,
        },
        maint_pcts={
            "NDXMEGASIM": 50.0,
            "NDX_TOP8_ANN": 50.0,
            "GLDSIM": 25.0,
            "VXUSSIM": 25.0,
            "QQQSIM": 75.0,
        },
        pm_maint_pcts={"NDX_TOP8_ANN": 30.0},
        start_date="2024-01-01",
        end_date="2025-12-31",
    )

    assert plan.has_dynamic
    assert is_dynamic_rotation_ticker("NDX_TOP8_ANN?L=2&E=AUTO")

    schedule_2024 = plan.dynamic_schedule[pd.Timestamp("2024-01-01")]
    dynamic_names = {
        ticker
        for ticker in schedule_2024
        if ticker.endswith(("E=0.96", "E=0.98", "E=0.99", "E=1.00", "E=0.83", "E=0.92", "E=1.02"))
        and not ticker.startswith(("NDXMEGASIM", "GLDSIM", "VXUSSIM", "QQQSIM"))
    }

    assert dynamic_names == {
        "AAPL?L=2&E=0.96",
        "MSFT?L=2&E=0.98",
        "GOOG?L=2&E=0.96",
        "AMZN?L=2&E=0.99",
        "TSLA?L=2&E=0.83",
        "NVDA?L=2&E=0.92",
        "META?L=2&E=1.02",
        "AVGO?L=2&E=1.00",
    }
    assert sum(schedule_2024[ticker] for ticker in dynamic_names) == pytest.approx(30.0)
    assert plan.maint_pcts["AAPL"] == 50.0
    assert plan.pm_maint_pcts["AAPL"] == 30.0


def test_ndx_top8_annual_filters_to_official_membership_when_available():
    rows = _ndx_top_rows_for_range("2014-01-01", "2014-12-31")
    selected = _top_unique_company_tickers(rows.iloc[0], 8)

    assert rows.iloc[0]["SelectionDate"] == pd.Timestamp("2013-12-23")
    assert selected == [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "INTC",
        "QCOM",
        "CSCO",
        "CMCSA",
    ]
    assert "ORCL" not in selected


def test_ndx_top8_annual_preserves_historical_symbol_aliases():
    rows = _ndx_top_rows_for_range("2007-01-01", "2010-12-31")
    selections = {
        pd.Timestamp(row["Date"]).year: _top_unique_company_tickers(row, 8)
        for _, row in rows.iterrows()
    }

    assert selections[2007][-1] == "GEN"
    assert "DELL" not in selections[2007]
    assert selections[2008][-2] == "BBRY"
    assert "GILD" not in selections[2008]
    assert selections[2010][-1] == "BBRY"
    assert "AMZN" not in selections[2010]


def test_ndx_top2_annual_supports_rank_and_cap_weighting():
    rank_plan = build_dynamic_allocation_plan(
        allocation={
            "NDX_TOP2_ANN?L=2&E=AUTO&W=INV_RANK": 40.0,
            "GLDSIM?E=0.40": 40.0,
            "CASHX": 20.0,
        },
        maint_pcts={"NDX_TOP2_ANN": 50.0},
        pm_maint_pcts={"NDX_TOP2_ANN": 30.0},
        start_date="2026-01-01",
        end_date="2026-05-01",
    )

    schedule_2026 = rank_plan.dynamic_schedule[pd.Timestamp("2026-01-01")]

    assert schedule_2026["NVDA?L=2&E=0.92"] == pytest.approx(40.0 * 2.0 / 3.0)
    assert schedule_2026["AAPL?L=2&E=0.96"] == pytest.approx(40.0 * 1.0 / 3.0)
    assert schedule_2026["GLDSIM?E=0.40"] == pytest.approx(40.0)
    assert schedule_2026["CASHX"] == pytest.approx(20.0)
    assert all("W=" not in ticker for ticker in rank_plan.universe_tickers)
    assert rank_plan.maint_pcts["NVDA"] == 50.0
    assert rank_plan.pm_maint_pcts["NVDA"] == 30.0

    cap_plan = build_dynamic_allocation_plan(
        allocation={"NDX_TOP2_ANN?L=2&E=AUTO&W=CAP": 40.0},
        maint_pcts={"NDX_TOP2_ANN": 50.0},
        pm_maint_pcts={"NDX_TOP2_ANN": 30.0},
        start_date="2026-01-01",
        end_date="2026-05-01",
    )

    cap_schedule_2026 = cap_plan.dynamic_schedule[pd.Timestamp("2026-01-01")]
    assert cap_schedule_2026["NVDA?L=2&E=0.92"] == pytest.approx(20.968703921162856)
    assert cap_schedule_2026["AAPL?L=2&E=0.96"] == pytest.approx(19.031296078837144)


def test_shadow_backtest_dynamic_schedule_ignores_inactive_missing_prices():
    dates = pd.bdate_range("2024-01-02", periods=7)
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106],
            "BBB": [100, 100, 100, 100, 100, 100, 100],
            "CCC": [None, None, 100, 101, 102, 103, 104],
        },
        index=dates,
    )
    schedule = {
        pd.Timestamp("2024-01-02"): {"AAA": 50.0, "BBB": 50.0},
        pd.Timestamp("2024-01-05"): {"BBB": 50.0, "CCC": 50.0},
    }

    trades_df, _, composition_df, _, logs, port_series, _, *_ = run_shadow_backtest(
        allocation={"AAA": 0.0, "BBB": 0.0, "CCC": 0.0},
        start_val=10000.0,
        start_date="2024-01-02",
        end_date="2024-01-10",
        prices_df=prices,
        rebalance_freq="None",
        dynamic_allocation_schedule=schedule,
    )

    assert not port_series.empty, logs
    assert port_series.index.min() == pd.Timestamp("2024-01-02")
    assert any("[DYNAMIC ALLOCATION]" in line for line in logs)
    assert "CCC" in set(composition_df["Ticker"])
    assert not trades_df.empty
    assert "CCC" in set(trades_df["Ticker"])


def test_shadow_backtest_dynamic_schedule_does_not_backfill_first_future_basket():
    dates = pd.bdate_range("2023-12-27", "2024-01-05")
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 100, 100, 100, 100, 100, 100, 100],
        },
        index=dates,
    )
    schedule = {
        pd.Timestamp("2024-01-01"): {"AAA": 100.0},
    }

    _, _, _, _, logs, port_series, _, *_ = run_shadow_backtest(
        allocation={"AAA": 0.0, "BBB": 0.0},
        start_val=10000.0,
        start_date="2023-12-27",
        end_date="2024-01-05",
        prices_df=prices,
        rebalance_freq="None",
        dynamic_allocation_schedule=schedule,
    )

    assert not port_series.empty, logs
    assert port_series.index.min() == pd.Timestamp("2024-01-01")
    assert any("First valid data found at: 2024-01-02" in line for line in logs)


def test_dynamic_schedule_uses_requested_start_not_prior_price_anchor():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-05"])
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 110.0, 110.0],
            "BBB": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    schedule = {
        pd.Timestamp("2024-01-01"): {"AAA": 100.0},
        pd.Timestamp("2024-01-05"): {"BBB": 100.0},
    }

    *_, port_series, _, _ = run_shadow_backtest(
        allocation={"AAA": 0.0, "BBB": 0.0},
        start_val=10_000.0,
        start_date="2024-01-01",
        end_date="2024-01-05",
        prices_df=prices,
        rebalance_freq="None",
        dynamic_allocation_schedule=schedule,
    )

    assert port_series.loc["2024-01-02"] == pytest.approx(11_000.0)
