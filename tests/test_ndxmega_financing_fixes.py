import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from app.common.constants import Tickers
from app.core.margin_interest import UsdMarginInterestLedger
from app.core.shadow_backtest import run_shadow_backtest
from app.services import data_service
from app.services.testfol_api import simulate_margin


def _run_twr(monkeypatch, ticker, prices, dff_pct=0.0):
    dff = pd.Series(
        dff_pct,
        index=pd.date_range(prices.index.min(), prices.index.max(), freq="D"),
    )
    monkeypatch.setattr(data_service, "get_fed_funds_rate", lambda: dff)
    return run_shadow_backtest(
        {ticker: 100.0},
        10_000.0,
        str(prices.index.min().date()),
        str(prices.index.max().date()),
        prices_df=prices,
    )[6]


def test_first_investable_return_is_compounded():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"TEST": [100.0, 110.0, 121.0]}, index=dates)

    result = run_shadow_backtest(
        {"TEST": 100.0},
        10_000.0,
        "2024-01-02",
        "2024-01-04",
        prices_df=prices,
    )

    twr = result[6]
    assert twr.to_list() == pytest.approx([1.0, 1.1, 1.21])
    assert result[2]["Value"].iloc[-1] == pytest.approx(12_100.0)


def test_prior_session_anchor_preserves_first_in_window_return():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"TEST": [100.0, 110.0, 121.0]}, index=dates)

    result = run_shadow_backtest(
        {"TEST": 100.0},
        10_000.0,
        "2024-01-01",
        "2024-01-03",
        prices_df=prices,
    )

    assert result[6].to_list() == pytest.approx([1.0, 1.1, 1.21])
    assert result[6].index[0] == pd.Timestamp("2023-12-29")
    assert result[2]["Value"].iloc[-1] == pytest.approx(12_100.0)


def test_prior_price_anchor_does_not_predate_margin_accrual(monkeypatch):
    recorded = {}

    class RecordingLedger:
        def __init__(self, dates, starting_balance, annual_rate_pct):
            recorded["dates"] = pd.DatetimeIndex(dates)
            recorded["advances"] = []
            self.total_liability = starting_balance

        def advance(self, date, loan_change=0.0):
            recorded["advances"].append(pd.Timestamp(date))
            self.total_liability += loan_change
            return SimpleNamespace(total_liability=self.total_liability)

    monkeypatch.setattr("app.core.margin_interest.UsdMarginInterestLedger", RecordingLedger)
    prices = pd.DataFrame(
        {"TEST": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"]),
    )

    run_shadow_backtest(
        {"TEST": 100.0},
        10_000.0,
        "2024-01-01",
        "2024-01-03",
        prices_df=prices,
        starting_loan=1_000.0,
    )

    assert recorded["dates"][0] == pd.Timestamp("2024-01-01")
    assert recorded["advances"][0] == pd.Timestamp("2024-01-01")


def test_margin_simulator_does_not_accrue_on_prior_price_anchor():
    port = pd.Series(
        10_000.0,
        index=pd.to_datetime(["2023-12-29", "2024-01-02"]),
    )

    loan, *_ = simulate_margin(
        port,
        starting_loan=1_000.0,
        rate_annual=36.0,
        draw_monthly=0.0,
        maint_pct=0.25,
        accrual_start_date="2024-01-01",
    )

    assert loan.loc["2023-12-29"] == pytest.approx(1_000.0)
    assert loan.loc["2024-01-02"] == pytest.approx(1_002.0)


@pytest.mark.parametrize(
    ("leverage", "expected"),
    [
        (2, [1.0, 1.2, 0.96]),
        (3, [1.0, 1.3, 0.91]),
    ],
)
def test_zero_cost_leverage_is_exact_daily_multiple(monkeypatch, leverage, expected):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"TEST": [100.0, 110.0, 99.0]}, index=dates)
    twr = _run_twr(
        monkeypatch,
        f"TEST?L={leverage}&SW=0&SP=0&E=0",
        prices,
    )
    assert twr.to_list() == pytest.approx(expected, abs=1e-12)


def test_testfol_cost_formula_uses_simple_252_and_no_weekend_multiplier(monkeypatch):
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    prices = pd.DataFrame({"TEST": [100.0, 100.0, 100.0]}, index=dates)
    twr = _run_twr(
        monkeypatch,
        "TEST?L=3&SW=1.2&SP=1&FR=EFFRX&E=1",
        prices,
        dff_pct=5.33,
    )

    expected_cost = 1.2 * (3 - 1) * ((5.33 / 100) / 252 + (1 / 100) / 252)
    expected_cost += (1 / 100) / 252
    observed = twr.pct_change().dropna()
    assert observed.to_list() == pytest.approx([-expected_cost, -expected_cost], abs=1e-12)


def test_default_financing_rate_matches_explicit_effrx(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"TEST": [100.0, 101.0, 100.5, 102.0]}, index=dates)
    implicit = _run_twr(monkeypatch, "TEST?L=2&SW=1&SP=0&E=0", prices, 5.33)
    explicit = _run_twr(monkeypatch, "TEST?L=2&SW=1&SP=0&FR=EFFRX&E=0", prices, 5.33)
    pd.testing.assert_series_equal(implicit, explicit)


@pytest.mark.parametrize(
    ("implicit_ticker", "explicit_ticker"),
    [
        ("TEST?L=0", "TEST?L=0&SW=1.1&SP=0&E=0"),
        ("TEST?L=-1", f"TEST?L=-1&SW=1.1&SP=-0.4&E={1 / 3}"),
        ("TEST?L=-2", f"TEST?L=-2&SW=1.1&SP=-0.4&E={2 / 3}"),
    ],
)
def test_default_spread_uses_leverage_sign(monkeypatch, implicit_ticker, explicit_ticker):
    dates = pd.bdate_range("2020-03-02", periods=8)
    prices = pd.DataFrame({"TEST": np.linspace(100.0, 106.0, len(dates))}, index=dates)
    implicit = _run_twr(monkeypatch, implicit_ticker, prices, 1.09)
    explicit = _run_twr(monkeypatch, explicit_ticker, prices, 1.09)
    pd.testing.assert_series_equal(implicit, explicit)


def test_dff_daily_source_is_preserved(monkeypatch):
    raw = pd.Series(
        [1.59, 1.09, 0.25],
        index=pd.to_datetime(["2020-03-03", "2020-03-04", "2020-03-16"]),
    )
    monkeypatch.setattr(data_service, "_get_fred_series", lambda *args, **kwargs: raw)
    data_service.get_fed_funds_rate.cache_clear()
    try:
        result = data_service.get_fed_funds_rate()
        assert result.loc["2020-03-03"] == pytest.approx(1.59)
        assert result.loc["2020-03-04"] == pytest.approx(1.09)
        assert result.loc["2020-03-16"] == pytest.approx(0.25)
    finally:
        data_service.get_fed_funds_rate.cache_clear()


def test_local_rate_pseudo_ticker_uses_testfol_simple_252():
    rates = pd.Series(
        5.04,
        index=pd.date_range("2024-01-05", "2024-01-09", freq="D"),
    )
    prices = data_service._build_total_return_from_annual_rates(
        rates,
        "2024-01-05",
        "2024-01-09",
        "EFFRX",
    )
    observed = prices.pct_change().dropna()
    assert observed.to_list() == pytest.approx([0.0504 / 252, 0.0504 / 252])


def test_local_rate_pseudo_ticker_uses_actual_xnys_sessions():
    rates = pd.Series(
        5.0,
        index=pd.to_datetime(["2001-09-01", "2012-10-01", "2022-06-01"]),
    )

    september_2001 = data_service._build_total_return_from_annual_rates(
        rates, "2001-09-10", "2001-09-18", "EFFRX"
    )
    sandy_2012 = data_service._build_total_return_from_annual_rates(
        rates, "2012-10-26", "2012-10-31", "EFFRX"
    )
    juneteenth_2022 = data_service._build_total_return_from_annual_rates(
        rates, "2022-06-17", "2022-06-21", "EFFRX"
    )

    assert list(september_2001.index) == list(
        pd.to_datetime(["2001-09-10", "2001-09-17", "2001-09-18"])
    )
    assert list(sandy_2012.index) == list(pd.to_datetime(["2012-10-26", "2012-10-31"]))
    assert list(juneteenth_2022.index) == list(
        pd.to_datetime(["2022-06-17", "2022-06-21"])
    )


def test_caps_apply_after_ue_and_before_leverage(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = pd.DataFrame({"TEST": [100.0, 103.0, 101.0, 102.0]}, index=dates)
    twr = _run_twr(
        monkeypatch,
        "TEST?UE=5&CU=0.5&CL=-0.25&L=2&SW=0&E=0",
        prices,
    )
    underlying = prices["TEST"].pct_change().dropna() + 0.05 / 252
    expected = 2.0 * underlying.clip(lower=-0.0025, upper=0.005)
    assert twr.pct_change().dropna().to_list() == pytest.approx(expected.to_list(), abs=1e-12)


@pytest.mark.parametrize(
    ("reference", "loader_name", "annual_pct"),
    [
        ("CASHX", "get_tbill_rate", 5.04),
        ("TBILL", "get_tbill_rate", 5.04),
        ("DGS3MO", "get_fred_yield_rate", 5.44),
    ],
)
def test_financing_reference_uses_named_simple_252_series(
    monkeypatch,
    reference,
    loader_name,
    annual_pct,
):
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"TEST": 100.0}, index=dates)
    dff = pd.Series(5.33, index=pd.date_range(dates[0], dates[-1], freq="D"))
    named = pd.Series(annual_pct, index=pd.date_range(dates[0], dates[-1], freq="D"))
    monkeypatch.setattr(data_service, "get_fed_funds_rate", lambda: dff)
    if loader_name == "get_fred_yield_rate":
        monkeypatch.setattr(data_service, loader_name, lambda series_id: named)
    else:
        monkeypatch.setattr(data_service, loader_name, lambda: named)

    result = run_shadow_backtest(
        {f"TEST?L=2&SW=1&SP=0&FR={reference}&E=0": 100.0},
        10_000.0,
        str(dates[0].date()),
        str(dates[-1].date()),
        prices_df=prices,
    )
    observed = result[6].pct_change().dropna()
    assert observed.to_list() == pytest.approx([-annual_pct / 100 / 252] * 4, abs=1e-12)
    assert any(f"FR={reference}" in line for line in result[4])


def test_unsupported_modifier_is_reported_not_silently_accepted(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"TEST": [100.0, 101.0, 102.0]}, index=dates)
    result = _run_twr(monkeypatch, "TEST?UR=9", prices)
    assert not result.empty
    full_result = run_shadow_backtest(
        {"TEST?UR=9": 100.0},
        10_000.0,
        str(dates[0].date()),
        str(dates[-1].date()),
        prices_df=prices,
    )
    assert any("unsupported Testfol modifier(s): UR" in line for line in full_result[4])


def test_legacy_drag_alias_matches_explicit_expense(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"TEST": [100.0, 101.0, 102.0, 103.0]}, index=dates)

    expense = _run_twr(monkeypatch, "TEST?E=0.5", prices)
    drag = _run_twr(monkeypatch, "TEST?D=0.5", prices)

    pd.testing.assert_series_equal(drag, expense)


def test_ndxmega_official_series_mappings():
    assert data_service.NDX_MEGA_FRED_SERIES == {
        Tickers.NDXMEGASIM: "NASDAQNDXMEGAT",
        Tickers.NDXMEGA2SIM: "NASDAQNDXMEGA2T",
    }


def test_qqupsim_formula_uses_price_returns_and_testfol_costs():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    underlying = pd.Series([100.0, 110.0, 99.0], index=dates)
    dff = pd.Series(5.33, index=dates)

    result = data_service._build_testfol_leveraged_price_series(
        underlying,
        dff,
        name="QQUPSIM",
    )

    daily_cost = 1.10 * ((5.33 / 100) / 252 + (0.40 / 100) / 252)
    daily_cost += (0.95 / 100) / 252
    expected_returns = [0.0, 2 * 0.10 - daily_cost, 2 * -0.10 - daily_cost]
    assert result.iloc[0] == pytest.approx(100.0)
    assert result.pct_change().fillna(0.0).to_list() == pytest.approx(expected_returns)


def test_qqupsim_formula_does_not_multiply_weekend_cost():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    underlying = pd.Series(100.0, index=dates)
    dff = pd.Series(5.0, index=pd.date_range(dates[0], dates[-1], freq="D"))

    result = data_service._build_testfol_leveraged_price_series(
        underlying,
        dff,
        name="QQUPSIM",
    )
    expected_cost = 1.10 * ((5.0 / 100) / 252 + (0.40 / 100) / 252)
    expected_cost += (0.95 / 100) / 252
    assert result.pct_change().dropna().to_list() == pytest.approx(
        [-expected_cost, -expected_cost]
    )


def test_qqupsim_splice_preserves_actual_returns_and_does_not_rededuct_expense(
    monkeypatch,
):
    official_dates = pd.to_datetime(
        ["2024-07-29", "2024-07-30", "2025-06-10", "2025-06-11", "2025-06-12"]
    )
    official = pd.Series([1000.0, 1010.0, 1200.0, 1210.0, 1220.0], index=official_dates)
    actual_dates = pd.to_datetime(["2025-06-10", "2025-06-11", "2025-06-12"])
    actual = pd.Series([100.0, 102.0, 99.0], index=actual_dates)

    class RecordingProvider:
        def __init__(self):
            self.calls = []

        def fetch_prices(self, tickers, start_date, end_date):
            self.calls.append(tuple(tickers))
            return pd.DataFrame({"QQUP": actual})

    provider = RecordingProvider()
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)
    monkeypatch.setattr(data_service, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_service, "cache_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_service, "_get_fred_series", lambda *args, **kwargs: official)
    monkeypatch.setattr(
        data_service,
        "get_fed_funds_rate",
        lambda: pd.Series(0.0, index=pd.date_range("2000-01-01", "2025-06-12")),
    )

    result = data_service.fetch_component_data(
        [Tickers.QQUPSIM],
        "2025-06-09",
        "2025-06-12",
        sync_end=False,
    )[Tickers.QQUPSIM]

    assert provider.calls == [("QQUP",)]
    assert result.loc["2025-06-11"] / result.loc["2025-06-10"] - 1 == pytest.approx(0.02)
    assert result.loc["2025-06-12"] / result.loc["2025-06-11"] - 1 == pytest.approx(
        99 / 102 - 1
    )


def test_price_return_reconstruction_is_distinct_and_tracks_official_index():
    price = pd.read_csv(
        "data/NDXMEGAPRICESIM.csv", parse_dates=["Date"]
    ).set_index("Date")["Close"].dropna()
    total = pd.read_csv(
        "data/NDXMEGASIM.csv", parse_dates=["Date"]
    ).set_index("Date")["Close"].dropna()
    official = pd.read_csv(
        "data/NASDAQNDXMEGA.csv", parse_dates=["observation_date"]
    ).set_index("observation_date")["NASDAQNDXMEGA"]
    official = pd.to_numeric(official, errors="coerce").dropna()

    common = price.index.intersection(total.index)
    price_growth = price.loc[common].iloc[-1] / price.loc[common].iloc[0]
    total_growth = total.loc[common].iloc[-1] / total.loc[common].iloc[0]
    assert price.index[0] == pd.Timestamp("2000-06-30")
    assert price_growth < total_growth

    official_common = price.index.intersection(official.index)
    price_returns = price.loc[official_common].pct_change().dropna()
    official_returns = official.loc[official_common].pct_change().dropna()
    assert price_returns.corr(official_returns) > 0.99


def test_official_splice_keeps_anchor_and_first_official_return():
    local = pd.Series(
        [100.0, 102.0, 104.0, 106.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    official = pd.Series(
        [200.0, 204.0, 210.0, 207.0],
        index=pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]),
    )

    combined = data_service._splice_official_total_return(local, official, name="NDXMEGASIM")
    anchor = pd.Timestamp("2024-01-04")
    assert combined.loc[:anchor].to_list() == pytest.approx(local.loc[:anchor].to_list())
    assert combined.loc["2024-01-05"] / combined.loc[anchor] - 1 == pytest.approx(0.02)
    assert combined.loc["2024-01-08"] / combined.loc["2024-01-05"] - 1 == pytest.approx(210 / 204 - 1)
    assert not np.isclose(combined.pct_change().loc["2024-01-05"], 0.0)


def test_ndxmega_historical_fetch_does_not_request_qbig(monkeypatch):
    local = pd.read_csv("data/NDXMEGASIM.csv", parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    anchor_dates = local.index[-3:]
    official = pd.Series([100.0, 101.0, 102.0], index=anchor_dates)

    class RecordingProvider:
        def __init__(self):
            self.calls = []

        def fetch_prices(self, tickers, start_date, end_date):
            self.calls.append(tuple(tickers))
            return pd.DataFrame()

    provider = RecordingProvider()
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)
    monkeypatch.setattr(data_service, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_service, "cache_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_service, "_get_fred_series", lambda *args, **kwargs: official)

    result = data_service.fetch_component_data(
        [Tickers.NDXMEGASIM],
        anchor_dates[0],
        anchor_dates[-1],
        sync_end=False,
    )

    assert not result.empty
    assert provider.calls == []


def test_ibkr_one_day_actual_360_example():
    date = pd.Timestamp("2024-01-02")
    ledger = UsdMarginInterestLedger([date], 100_000.0, 6.82)
    snapshot = ledger.advance(date)
    assert snapshot.accrued_interest == pytest.approx(18.9444444444)
    assert snapshot.total_liability == pytest.approx(100_018.9444444444)


def test_actual_360_accrues_three_days_from_friday_to_monday():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    ledger = UsdMarginInterestLedger(dates, 100_000.0, 8.0)
    friday = ledger.advance(dates[0])
    monday = ledger.advance(dates[1])
    expected_daily = 100_000 * 0.08 / 360
    assert monday.total_liability - friday.total_liability == pytest.approx(3 * expected_daily)


def test_margin_interest_posts_on_third_observed_trading_day():
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"])
    ledger = UsdMarginInterestLedger(dates, 100_000.0, 8.0)
    snapshots = [ledger.advance(date) for date in dates]
    assert snapshots[3].settled_principal == pytest.approx(100_000.0)
    assert snapshots[4].settled_principal == pytest.approx(100_000 + 2 * 100_000 * 0.08 / 360)
    assert snapshots[4].accrued_interest > 0


def test_ibkr_tiered_one_day_example():
    date = pd.Timestamp("2024-01-02")
    config = {
        "type": "Tiered",
        "base_series": pd.Series([5.32], index=[date]),
        "tiers": [(0, 1.5), (100_000, 1.0)],
    }
    ledger = UsdMarginInterestLedger([date], 600_000.0, config)
    snapshot = ledger.advance(date)
    expected = 100_000 * 0.0682 / 360 + 500_000 * 0.0632 / 360
    assert snapshot.accrued_interest == pytest.approx(expected)
    assert snapshot.effective_rate_pct == pytest.approx(expected / 600_000 * 360 * 100)


def test_credit_balance_is_not_charged_interest():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    ledger = UsdMarginInterestLedger(dates, -10_000.0, 8.0)
    assert ledger.advance(dates[0]).accrued_interest == 0
    assert ledger.advance(dates[1]).total_liability == pytest.approx(-10_000.0)


def test_multi_decade_testfol_formula_does_not_drift(monkeypatch):
    dates = pd.bdate_range("1955-01-03", "2026-07-10")
    steps = np.arange(len(dates), dtype=float)
    underlying_returns = 0.00025 + 0.006 * np.sin(steps / 37.0)
    underlying_returns[0] = 0.0
    prices = pd.DataFrame(
        {"TEST": 100.0 * np.cumprod(1.0 + underlying_returns)},
        index=dates,
    )
    calendar_dates = pd.date_range(dates[0], dates[-1], freq="D")
    dff_pct = pd.Series(4.0 + 3.5 * np.sin(np.arange(len(calendar_dates)) / 503.0), index=calendar_dates)
    monkeypatch.setattr(data_service, "get_fed_funds_rate", lambda: dff_pct)

    twr = run_shadow_backtest(
        {"TEST?L=2": 100.0},
        10_000.0,
        str(dates[0].date()),
        str(dates[-1].date()),
        prices_df=prices,
    )[6]

    funding = dff_pct.reindex(dates).to_numpy() / 100.0 / 252.0
    expected_returns = 2.0 * underlying_returns - 1.1 * (funding + 0.004 / 252.0) - 0.005 / 252.0
    expected_returns[0] = 0.0
    expected_terminal = float(np.prod(1.0 + expected_returns))
    assert len(twr) == len(dates)
    assert twr.iloc[-1] == pytest.approx(expected_terminal, rel=2e-12)


def test_multi_decade_margin_ledger_stays_finite_and_respects_credit_balances():
    from app.services.testfol_api import simulate_margin

    dates = pd.bdate_range("1955-01-03", "2026-07-10")
    port = pd.Series(10_000_000.0, index=dates)
    debt, *_ = simulate_margin(port, 100_000.0, 8.0, 0.0, 0.25)
    credit, *_ = simulate_margin(port, -100_000.0, 8.0, 0.0, 0.25)

    assert np.isfinite(debt).all()
    assert debt.is_monotonic_increasing
    assert debt.iloc[-1] == pytest.approx(32_425_407.504952103, rel=1e-12)
    assert credit.to_numpy() == pytest.approx(np.full(len(credit), -100_000.0))
