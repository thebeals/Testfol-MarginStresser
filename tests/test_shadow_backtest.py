"""Tests for app/core/shadow_backtest.py — pass prices_df to bypass yfinance."""

import pandas as pd
import numpy as np
import pytest
from app.core.shadow_backtest import parse_ticker, run_shadow_backtest


# ---------------------------------------------------------------------------
# parse_ticker
# ---------------------------------------------------------------------------

def test_parse_ticker_simple():
    """Plain ticker -> (ticker, {})."""
    base, params = parse_ticker("SPY")
    # SPY maps to itself (or SPYSIM maps to SPY, but plain SPY stays SPY)
    assert base == "SPY"
    assert params == {}


def test_parse_ticker_with_params():
    """Ticker with query params -> parsed correctly."""
    base, params = parse_ticker("SPY?L=2&SW=30")
    assert base == "SPY"
    assert params == {"L": "2", "SW": "30"}


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("SPYSIM", "SPY"),
        ("SPYTR", "SPY"),
        ("OEFSIM", "OEF"),
        ("MDYSIM", "MDY"),
        ("IJRSIM", "IJR"),
        ("IWMSIM", "IWM"),
        ("USMVSIM", "USMV"),
        ("KMLMSIM", "KMLM"),
        ("KMLMX", "KMLM"),
        ("GLDSIM", "GLD"),
        ("GOLDX", "GLD"),
        ("SLVSIM", "SLV"),
        ("SVIXSIM", "SVIX"),
        ("SVIXX", "SVIX"),
        ("UVIXSIM", "UVIX"),
        ("ZVOLSIM", "ZVOL"),
        ("ZIVBX", "ZVOL"),
        ("TLTSIM", "TLT"),
        ("TLTTR", "TLT"),
        ("ZROZSIM", "ZROZ"),
        ("ZROZX", "ZROZ"),
        ("VXUSSIM", "VXUS"),
        ("VXUSX", "VXUS"),
        ("EFASIM", "EFA"),
        ("VEASIM", "VEA"),
        ("VWOSIM", "VWO"),
        ("VSSSIM", "VSS"),
        ("EFVSIM", "EFV"),
        ("VTISIM", "VTI"),
        ("VTITR", "VTI"),
        ("VTSIM", "VT"),
        ("DBMFSIM", "DBMF"),
        ("DBMFX", "DBMF"),
        ("VIXSIM", "^VIX"),
        ("VOLIX", "^VIX"),
        ("GSGSIM", "GSG"),
        ("GSGTR", "GSG"),
        ("IEFSIM", "IEF"),
        ("IEFTR", "IEF"),
        ("IEISIM", "IEI"),
        ("IEITR", "IEI"),
        ("SHYSIM", "SHY"),
        ("SHYTR", "SHY"),
        ("TIPSIM", "TIP"),
        ("BTCSIM", "BTC-USD"),
        ("BTCTR", "BTC-USD"),
        ("ETHSIM", "ETH-USD"),
        ("ETHTR", "ETH-USD"),
        ("MTUMSIM", "MTUM"),
        ("XLBSIM", "XLB"),
        ("XLBTR", "XLB"),
        ("XLCSIM", "XLC"),
        ("XLCTR", "XLC"),
        ("XLESIM", "XLE"),
        ("XLETR", "XLE"),
        ("XLFSIM", "XLF"),
        ("XLFTR", "XLF"),
        ("XLISIM", "XLI"),
        ("XLITR", "XLI"),
        ("XLKSIM", "XLK"),
        ("XLKTR", "XLK"),
        ("XLPSIM", "XLP"),
        ("XLPTR", "XLP"),
        ("XLUSIM", "XLU"),
        ("XLUTR", "XLU"),
        ("XLVSIM", "XLV"),
        ("XLVTR", "XLV"),
        ("XLYSIM", "XLY"),
        ("XLYTR", "XLY"),
        ("QQQSIM", "QQQ"),
        ("QQQTR", "QQQ"),
        ("CAOSSIM", "CAOS"),
        ("FNGUSIM", "FNGU"),
        ("MCISIM", "MCI"),
        ("GDESIM", "GDE"),
        ("RSSBSIM", "RSSB"),
        ("NTSDSIM", "NTSD"),
        ("UUPSIM", "UUP"),
        ("VVSIM", "VOO"),
        ("VOOSIM", "VOO"),
        ("VTVSIM", "VTV"),
        ("VUGSIM", "VUG"),
        ("VOSIM", "VO"),
        ("VOESIM", "VOE"),
        ("VOTSIM", "VOT"),
        ("VBSIM", "VB"),
        ("VBRSIM", "VBR"),
        ("VBKSIM", "VBK"),
        ("IWCSIM", "IWC"),
        ("BNDSIM", "BND"),
        ("REITSIM", "VNQ"),
        ("TBILL", "BIL"),
        ("CASHX", "BIL"),
        ("EFFRX", "BIL"),
    ],
)
def test_parse_ticker_special_provider_fallbacks(ticker, expected):
    base, params = parse_ticker(f"{ticker}?L=2")
    assert base == expected
    assert params == {"L": "2"}


def test_parse_ticker_keeps_non_provider_presets_for_api_handling():
    from app.common.special_tickers import is_testfol_preset_ticker

    assert parse_ticker("ZEROX")[0] == "ZEROX"
    assert parse_ticker("INFLATION")[0] == "INFLATION"
    assert is_testfol_preset_ticker("FF3MKT")
    assert is_testfol_preset_ticker("FF5CMA")


# ---------------------------------------------------------------------------
# run_shadow_backtest — basic scenarios
# ---------------------------------------------------------------------------

def test_basic_flat_backtest(flat_prices, sample_allocation):
    """Flat prices -> portfolio value stays at start_val."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    assert not port_series.empty
    # With flat prices, portfolio should stay near start_val
    assert port_series.iloc[-1] == pytest.approx(10000.0, rel=0.01)


def test_growth_backtest(growing_prices, sample_allocation):
    """Prices double -> portfolio roughly doubles, CAGR ~ 100% over the period."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2020-01-02",
        end_date="2024-12-31",
        prices_df=growing_prices,
        rebalance_freq="Yearly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    assert not port_series.empty
    # Prices go from 100 to 200 => portfolio should roughly double
    final_val = port_series.iloc[-1]
    assert 18000 < final_val < 22000


def test_shadow_backtest_does_not_forward_fill_stale_component_prices():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {
            "SIM": [100.0, 101.0, None],
            "LIVE": [100.0, 101.0, 102.0],
        },
        index=dates,
    )

    result = run_shadow_backtest(
        allocation={"SIM": 50.0, "LIVE": 50.0},
        start_val=10000.0,
        start_date="2024-01-02",
        end_date="2024-01-04",
        prices_df=prices,
        rebalance_freq="Yearly",
    )

    _, _, _, _, logs, port_series, _, *_ = result
    assert not port_series.empty, logs
    assert port_series.index.max() == pd.Timestamp("2024-01-03")


def test_leveraged_returns_use_fed_funds_for_financing(monkeypatch):
    from app.services import data_service

    dates = pd.bdate_range("2024-01-02", "2024-01-12")
    prices = pd.DataFrame({"TEST": 100.0}, index=dates)

    def run_with_rate(rate_pct):
        fed_funds = pd.Series(
            rate_pct,
            index=pd.date_range(dates.min(), dates.max(), freq="D"),
        )
        monkeypatch.setattr(data_service, "get_fed_funds_rate", lambda: fed_funds)
        result = run_shadow_backtest(
            allocation={"TEST?L=2&SW=1&SP=0&E=0": 100.0},
            start_val=10000.0,
            start_date=str(dates.min().date()),
            end_date=str(dates.max().date()),
            prices_df=prices,
            rebalance_freq="Yearly",
        )
        _, _, _, _, logs, port_series, _, *_ = result
        return logs, port_series

    zero_logs, zero_rate_port = run_with_rate(0.0)
    high_logs, high_rate_port = run_with_rate(10.0)

    assert zero_rate_port.iloc[-1] == pytest.approx(10000.0)
    assert high_rate_port.iloc[-1] < zero_rate_port.iloc[-1]
    assert any("Leverage Financing Source: DFF / 252" in entry for entry in zero_logs)
    assert any("Leverage Financing Source: DFF / 252" in entry for entry in high_logs)


def test_leveraged_returns_apply_default_implementation_spread(monkeypatch):
    from app.services import data_service

    dates = pd.bdate_range("2024-01-02", "2024-01-12")
    prices = pd.DataFrame({"TEST": 100.0}, index=dates)
    fed_funds = pd.Series(0.0, index=pd.date_range(dates.min(), dates.max(), freq="D"))
    monkeypatch.setattr(data_service, "get_fed_funds_rate", lambda: fed_funds)

    no_spread = run_shadow_backtest(
        allocation={"TEST?L=2&SW=1&SP=0&E=0": 100.0},
        start_val=10000.0,
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
        prices_df=prices,
        rebalance_freq="Yearly",
    )[5]
    default_spread = run_shadow_backtest(
        allocation={"TEST?L=2&SW=1&E=0": 100.0},
        start_val=10000.0,
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
        prices_df=prices,
        rebalance_freq="Yearly",
    )[5]

    assert no_spread.iloc[-1] == pytest.approx(10000.0)
    assert default_spread.iloc[-1] < no_spread.iloc[-1]


def test_dca_monthly(flat_prices, sample_allocation):
    """$1000/month cashflow -> correct number of buy trades."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
        cashflow=1000.0,
        cashflow_freq="Monthly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    # Should have roughly 11-12 monthly cashflow buy events
    if not trades_df.empty:
        buy_trades = trades_df[trades_df["Trade Amount"] > 0]
        assert len(buy_trades) >= 10  # At least 10 months of DCA


def test_single_asset_dca_buys_target_before_rebalance():
    """Targeted DCA should buy one sleeve first, then let the rebalance trade back to weights."""
    dates = pd.to_datetime(["2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 100.0],
            "B": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    result = run_shadow_backtest(
        allocation={"A": 60.0, "B": 40.0},
        start_val=10000.0,
        start_date="2023-01-27",
        end_date="2023-02-01",
        prices_df=prices,
        rebalance_freq="Monthly",
        cashflow=1000.0,
        cashflow_freq="Monthly",
        dca_mode="Single Asset",
        dca_target_ticker="A",
    )

    trades_df, _, _, _, logs, port_series, _, *_ = result
    jan31_trades = trades_df[trades_df["Date"] == pd.Timestamp("2023-01-31")]

    assert port_series.loc[pd.Timestamp("2023-01-31")] == pytest.approx(11000.0)
    assert not jan31_trades[
        (jan31_trades["Ticker"] == "A")
        & (jan31_trades["Trade Amount"].sub(1000.0).abs() < 1e-9)
    ].empty
    assert not jan31_trades[
        (jan31_trades["Ticker"] == "A")
        & (jan31_trades["Trade Amount"].sub(-400.0).abs() < 1e-9)
    ].empty
    assert not jan31_trades[
        (jan31_trades["Ticker"] == "B")
        & (jan31_trades["Trade Amount"].sub(400.0).abs() < 1e-9)
    ].empty

    dca_log_idx = next(idx for idx, entry in enumerate(logs) if "[DCA]" in entry)
    rebal_log_idx = next(idx for idx, entry in enumerate(logs) if "[REBALANCE]" in entry)
    assert dca_log_idx < rebal_log_idx


def test_return_structure(flat_prices, sample_allocation):
    """Verify 7-tuple, correct types."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
    )
    assert len(result) == 8
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(logs, list)
    assert isinstance(port_series, pd.Series)


def test_monthly_rebalance_keeps_running_lot_basis_consistent():
    """Selling at rebalance should update realized gains and unrealized basis correctly."""
    dates = pd.to_datetime(["2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0, 110.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    trades_df, _, _, unrealized_pl_df, _, port_series, _, *_ = run_shadow_backtest(
        allocation={"A": 50.0, "B": 50.0},
        start_val=100.0,
        start_date="2023-01-27",
        end_date="2023-02-02",
        prices_df=prices,
        rebalance_freq="Monthly",
    )

    assert port_series.loc[pd.Timestamp("2023-01-31")] == pytest.approx(105.0)

    sell_trade = trades_df[(trades_df["Ticker"] == "A") & (trades_df["Trade Amount"] < 0)].iloc[0]
    buy_trade = trades_df[(trades_df["Ticker"] == "B") & (trades_df["Trade Amount"] > 0)].iloc[0]

    assert sell_trade["Trade Amount"] == pytest.approx(-2.5)
    assert sell_trade["Realized ST P&L"] == pytest.approx(0.2272727273, rel=1e-6)
    assert buy_trade["Trade Amount"] == pytest.approx(2.5)

    jan_unrealized = unrealized_pl_df.loc[pd.Timestamp("2023-01-31"), "Unrealized P&L"]
    assert jan_unrealized == pytest.approx(4.7727272727, rel=1e-6)
