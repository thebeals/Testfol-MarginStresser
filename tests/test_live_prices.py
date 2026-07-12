import math
import sys
from types import SimpleNamespace

import pandas as pd

from app.common.special_tickers import live_price_fallback_ticker
from app.services.live_prices import build_live_returns_snapshot, fetch_yahoo_live_price_series


def test_live_price_fallback_ticker_maps_simulated_symbols():
    assert live_price_fallback_ticker("SPYSIM?L=2") == "SPY"
    assert live_price_fallback_ticker("AVGO?L=2&E=1.00") == "AVL"
    assert live_price_fallback_ticker("AMZN?L=2&E=0.99") == "AMZU"
    assert live_price_fallback_ticker("NVDA?L=2&E=0.92") == "NVDU"
    assert live_price_fallback_ticker("GOOG?L=2&E=0.96") == "GGLL"
    assert live_price_fallback_ticker("GOOGL?L=2&E=0.96") == "GGLL"
    assert live_price_fallback_ticker("META?L=2&E=1.02") == "METU"
    assert live_price_fallback_ticker("AAPL?L=2&E=0.96") == "AAPU"
    assert live_price_fallback_ticker("MSFT?L=2&E=0.98") == "MSFU"
    assert live_price_fallback_ticker("TSLA?L=2&E=0.83") == "TSLL"
    assert live_price_fallback_ticker("QQQSIM?L=2") == "QLD"
    assert live_price_fallback_ticker("QQQSIM?L=3") == "TQQQ"
    assert live_price_fallback_ticker("NDXMEGASIM") == "QBIG"
    assert live_price_fallback_ticker("NDXMEGASIM?L=2") == "QQUP"
    assert live_price_fallback_ticker("NDXMEGA2SIM") == "QBIG"
    assert live_price_fallback_ticker("QQUPSIM") == "QQUP"
    assert live_price_fallback_ticker("NDX30SIM") == "QTOP"


def test_build_live_returns_snapshot_uses_live_proxy_and_leverage():
    ref_date = pd.Timestamp("2026-04-27")
    port_series = pd.Series([1000.0], index=[ref_date])

    def fake_fetcher(symbols, reference_date):
        assert tuple(symbols) == ("BND", "SPY")
        assert reference_date == ref_date
        next_date = ref_date + pd.Timedelta(days=1)
        return {
            "SPY": {
                "prices": pd.Series([100.0, 110.0], index=[ref_date, next_date]),
                "live_time": next_date,
                "error": None,
            },
            "BND": {
                "prices": pd.Series([100.0, 90.0], index=[ref_date, next_date]),
                "live_time": next_date,
                "error": None,
            },
        }

    snapshot = build_live_returns_snapshot(
        port_series,
        allocation={"SPYSIM?L=2": 50.0, "BND": 50.0},
        price_fetcher=fake_fetcher,
    )

    assert snapshot["ok"] is True
    assert math.isclose(snapshot["live_return"], 0.05, rel_tol=1e-9)

    rows = snapshot["rows"].set_index("Ticker")
    assert rows.loc["SPYSIM?L=2", "Live Ticker"] == "SPY"
    assert math.isclose(rows.loc["SPYSIM?L=2", "Effective Return"], 0.20, rel_tol=1e-9)
    assert math.isclose(rows.loc["BND", "Effective Return"], -0.10, rel_tol=1e-9)


def test_build_live_returns_snapshot_does_not_double_apply_native_leveraged_etfs():
    ref_date = pd.Timestamp("2026-04-27")
    port_series = pd.Series([1000.0], index=[ref_date])

    def fake_fetcher(symbols, reference_date):
        assert tuple(symbols) == ("QLD",)
        next_date = ref_date + pd.Timedelta(days=1)
        return {
            "QLD": {
                "prices": pd.Series([100.0, 110.0], index=[ref_date, next_date]),
                "live_time": next_date,
                "error": None,
            },
        }

    snapshot = build_live_returns_snapshot(
        port_series,
        allocation={"QQQSIM?L=2": 100.0},
        price_fetcher=fake_fetcher,
    )

    assert snapshot["ok"] is True
    assert math.isclose(snapshot["live_return"], 0.10, rel_tol=1e-9)

    rows = snapshot["rows"].set_index("Ticker")
    assert rows.loc["QQQSIM?L=2", "Live Ticker"] == "QLD"
    assert math.isclose(rows.loc["QQQSIM?L=2", "Leverage"], 2.0, rel_tol=1e-9)
    assert math.isclose(rows.loc["QQQSIM?L=2", "Effective Return"], 0.10, rel_tol=1e-9)


def test_build_live_returns_snapshot_uses_native_single_stock_leveraged_etfs():
    ref_date = pd.Timestamp("2026-04-27")
    port_series = pd.Series([1000.0], index=[ref_date])

    def fake_fetcher(symbols, reference_date):
        assert tuple(symbols) == ("AMZU",)
        next_date = ref_date + pd.Timedelta(days=1)
        return {
            "AMZU": {
                "prices": pd.Series([100.0, 105.0], index=[ref_date, next_date]),
                "live_time": next_date,
                "error": None,
            },
        }

    snapshot = build_live_returns_snapshot(
        port_series,
        allocation={"AMZN?L=2&E=0.99": 100.0},
        price_fetcher=fake_fetcher,
    )

    rows = snapshot["rows"].set_index("Ticker")
    assert rows.loc["AMZN?L=2&E=0.99", "Live Ticker"] == "AMZU"
    assert math.isclose(snapshot["live_return"], 0.05, rel_tol=1e-9)
    assert math.isclose(rows.loc["AMZN?L=2&E=0.99", "Effective Return"], 0.05, rel_tol=1e-9)


def test_build_live_returns_snapshot_scales_latest_composition_to_displayed_value():
    ref_date = pd.Timestamp("2026-04-27")
    port_series = pd.Series([1000.0], index=[ref_date])
    composition_df = pd.DataFrame(
        [
            {"Date": ref_date, "Ticker": "SPY", "Value": 1500.0},
            {"Date": ref_date, "Ticker": "BND", "Value": 500.0},
        ]
    )

    def fake_fetcher(symbols, reference_date):
        next_date = ref_date + pd.Timedelta(days=1)
        flat = pd.Series([100.0, 100.0], index=[ref_date, next_date])
        return {
            symbol: {"prices": flat, "live_time": next_date, "error": None}
            for symbol in symbols
        }

    snapshot = build_live_returns_snapshot(
        port_series,
        allocation={"SPY": 50.0, "BND": 50.0},
        composition_df=composition_df,
        price_fetcher=fake_fetcher,
    )

    rows = snapshot["rows"].set_index("Ticker")
    assert math.isclose(rows.loc["SPY", "Position Value"], 750.0, rel_tol=1e-9)
    assert math.isclose(rows.loc["BND", "Position Value"], 250.0, rel_tol=1e-9)


def test_fetch_yahoo_live_price_series_uses_extended_hours_intraday(monkeypatch):
    ref_date = pd.Timestamp("2026-04-27")
    after_hours_time = pd.Timestamp("2026-04-28 16:30", tz="America/New_York")
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return {
                "marketState": "POST",
                "regularMarketPrice": 100.0,
                "regularMarketTime": int(pd.Timestamp("2026-04-28 16:00", tz="America/New_York").timestamp()),
                "postMarketPrice": 105.0,
                "postMarketTime": int(pd.Timestamp("2026-04-28 16:15", tz="America/New_York").timestamp()),
            }

        @property
        def fast_info(self):
            return {"last_price": 104.0}

        def history(self, *args, **kwargs):
            calls.append(kwargs)
            if kwargs.get("interval") == "1m":
                assert kwargs.get("prepost") is True
                return pd.DataFrame({"Close": [106.0]}, index=[after_hours_time])
            return pd.DataFrame({"Close": [100.0]}, index=[ref_date])

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))

    result = fetch_yahoo_live_price_series(("AAPL",), ref_date)
    quote = result["AAPL"]

    assert quote["error"] is None
    assert math.isclose(quote["live_price"], 106.0, rel_tol=1e-9)
    assert quote["live_time"] == pd.Timestamp("2026-04-28 16:30")
    assert quote["market_session"] == "After-hours"
    assert any(call.get("prepost") is True for call in calls)


def test_build_live_returns_snapshot_reports_market_session():
    ref_date = pd.Timestamp("2026-04-27")
    port_series = pd.Series([1000.0], index=[ref_date])

    def fake_fetcher(symbols, reference_date):
        next_date = ref_date + pd.Timedelta(hours=18)
        return {
            "SPY": {
                "prices": pd.Series([100.0, 101.0], index=[ref_date, next_date]),
                "live_time": next_date,
                "market_session": "After-hours",
                "error": None,
            },
        }

    snapshot = build_live_returns_snapshot(
        port_series,
        allocation={"SPY": 100.0},
        price_fetcher=fake_fetcher,
    )

    assert snapshot["market_session"] == "After-hours"
    rows = snapshot["rows"].set_index("Ticker")
    assert rows.loc["SPY", "Market Session"] == "After-hours"
