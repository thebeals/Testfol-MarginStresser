"""Tests for app.core.backtest_orchestrator module."""

import pandas as pd
import pytest

from app.common.constants import Freq, RebalMode
from app.core import backtest_orchestrator as orchestrator
from app.core.backtest_orchestrator import calc_rebal_offset, run_multi_backtest, run_single_backtest


# ---------------------------------------------------------------------------
# calc_rebal_offset
# ---------------------------------------------------------------------------

class TestCalcRebalOffset:
    def test_standard_mode_returns_zero(self):
        """Standard mode always returns 0 offset."""
        reb = {"mode": RebalMode.STANDARD, "freq": Freq.YEARLY}
        assert calc_rebal_offset(reb, Freq.YEARLY) == 0

    def test_custom_yearly_jan(self):
        """Custom yearly rebalancing in January yields high offset (far from Dec 31)."""
        reb = {"mode": RebalMode.CUSTOM, "month": 1, "day": 1}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert offset > 200  # ~252 trading days minus a few

    def test_custom_yearly_dec(self):
        """Custom yearly rebalancing in December yields low offset."""
        reb = {"mode": RebalMode.CUSTOM, "month": 12, "day": 31}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert offset == 0

    def test_custom_monthly(self):
        """Custom monthly returns a small offset."""
        reb = {"mode": RebalMode.CUSTOM, "month": 1, "day": 15}
        offset = calc_rebal_offset(reb, Freq.MONTHLY)
        assert 0 <= offset <= 21  # trading days in a month

    def test_missing_keys_defaults(self):
        """Missing month/day keys use defaults without error."""
        reb = {"mode": RebalMode.CUSTOM}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert isinstance(offset, int)


def test_dynamic_price_slice_only_syncs_constituents_active_at_end_date():
    """A historical delisting must not truncate a dynamic portfolio's end date."""
    dates = pd.to_datetime(["2005-01-03", "2005-01-04", "2026-08-06"])
    prices = pd.DataFrame(
        {
            "OLD": [100.0, 101.0, float("nan")],
            "LIVE": [50.0, 51.0, 200.0],
        },
        index=dates,
    )

    sliced = orchestrator._slice_prefetched_component_prices(
        prices,
        ["OLD", "LIVE"],
        "2005-01-03",
        "2026-08-06",
        sync_tickers=["LIVE"],
    )

    assert sliced.index.max() == pd.Timestamp("2026-08-06")
    assert list(sliced.columns) == ["OLD", "LIVE"]


# ---------------------------------------------------------------------------
# run_single_backtest
# ---------------------------------------------------------------------------

def _mock_fetch(start_date, end_date, start_val, cashflow, cashfreq, rolling,
                invest_div, rebalance, allocation, return_raw=False,
                include_raw=False, rebalance_offset=0, cashflow_offset=0, **kwargs):
    """Minimal mock for fetch_backtest that returns deterministic data."""
    dates = pd.bdate_range(start_date, end_date)
    vals = [start_val * (1 + 0.0003 * i) for i in range(len(dates))]
    series = pd.Series(vals, index=dates, name="Portfolio")
    stats = {"cagr": 0.08, "max_drawdown": -5.0, "sharpe": 1.2, "volatility": 0.15}
    extra = {"rebalancing_events": [], "rebalancing_stats": [], "daily_returns": []}
    if return_raw:
        return {"charts": {"history": [[int(d.timestamp()) for d in dates], vals]}, "stats": stats}
    return (series, stats, extra)


def _mock_shadow(allocation, start_val, start_date, end_date,
                 api_port_series=None, rebalance_freq="Yearly",
                 cashflow=0.0, cashflow_freq="Monthly", prices_df=None,
                 rebalance_month=1, rebalance_day=1, custom_freq="Yearly",
                 invest_dividends=True, pay_down_margin=False,
                 tax_config=None, custom_rebal_config=None, **kwargs):
    """Minimal mock for run_shadow_backtest."""
    dates = pd.bdate_range(start_date, end_date)
    vals = [start_val * (1 + 0.0003 * i) for i in range(len(dates))]
    series = pd.Series(vals, index=dates, name="Portfolio")
    twr = pd.Series([1 + 0.0003 * i for i in range(len(dates))], index=dates, name="TWR")
    return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], series, twr)


def _mock_component_prices(tickers, start_date, end_date):
    """Return deterministic component prices for the requested tickers/date range."""
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for idx, ticker in enumerate(dict.fromkeys(ticker.split("?")[0] for ticker in tickers), start=1):
        data[ticker] = [100.0 + idx + day for day in range(len(dates))]
    return pd.DataFrame(data, index=dates)


class TestRunSingleBacktest:
    def test_returns_expected_keys(self):
        """Result dict has all required keys."""
        result = run_single_backtest(
            allocation={"SPY": 60.0, "BND": 40.0},
            maint_pcts={"SPY": 25.0, "BND": 25.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2020-01-02",
            end_date="2020-12-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="Test",
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        expected_keys = {
            "name", "series", "port_series", "stats", "twr_series",
            "trades_df", "pl_by_year", "composition_df", "unrealized_pl_df",
            "component_prices", "allocation", "logs", "raw_response",
            "start_val", "sim_range", "shadow_range", "wmaint",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_single_asset_dca_forces_local_engine(self, monkeypatch):
        """Targeted DCA cannot be represented by the external API cashflow model."""
        captured = {}

        def _component_fetch(tickers, start_date, end_date, **kwargs):
            return _mock_component_prices(tickers, start_date, end_date)

        def _fetch_should_not_run(**kwargs):
            raise AssertionError("targeted DCA should use the local engine")

        def _shadow_records_dca(
            allocation,
            start_val,
            start_date,
            end_date,
            api_port_series=None,
            rebalance_freq="Yearly",
            cashflow=0.0,
            cashflow_freq="Monthly",
            prices_df=None,
            rebalance_month=1,
            rebalance_day=1,
            custom_freq="Yearly",
            invest_dividends=True,
            pay_down_margin=False,
            tax_config=None,
            custom_rebal_config=None,
            **kwargs,
        ):
            captured["dca_mode"] = kwargs.get("dca_mode")
            captured["dca_target_ticker"] = kwargs.get("dca_target_ticker")
            return _mock_shadow(
                allocation,
                start_val,
                start_date,
                end_date,
                api_port_series=api_port_series,
                rebalance_freq=rebalance_freq,
                cashflow=cashflow,
                cashflow_freq=cashflow_freq,
                prices_df=prices_df,
                rebalance_month=rebalance_month,
                rebalance_day=rebalance_day,
                custom_freq=custom_freq,
                invest_dividends=invest_dividends,
                pay_down_margin=pay_down_margin,
                tax_config=tax_config,
                custom_rebal_config=custom_rebal_config,
                **kwargs,
            )

        monkeypatch.setattr(orchestrator, "fetch_component_data", _component_fetch)

        result = run_single_backtest(
            allocation={"SPY": 60.0, "BND": 40.0},
            maint_pcts={"SPY": 25.0, "BND": 25.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2020-01-02",
            end_date="2020-01-10",
            start_val=10000.0,
            cashflow_amount=1000.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="Targeted DCA",
            fetch_backtest_fn=_fetch_should_not_run,
            run_shadow_fn=_shadow_records_dca,
            dca_config={"mode": "Single Asset", "target_ticker": "SPY"},
        )

        assert result["is_local"]
        assert captured == {"dca_mode": "Single Asset", "dca_target_ticker": "SPY"}

    def test_qqupsim_forces_local_engine(self, monkeypatch):
        def _component_fetch(tickers, start_date, end_date, **kwargs):
            return _mock_component_prices(tickers, start_date, end_date)

        def _fetch_should_not_run(**kwargs):
            raise AssertionError("QQUPSIM is a local realized-tracker simulation")

        monkeypatch.setattr(orchestrator, "fetch_component_data", _component_fetch)
        result = run_single_backtest(
            allocation={"QQUPSIM": 100.0},
            maint_pcts={"QQUPSIM": 50.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2020-01-02",
            end_date="2020-01-10",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="QQUPSIM",
            fetch_backtest_fn=_fetch_should_not_run,
            run_shadow_fn=_mock_shadow,
        )

        assert result["is_local"]

    def test_name_propagated(self):
        """Portfolio name is propagated to result."""
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-06-30",
            start_val=5000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="My Portfolio",
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        assert result["name"] == "My Portfolio"

    def test_wmaint_calculation(self):
        """Weighted maintenance is calculated from allocation."""
        result = run_single_backtest(
            allocation={"SPY": 50.0, "TQQQ": 50.0},
            maint_pcts={"SPY": 25.0, "TQQQ": 75.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-03-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        # 50% * 25% + 50% * 75% = 50%
        assert abs(result["wmaint"] - 0.50) < 0.01

    def test_stats_populated(self):
        """Stats dict has standard fields from mock."""
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-06-30",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        assert "cagr" in result["stats"]
        assert "sharpe" in result["stats"]


class TestRunMultiBacktest:
    def test_reuses_shared_component_prices_in_pass1(self, monkeypatch):
        """Pass 1 fetches one shared component universe instead of per-portfolio fetches."""
        fetch_calls = []

        def _recording_component_fetch(tickers, start_date, end_date):
            fetch_calls.append((tuple(tickers), start_date, end_date))
            return _mock_component_prices(tickers, start_date, end_date)

        monkeypatch.setattr(orchestrator, "fetch_component_data", _recording_component_fetch)

        portfolios = [
            {
                "name": "First",
                "allocation": {"AAA": 50.0, "BBB": 50.0},
                "maint_pcts": {"AAA": 25.0, "BBB": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            },
            {
                "name": "Second",
                "allocation": {"BBB": 40.0, "CCC": 60.0},
                "maint_pcts": {"BBB": 25.0, "CCC": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            },
        ]

        results, bench_series = run_multi_backtest(
            portfolios=portfolios,
            start_date="2020-01-02",
            end_date="2020-12-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )

        assert [result["name"] for result in results] == ["First", "Second"]
        assert bench_series == []
        assert len(fetch_calls) == 1
        assert set(fetch_calls[0][0]) == {"AAA", "BBB", "CCC"}

    def test_effective_start_excludes_retained_prior_day_anchor(self, monkeypatch):
        """A return anchor may precede the requested start but is not the common start."""

        def _component_fetch(tickers, start_date, end_date, **kwargs):
            return _mock_component_prices(tickers, start_date, end_date)

        def _shadow_with_anchor(
            allocation,
            start_val,
            start_date,
            end_date,
            prices_df=None,
            **kwargs,
        ):
            requested_start = pd.Timestamp(start_date)
            prior = prices_df[prices_df.index < requested_start].tail(1).index
            in_window = pd.bdate_range(requested_start, end_date)
            dates = prior.append(in_window)
            values = [start_val * (1 + 0.001 * i) for i in range(len(dates))]
            series = pd.Series(values, index=dates, name="Portfolio")
            twr = series / series.iloc[0]
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                [],
                series,
                twr,
            )

        monkeypatch.setattr(orchestrator, "fetch_component_data", _component_fetch)

        results, _ = run_multi_backtest(
            portfolios=[{
                "name": "Anchored",
                "allocation": {"AAA": 100.0},
                "maint_pcts": {"AAA": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            }],
            start_date="2004-01-01",
            end_date="2004-03-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_shadow_with_anchor,
        )

        assert results[0]["series"].index.min() == pd.Timestamp("2003-12-31")
        assert results[0]["effective_start_date"] == pd.Timestamp("2004-01-01")

    def test_reuses_shared_component_prices_for_local_pass2_reruns(self, monkeypatch):
        """Aligned local reruns share one common-start fetch instead of refetching per portfolio."""
        fetch_calls = []

        def _recording_component_fetch(tickers, start_date, end_date):
            fetch_calls.append((tuple(tickers), start_date, end_date))
            return _mock_component_prices(tickers, start_date, end_date)

        def _shadow_with_late_portfolio(
            allocation,
            start_val,
            start_date,
            end_date,
            api_port_series=None,
            rebalance_freq="Yearly",
            cashflow=0.0,
            cashflow_freq="Monthly",
            prices_df=None,
            rebalance_month=1,
            rebalance_day=1,
            custom_freq="Yearly",
            invest_dividends=True,
            pay_down_margin=False,
            tax_config=None,
            custom_rebal_config=None,
            **kwargs,
        ):
            dates = pd.bdate_range(start_date, end_date)
            if "LATE" in allocation:
                dates = dates[5:]
            vals = [start_val * (1 + 0.0003 * i) for i in range(len(dates))]
            series = pd.Series(vals, index=dates, name="Portfolio")
            twr = pd.Series([1 + 0.0003 * i for i in range(len(dates))], index=dates, name="TWR")
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], series, twr)

        monkeypatch.setattr(orchestrator, "fetch_component_data", _recording_component_fetch)

        portfolios = [
            {
                "name": "Early A",
                "allocation": {"AAA": 100.0},
                "maint_pcts": {"AAA": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            },
            {
                "name": "Early B",
                "allocation": {"BBB": 100.0},
                "maint_pcts": {"BBB": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            },
            {
                "name": "Late",
                "allocation": {"LATE": 100.0},
                "maint_pcts": {"LATE": 25.0},
                "rebalance": {"mode": RebalMode.NONE, "freq": Freq.YEARLY},
            },
        ]

        results, _ = run_multi_backtest(
            portfolios=portfolios,
            start_date="2020-01-02",
            end_date="2020-12-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_shadow_with_late_portfolio,
        )

        assert len(fetch_calls) == 2
        assert set(fetch_calls[0][0]) == {"AAA", "BBB", "LATE"}
        assert set(fetch_calls[1][0]) == {"AAA", "BBB"}
        common_start_str = results[2]["series"].index[0].strftime("%Y-%m-%d")
        assert fetch_calls[1][1] == orchestrator._return_anchor_start(common_start_str)
        assert results[0]["shadow_range"].startswith(common_start_str)
        assert results[1]["shadow_range"].startswith(common_start_str)

    def test_local_cashflow_result_keeps_money_weighted_series(self, monkeypatch):
        """Local portfolios must chart deposit-inclusive value, not rebased TWR."""

        def _recording_component_fetch(tickers, start_date, end_date, **kwargs):
            return _mock_component_prices(tickers, start_date, end_date)

        def _shadow_with_dca_series(
            allocation,
            start_val,
            start_date,
            end_date,
            api_port_series=None,
            rebalance_freq="Yearly",
            cashflow=0.0,
            cashflow_freq="Monthly",
            prices_df=None,
            rebalance_month=1,
            rebalance_day=1,
            custom_freq="Yearly",
            invest_dividends=True,
            pay_down_margin=False,
            tax_config=None,
            custom_rebal_config=None,
            **kwargs,
        ):
            dates = pd.bdate_range(start_date, end_date)
            twr = pd.Series(1.0, index=dates, name="TWR")
            deposits = pd.Series(0.0, index=dates)
            deposits.iloc[2:] = cashflow
            series = pd.Series(start_val, index=dates, name="Portfolio") + deposits
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], series, twr)

        monkeypatch.setattr(orchestrator, "fetch_component_data", _recording_component_fetch)

        portfolios = [
            {
                "name": "Local NDX",
                "allocation": {"NDXMEGASIM?L=2": 100.0},
                "maint_pcts": {"NDXMEGASIM": 50.0},
                "rebalance": {"mode": RebalMode.STANDARD, "freq": Freq.YEARLY},
            }
        ]

        results, _ = run_multi_backtest(
            portfolios=portfolios,
            start_date="2020-01-02",
            end_date="2020-01-10",
            start_val=100000.0,
            cashflow_amount=5000.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_shadow_with_dca_series,
        )

        result = results[0]
        assert result["is_local"]
        assert result["series"].iloc[-1] == pytest.approx(105000.0)
        assert (result["twr_series"] * 100000.0).iloc[-1] == pytest.approx(100000.0)
