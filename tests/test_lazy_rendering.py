"""Regression tests for lazy-rendered UI panels."""

from __future__ import annotations

from contextlib import nullcontext

import pandas as pd
import pytest

import app.services.data_service as data_service
import app.ui.charts.metrics as metrics_chart
import app.ui.charts.returns as returns_chart
import app.ui.charts.rolling as rolling_chart
import app.ui.results.tabs_chart as tabs_chart


class _DummyColumn:
    def metric(self, *args, **kwargs):
        return None


def _noop(*args, **kwargs):
    return None


def _unexpected(name: str):
    def _raiser(*args, **kwargs):
        raise AssertionError(f"{name} should not run for the selected view")

    return _raiser


def _sample_series() -> pd.Series:
    dates = pd.bdate_range("2023-01-02", periods=6)
    return pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 105.0], index=dates)


def test_monthly_timeline_state_clips_to_selected_month():
    dates = pd.to_datetime(["2023-01-03", "2023-01-31", "2023-02-28", "2023-03-31"])
    series = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)

    visible_series, monthly_ret, options, selected_end = returns_chart._monthly_timeline_state(
        series,
        pd.Timestamp("2023-02-28"),
    )

    assert selected_end == pd.Timestamp("2023-02-28")
    assert options == [
        pd.Timestamp("2023-01-31"),
        pd.Timestamp("2023-02-28"),
        pd.Timestamp("2023-03-31"),
    ]
    assert visible_series.index.max() == pd.Timestamp("2023-02-28")
    assert list(monthly_ret.index) == [pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-28")]
    assert monthly_ret.iloc[0] == pytest.approx(0.10)
    assert monthly_ret.iloc[1] == pytest.approx(0.10)


def test_monthly_timeline_state_hides_prior_year_anchor_without_losing_return():
    dates = pd.to_datetime(["2003-12-31", "2004-01-30", "2004-02-27"])
    series = pd.Series([100.0, 110.0, 121.0], index=dates)

    visible_series, monthly_ret, options, selected_end = returns_chart._monthly_timeline_state(
        series,
        period_start=pd.Timestamp("2004-01-01"),
    )

    assert visible_series.index.min() == pd.Timestamp("2003-12-31")
    assert selected_end == pd.Timestamp("2004-02-29")
    assert options == [
        pd.Timestamp("2004-01-31"),
        pd.Timestamp("2004-02-29"),
    ]
    assert list(monthly_ret.index) == options
    assert monthly_ret.iloc[0] == pytest.approx(0.10)
    assert monthly_ret.iloc[1] == pytest.approx(0.10)


def test_quarterly_timeline_state_clips_to_selected_quarter():
    dates = pd.to_datetime(["2023-01-03", "2023-03-31", "2023-06-30", "2023-09-29"])
    series = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)

    visible_series, quarterly_ret, options, selected_end = returns_chart._quarterly_timeline_state(
        series,
        pd.Timestamp("2023-06-30"),
    )

    assert selected_end == pd.Timestamp("2023-06-30")
    assert options == [
        pd.Timestamp("2023-03-31"),
        pd.Timestamp("2023-06-30"),
        pd.Timestamp("2023-09-30"),
    ]
    assert visible_series.index.max() == pd.Timestamp("2023-06-30")
    assert list(quarterly_ret.index) == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert quarterly_ret.iloc[0] == pytest.approx(0.10)
    assert quarterly_ret.iloc[1] == pytest.approx(0.10)


def test_chart_tab_only_renders_selected_panel(monkeypatch):
    series = _sample_series()
    component_prices = pd.DataFrame({"SPY": series}, index=series.index)

    monkeypatch.setattr(tabs_chart.st, "segmented_control", lambda *args, **kwargs: "📉 200DMA")
    monkeypatch.setattr(tabs_chart.st, "tabs", lambda labels: [nullcontext() for _ in labels])

    calls = []

    def record_ma(*args, **kwargs):
        if kwargs.get("window") != 200:
            raise AssertionError("Only the selected 200DMA panel should render")
        calls.append(kwargs["window"])

    monkeypatch.setattr(tabs_chart.charts, "render_ma_analysis_tab", record_ma)
    monkeypatch.setattr(tabs_chart.charts, "render_classic_chart", _unexpected("render_classic_chart"))
    monkeypatch.setattr(tabs_chart.charts, "render_candlestick_chart", _unexpected("render_candlestick_chart"))
    monkeypatch.setattr(tabs_chart.charts, "render_munger_wma_tab", _unexpected("render_munger_wma_tab"))
    monkeypatch.setattr(tabs_chart.charts, "render_cheat_sheet", _unexpected("render_cheat_sheet"))

    tabs_chart.render_chart_tab(
        nullcontext(),
        chart_style="Classic (Combined)",
        tax_adj_port_series=series,
        final_adj_series=series,
        loan_series=series * 0,
        tax_adj_equity_pct_series=series * 0 + 0.5,
        tax_adj_usage_series=series * 0,
        equity_series=series,
        usage_series=series * 0,
        equity_pct_series=series * 0 + 0.5,
        effective_rate_series=series * 0,
        ohlc_data=pd.DataFrame(),
        equity_resampled=series,
        loan_resampled=series * 0,
        usage_resampled=series * 0,
        equity_pct_resampled=series * 0 + 0.5,
        effective_rate_resampled=series * 0,
        bench_resampled=None,
        comp_resampled=None,
        port_series=series,
        component_prices=component_prices,
        portfolio_name="Lazy Chart",
        log_scale=False,
        show_range_slider=False,
        show_volume=False,
        timeframe="Daily",
        wmaint=0.25,
        stats={},
        config={},
        pay_tax_cash=False,
        draw_monthly=0.0,
        draw_monthly_retirement=0.0,
        draw_start_date=None,
        retirement_date=None,
        logs=[],
        final_tax_series=series * 0,
        tax_payment_series=series * 0,
        start_val=10000.0,
        rate_annual=0.0,
        pm_enabled=False,
    )

    assert calls == [200]


def test_returns_analysis_only_renders_selected_section(monkeypatch):
    series = _sample_series()

    monkeypatch.setattr(returns_chart.st, "segmented_control", lambda *args, **kwargs: "📊 Daily")
    monkeypatch.setattr(returns_chart.st, "tabs", lambda labels: [nullcontext() for _ in labels])
    monkeypatch.setattr(returns_chart.st, "subheader", _noop)
    monkeypatch.setattr(returns_chart.st, "dataframe", _noop)
    monkeypatch.setattr(returns_chart.st, "plotly_chart", _noop)
    monkeypatch.setattr(returns_chart.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(returns_chart.st, "columns", lambda n: [_DummyColumn() for _ in range(n)])

    monkeypatch.setattr(rolling_chart, "render_rolling_metrics", _unexpected("render_rolling_metrics"))
    monkeypatch.setattr(metrics_chart, "render_risk_return_metrics", _unexpected("render_risk_return_metrics"))
    monkeypatch.setattr(data_service, "fetch_component_data", _unexpected("fetch_component_data"))

    returns_chart.render_returns_analysis(
        series,
        unique_id="lazy-returns",
        portfolio_name="Lazy Returns",
        stats={},
        raw_response={},
    )


def test_monthly_returns_view_timeline_slider_limits_visible_period(monkeypatch):
    dates = pd.to_datetime(["2023-01-03", "2023-01-31", "2023-02-28", "2023-03-31"])
    series = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)
    captured_slider = {}
    captured_frames = []

    def select_first_month(label, options, value, **kwargs):
        captured_slider["label"] = label
        captured_slider["options"] = options
        captured_slider["default"] = value
        return options[0]

    def capture_dataframe(obj, **kwargs):
        captured_frames.append(getattr(obj, "data", obj).copy())

    monkeypatch.setattr(returns_chart.st, "segmented_control", lambda *args, **kwargs: "🗓️ Monthly")
    monkeypatch.setattr(returns_chart.st, "select_slider", select_first_month)
    monkeypatch.setattr(returns_chart.st, "subheader", _noop)
    monkeypatch.setattr(returns_chart.st, "caption", _noop)
    monkeypatch.setattr(returns_chart.st, "dataframe", capture_dataframe)
    monkeypatch.setattr(returns_chart.st, "plotly_chart", _noop)
    monkeypatch.setattr(returns_chart.st, "toggle", lambda *args, **kwargs: False)

    returns_chart.render_returns_analysis(
        series,
        unique_id="monthly-timeline",
        portfolio_name="Monthly Timeline",
        stats={},
        raw_response={},
    )

    assert captured_slider["label"] == "Timeline"
    assert captured_slider["default"] == pd.Timestamp("2023-03-31")
    assert captured_slider["options"] == [
        pd.Timestamp("2023-01-31"),
        pd.Timestamp("2023-02-28"),
        pd.Timestamp("2023-03-31"),
    ]
    assert captured_frames[0]["Date"].tolist() == ["2023-01"]


def test_quarterly_returns_view_timeline_slider_limits_visible_period(monkeypatch):
    dates = pd.to_datetime(["2023-01-03", "2023-03-31", "2023-06-30", "2023-09-29"])
    series = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)
    captured_slider = {}
    captured_frames = []

    def select_first_quarter(label, options, value, **kwargs):
        captured_slider["label"] = label
        captured_slider["options"] = options
        captured_slider["default"] = value
        return options[0]

    def capture_dataframe(obj, **kwargs):
        captured_frames.append(getattr(obj, "data", obj).copy())

    monkeypatch.setattr(returns_chart.st, "segmented_control", lambda *args, **kwargs: "📆 Quarterly")
    monkeypatch.setattr(returns_chart.st, "select_slider", select_first_quarter)
    monkeypatch.setattr(returns_chart.st, "subheader", _noop)
    monkeypatch.setattr(returns_chart.st, "caption", _noop)
    monkeypatch.setattr(returns_chart.st, "dataframe", capture_dataframe)
    monkeypatch.setattr(returns_chart.st, "plotly_chart", _noop)
    monkeypatch.setattr(returns_chart.st, "toggle", lambda *args, **kwargs: False)

    returns_chart.render_returns_analysis(
        series,
        unique_id="quarterly-timeline",
        portfolio_name="Quarterly Timeline",
        stats={},
        raw_response={},
    )

    assert captured_slider["label"] == "Timeline"
    assert captured_slider["default"] == pd.Timestamp("2023-09-30")
    assert captured_slider["options"] == [
        pd.Timestamp("2023-03-31"),
        pd.Timestamp("2023-06-30"),
        pd.Timestamp("2023-09-30"),
    ]
    assert captured_frames[0]["Period"].tolist() == ["2023Q1"]
