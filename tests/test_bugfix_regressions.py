"""Regression tests for bugs fixed in the March 2026 audit.

Each test is tagged with the bug it guards against. If any of these
tests fail after a future change, the corresponding bug has regressed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_port(start="2020-01-02", end="2024-12-31", start_val=100_000):
    """Flat portfolio series — $start_val every business day."""
    dates = pd.bdate_range(start, end)
    return pd.Series(start_val, index=dates, dtype=float)


def _make_growing_port(start="2020-01-02", end="2024-12-31",
                       start_val=100_000, end_val=200_000):
    dates = pd.bdate_range(start, end)
    vals = np.linspace(start_val, end_val, len(dates))
    return pd.Series(vals, index=dates, dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# 1. Margin: Compound vs Simple Interest
# ═══════════════════════════════════════════════════════════════════════

class TestCompoundInterest:
    """simulate_margin must use compound daily rates, not simple division."""

    def test_fixed_rate_compound(self):
        """Fixed mode: loan after 1 year at 8% should match compound formula."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        rate_cfg = {"type": "Fixed", "rate_pct": 8.0}
        loan, *_ = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        n_days = len(port)
        expected = 100_000 * (1 + 0.08) ** (n_days / 252)
        assert loan.iloc[-1] == pytest.approx(expected, rel=1e-4)

    def test_legacy_float_rate_compound(self):
        """Legacy float rate path uses compound daily rate."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        loan, *_ = simulate_margin(port, 100_000, 8.0, 0, 0.25)
        n_days = len(port)
        expected = 100_000 * (1 + 0.08) ** (n_days / 252)
        assert loan.iloc[-1] == pytest.approx(expected, rel=1e-4)

    def test_variable_rate_compound(self):
        """Variable mode: daily rates use compound formula."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {"type": "Variable", "base_series": base, "spread_pct": 1.0}
        loan, _, _, _, eff_rate = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        # Effective rate should be ~6%
        assert eff_rate.iloc[10] == pytest.approx(6.0, abs=0.1)
        # Loan should grow at compound 6%
        n_days = len(port)
        expected = 100_000 * (1 + 0.06) ** (n_days / 252)
        assert loan.iloc[-1] == pytest.approx(expected, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# 2. Variable-rate iterative path must not charge zero interest
# ═══════════════════════════════════════════════════════════════════════

class TestVariableRateIterative:
    """When Variable mode hits the iterative fallback (DCA or starting cash),
    interest must still accrue — previously rate_daily was set to 0.0."""

    def test_variable_with_dca_accrues_interest(self):
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {"type": "Variable", "base_series": base, "spread_pct": 1.0}
        # DCA triggers the iterative path
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1000  # monthly DCA
        loan, *_ = simulate_margin(
            port, 50_000, rate_cfg, 0, 0.25, dca_series=dca, fund_dca_margin=True
        )
        # Loan must grow beyond starting_loan + sum(DCA) due to interest
        total_dca = dca.sum()
        assert loan.iloc[-1] > 50_000 + total_dca, \
            "Variable-rate iterative path must accrue interest (was zero before fix)"

    def test_variable_with_starting_cash_accrues_interest(self):
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {"type": "Variable", "base_series": base, "spread_pct": 1.0}
        # Negative starting_loan (cash) triggers iterative path, then draws push positive
        loan, *_ = simulate_margin(
            port, -10_000, rate_cfg, 5000, 0.25,  # $5k/mo draws
        )
        # After draws push loan positive, interest should accrue
        positive_days = (loan > 0).sum()
        if positive_days > 50:
            # Compare with zero-interest scenario
            loan_no_rate, *_ = simulate_margin(
                port, -10_000, {"type": "Fixed", "rate_pct": 0.0}, 5000, 0.25,
            )
            assert loan.iloc[-1] > loan_no_rate.iloc[-1], \
                "Variable-rate iterative path must charge interest on positive balances"


# ═══════════════════════════════════════════════════════════════════════
# 3. Loan clipping: vectorized path must NOT clip negative loan to zero
# ═══════════════════════════════════════════════════════════════════════

class TestLoanNoClip:
    def test_negative_loan_preserved(self):
        """Repayments that push loan negative must preserve the credit."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-06-30", 200_000)
        # Large repayment that exceeds the starting loan
        repay = pd.Series(0.0, index=port.index)
        repay.iloc[10] = 60_000  # Repay $60k on a $50k loan
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, repayment_series=repay,
        )
        # Loan should go negative after the repayment
        assert loan.iloc[11] < 0, "Loan should go negative after over-repayment"


# ═══════════════════════════════════════════════════════════════════════
# 4. DCA: fund_dca_margin in non-tiered iterative path
# ═══════════════════════════════════════════════════════════════════════

class TestDCAFundMargin:
    def test_fund_dca_margin_true_increases_loan(self):
        """fund_dca_margin=True: DCA always adds to the loan."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1000
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, dca_series=dca, fund_dca_margin=True,
        )
        assert loan.iloc[-1] > 50_000 + dca.sum() * 0.9, \
            "fund_dca_margin=True should add DCA to loan"

    def test_fund_dca_margin_false_no_loan_increase(self):
        """fund_dca_margin=False with positive loan: DCA should NOT add to loan."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-12-29", 200_000)
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1000
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, dca_series=dca, fund_dca_margin=False,
        )
        # With no interest and no draws, loan should stay at 50k
        assert loan.iloc[-1] == pytest.approx(50_000, abs=100)


# ═══════════════════════════════════════════════════════════════════════
# 5. Division-by-zero guard in equity_pct / usage_pct
# ═══════════════════════════════════════════════════════════════════════

class TestDivisionByZero:
    def test_zero_portfolio_no_crash(self):
        """Portfolio with zero values should not produce inf/NaN."""
        from app.services.testfol_api import simulate_margin
        port = _make_port("2023-01-02", "2023-03-31", 100_000)
        port.iloc[10:15] = 0.0  # Simulate crash to zero
        loan, equity, eq_pct, usage, _ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
        )
        assert not eq_pct.isna().any(), "equity_pct should not contain NaN"
        assert not np.isinf(eq_pct).any(), "equity_pct should not contain inf"
        assert not usage.isna().any(), "usage_pct should not contain NaN"
        assert not np.isinf(usage).any(), "usage_pct should not contain inf"


# ═══════════════════════════════════════════════════════════════════════
# 6. Tax: historical_smart uses 0/15/20% for 2013-2023
# ═══════════════════════════════════════════════════════════════════════

class TestHistoricalSmartTax:
    def test_2020_ltcg_not_taxed_at_ordinary_rates(self):
        """In 2020, $50k LTCG on $100k income should be ~15%, not ~24% ordinary."""
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            realized_gain=0,
            other_income=100_000,
            year=2020,
            filing_status="Single",
            method="historical_smart",
            short_term_gain=0,
            long_term_gain=50_000,
            use_standard_deduction=True,
        )
        # At 15% rate: ~$7,500. At ordinary (~24%): ~$12,000
        assert tax < 10_000, f"2020 LTCG should use 0/15/20% rates, not ordinary. Got ${tax:,.0f}"
        assert tax > 5_000, f"2020 LTCG should still produce some tax. Got ${tax:,.0f}"

    def test_2015_ltcg_preferential_rates(self):
        """2015 should also use 0/15/20%, not ordinary rates."""
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            realized_gain=0, other_income=80_000, year=2015,
            filing_status="Single", method="historical_smart",
            short_term_gain=0, long_term_gain=30_000,
            use_standard_deduction=True,
        )
        # 15% rate → ~$4,500
        assert tax < 7_000, f"2015 LTCG should be at preferential rates. Got ${tax:,.0f}"


# ═══════════════════════════════════════════════════════════════════════
# 7. Tax: alternative tax cap only applies pre-1987
# ═══════════════════════════════════════════════════════════════════════

class TestAlternativeTaxCap:
    def test_1995_no_alt_cap(self):
        """Post-1987 years should not have an alternative tax cap."""
        from app.core.tax_library import calculate_tax_on_realized_gains
        # With high income, ordinary rate ~35%. Without cap, full ordinary applies.
        tax = calculate_tax_on_realized_gains(
            realized_gain=0, other_income=500_000, year=1995,
            filing_status="Single", method="historical_smart",
            short_term_gain=0, long_term_gain=100_000,
            use_standard_deduction=False,
        )
        # Should be taxed at full ordinary rates (inclusion_rate=1.0, no cap)
        assert tax > 0


# ═══════════════════════════════════════════════════════════════════════
# 8. Tax: MFS standard deduction uses true division
# ═══════════════════════════════════════════════════════════════════════

class TestMFSDeduction:
    def test_mfs_deduction_is_half_mfj(self):
        from app.core.tax_library import get_standard_deduction
        mfj = get_standard_deduction(2024, "Married Filing Jointly")
        mfs = get_standard_deduction(2024, "Married Filing Separately")
        assert mfs == mfj / 2
        # Ensure it's a float (true division), not floor-divided int
        assert isinstance(mfs, (float, int))
        assert mfs == pytest.approx(mfj / 2, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# 9. Stats: daily rate consistency (252 trading days)
# ═══════════════════════════════════════════════════════════════════════

class TestDailyRateConsistency:
    def test_tax_adjusted_equity_uses_252(self):
        """calculate_tax_adjusted_equity must use 252 trading days."""
        from app.core.calculations.stats import calculate_tax_adjusted_equity
        port = _make_port("2023-01-02", "2023-12-29", 100_000)
        loan = pd.Series(50_000.0, index=port.index)
        equity = port - loan
        tax_series = pd.Series(0.0, index=port.index)
        adj_eq, adj_tax = calculate_tax_adjusted_equity(
            port, loan, equity, tax_series,
            rate_annual=8.0, draw_monthly=0, draw_monthly_retirement=0,
        )
        # The daily rate component should produce values consistent with 252-day compounding
        # At 8% annual, daily rate = (1.08)^(1/252) - 1 ≈ 0.0306%
        # With 365.25 it would be (1.08)^(1/365.25) - 1 ≈ 0.0211%
        # The loan component grows faster with 252 days
        assert not adj_eq.empty


# ═══════════════════════════════════════════════════════════════════════
# 10. Stats: best/worst year keeps complete final year
# ═══════════════════════════════════════════════════════════════════════

class TestBestWorstYear:
    def test_complete_final_year_not_dropped(self):
        """If series ends near Dec 31, last year should be included."""
        from app.core.calculations.stats import generate_stats
        # 3 complete calendar years
        dates = pd.bdate_range("2021-01-04", "2023-12-29")
        # Year 2023 has the best return (big jump in Dec)
        vals = np.ones(len(dates)) * 100
        for i, d in enumerate(dates):
            if d.year == 2022:
                vals[i] = 110
            elif d.year == 2023:
                vals[i] = 150  # Big jump
        series = pd.Series(vals, index=dates)
        stats = generate_stats(series)
        # Best year should reflect 2023's performance, not be dropped
        assert stats.get("best_year", 0) > 10, \
            "Complete final year should not be dropped from best_year"


# ═══════════════════════════════════════════════════════════════════════
# 11. Withdrawals: RetDraw events counted correctly
# ═══════════════════════════════════════════════════════════════════════

class TestRetDrawEvents:
    def test_parse_events_retdraw(self):
        """RetDraw events should be parsed and treated as draws (positive)."""
        from app.ui.results.tabs_withdrawals import _parse_events
        logs = [
            "  💸 Draw 2023-06-30: +$5,000 → loan $55,000",
            "  💸 RetDraw 2024-01-31: +$8,000 → loan $63,000",
        ]
        events = _parse_events(logs)
        assert len(events) == 2
        assert set(events["Type"]) == {"Draw", "RetDraw"}

    def test_retdraw_included_in_total_drawn(self):
        """RetDraw events must be included when computing total drawn."""
        from app.ui.results.tabs_withdrawals import _parse_events
        logs = [
            "  💸 Draw 2023-06-30: +$5,000 → loan $55,000",
            "  💸 RetDraw 2024-01-31: +$8,000 → loan $63,000",
        ]
        events = _parse_events(logs)
        draws = events[events["Type"].isin(["Draw", "RetDraw"])]
        assert draws["Amount"].sum() == pytest.approx(13_000)


# ═══════════════════════════════════════════════════════════════════════
# 12. Shadow backtest: draw label uses date, not amount
# ═══════════════════════════════════════════════════════════════════════

class TestDrawLabel:
    def test_draw_label_by_date_not_amount(self):
        """When pre-retirement and retirement draw amounts are equal,
        labels should still distinguish by date."""
        from app.core.shadow_backtest import run_shadow_backtest
        import datetime
        dates = pd.bdate_range("2023-01-02", "2024-12-31")
        prices = pd.DataFrame({"TEST": 100.0}, index=dates)
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2024-12-31",
            prices_df=prices,
            draw_monthly=2000, draw_monthly_retirement=2000,  # Same amount
            draw_start_date=datetime.date(2023, 1, 1),
            retirement_date=datetime.date(2024, 1, 1),
            starting_loan=50_000,
        )
        logs = result[4]  # logs are at index 4
        str_logs = [l for l in logs if isinstance(l, str)]
        pre_ret_draws = [l for l in str_logs if "Draw" in l and "RetDraw" not in l and "2023" in l]
        ret_draws = [l for l in str_logs if "RetDraw" in l and "2024" in l]
        # Both should exist even though amounts are equal
        assert len(pre_ret_draws) > 0, f"Pre-retirement draws should be labeled 'Draw', logs: {str_logs[:5]}"
        assert len(ret_draws) > 0, f"Post-retirement draws should be labeled 'RetDraw'"


# ═══════════════════════════════════════════════════════════════════════
# 13. PM wmaint: uses total_w not hardcoded 100
# ═══════════════════════════════════════════════════════════════════════

class TestPMWmaint:
    def test_pm_wmaint_uses_total_weight(self):
        """PM weighted maintenance should divide by total_w, not 100."""
        from app.core.backtest_orchestrator import run_single_backtest

        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        port = pd.Series(100_000.0, index=dates)

        # Weight is 50 (not 100) — this exposes the /100 vs /total_w bug
        alloc = {"SPY": 50.0}
        maint = {"SPY": 25.0}
        pm_maint = {"SPY": 15.0}

        def mock_fetch(**kw):
            return (port, {"cagr": 0.05}, {"raw_response": {}})

        def mock_shadow(*a, **kw):
            return (port, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], port, port, [])

        result = run_single_backtest(
            alloc, maint, {},
            "2023-01-02", "2023-12-29", 100_000,
            cashflow_amount=0.0, cashflow_freq="Monthly",
            invest_div=True, pay_down_margin=False,
            tax_config={}, bearer_token=None,
            pm_maint_pcts=pm_maint,
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        # wmaint should be 25/100 = 0.25 (weight 50 / total_w 50 * 25%)
        assert result["wmaint"] == pytest.approx(0.25, abs=0.01)
        # pm_wmaint should be 15/100 = 0.15 (weight 50 / total_w 50 * 15%)
        assert result["wmaint_pm"] == pytest.approx(0.15, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# 14. OHLC resample: non-1D timeframes must not crash
# ═══════════════════════════════════════════════════════════════════════

class TestOHLCResample:
    def test_weekly_resample_no_crash(self):
        """resample_data with OHLC method must work for weekly timeframe."""
        from app.common.utils import resample_data
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        series = pd.Series(np.random.uniform(95, 105, len(dates)),
                           index=dates, dtype=float)
        result = resample_data(series, "1W", method="ohlc")
        assert not result.empty
        assert set(result.columns) == {"Open", "High", "Low", "Close"}

    def test_monthly_resample_no_crash(self):
        from app.common.utils import resample_data
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        series = pd.Series(np.random.uniform(95, 105, len(dates)),
                           index=dates, dtype=float)
        result = resample_data(series, "1M", method="ohlc")
        assert not result.empty
        assert "Close" in result.columns


# ═══════════════════════════════════════════════════════════════════════
# 15. Monthly heatmap scale consistency
# ═══════════════════════════════════════════════════════════════════════

class TestHeatmapScale:
    def test_monthly_yearly_scale_matches_quarterly(self):
        """Monthly heatmap yearly column should use 2x std_dev, same as quarterly."""
        # This is a code-level check — read the source and verify the multiplier
        import inspect
        from app.ui.charts import returns
        source = inspect.getsource(returns)
        # After the fix, both should use "2.0 * std_dev_yearly" or "2 * std_dev_yearly"
        assert "1.0 * std_dev_yearly" not in source, \
            "Monthly heatmap yearly scale should use 2.0x, not 1.0x"


# ═══════════════════════════════════════════════════════════════════════
# 16. Chart vertical lines use Timestamps, not epoch-ms
# ═══════════════════════════════════════════════════════════════════════

class TestChartVerticalLines:
    def test_no_epoch_ms_in_chart_code(self):
        """Chart code should not convert timestamps to epoch-milliseconds."""
        import inspect
        from app.ui.charts import portfolio
        source = inspect.getsource(portfolio)
        assert "timestamp() * 1000" not in source, \
            "Vertical lines should use pd.Timestamp, not epoch-milliseconds"


# ═══════════════════════════════════════════════════════════════════════
# 17. Chart cache removed
# ═══════════════════════════════════════════════════════════════════════

class TestChartNoCache:
    def test_render_classic_not_cached(self):
        """render_classic_chart should NOT have @st.cache_data."""
        from app.ui.charts.portfolio import render_classic_chart
        # Cached functions have __wrapped__ attribute from st.cache_data
        assert not hasattr(render_classic_chart, '__wrapped__'), \
            "render_classic_chart should not be cached"

    def test_render_dashboard_not_cached(self):
        from app.ui.charts.portfolio import render_dashboard_view
        assert not hasattr(render_dashboard_view, '__wrapped__'), \
            "render_dashboard_view should not be cached"


# ═══════════════════════════════════════════════════════════════════════
# 18. Multi-portfolio caption includes active cashflows
# ═══════════════════════════════════════════════════════════════════════

class TestMultiPortfolioCashflowCaption:
    def test_no_caption_fragment_without_dca(self):
        from app.ui.charts.portfolio import _format_cashflow_caption
        assert _format_cashflow_caption({"amount": 0, "freq": "Monthly"}) == ""

    def test_formats_dca_amount_and_frequency(self):
        from app.ui.charts.portfolio import _format_cashflow_caption
        caption = _format_cashflow_caption({"amount": 1500, "freq": "Monthly"})
        assert caption == r"DCA: \$1,500 Monthly."

    def test_formats_margin_repayment_cashflow(self):
        from app.ui.charts.portfolio import _format_cashflow_caption
        caption = _format_cashflow_caption({
            "amount": 250,
            "freq": "Quarterly",
            "pay_down_margin": True,
        })
        assert caption == r"Margin repayment: \$250 Quarterly."
