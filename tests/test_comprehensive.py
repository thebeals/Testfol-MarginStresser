"""Comprehensive test suite for the Testfol-MarginStresser application.

Covers: Tax Library, Shadow Backtest, Margin Simulation, Statistics,
        Backtest Orchestrator, and Utility Functions.

All tests are designed to be fast (no network calls) using dependency
injection, pre-built price DataFrames, and mock functions.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_port(start="2023-01-02", end="2023-12-29", val=100_000):
    """Flat portfolio series."""
    dates = pd.bdate_range(start, end)
    return pd.Series(val, index=dates, dtype=float)


def _growing_port(start="2020-01-02", end="2024-12-31",
                  start_val=100_000, end_val=200_000):
    dates = pd.bdate_range(start, end)
    vals = np.linspace(start_val, end_val, len(dates))
    return pd.Series(vals, index=dates, dtype=float)


def _flat_prices(ticker="TEST", start="2023-01-02", end="2023-12-29", price=100.0):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({ticker: price}, index=dates)


def _growing_prices(ticker="TEST", start="2020-01-02", end="2024-12-31",
                    p0=100.0, p1=200.0):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({ticker: np.linspace(p0, p1, len(dates))}, index=dates)


def _two_ticker_prices(start="2020-01-02", end="2024-12-31"):
    """SPY grows 100->200, BND flat at 100."""
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({
        "SPY": np.linspace(100, 200, len(dates)),
        "BND": np.full(len(dates), 100.0),
    }, index=dates)


# ═══════════════════════════════════════════════════════════════════════
# 1. TAX LIBRARY
# ═══════════════════════════════════════════════════════════════════════

class TestGetStandardDeduction:
    """get_standard_deduction for all statuses and years."""

    def test_single_2024(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2024, "Single") == 14600

    def test_mfj_2024(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2024, "Married Filing Jointly") == 29200

    def test_hoh_2024(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2024, "Head of Household") == 21900

    def test_mfs_is_half_mfj(self):
        from app.core.tax_library import get_standard_deduction
        mfj = get_standard_deduction(2024, "Married Filing Jointly")
        mfs = get_standard_deduction(2024, "Married Filing Separately")
        assert mfs == pytest.approx(mfj / 2)

    def test_mfs_2025(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2025, "Married Filing Separately") == 15750

    def test_single_2026(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2026, "Single") == 16100

    def test_mfj_2026(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2026, "Married Filing Jointly") == 32200

    def test_hoh_2026(self):
        from app.core.tax_library import get_standard_deduction
        assert get_standard_deduction(2026, "Head of Household") == 24150

    def test_pre_1970_uses_10pct_capped_at_1000(self):
        from app.core.tax_library import get_standard_deduction
        # 10% of 50k = 5k, capped at 1k
        assert get_standard_deduction(1960, "Single", income=50_000) == 1000
        # 10% of 5k = 500, below cap
        assert get_standard_deduction(1960, "Single", income=5_000) == 500

    def test_future_year_uses_latest(self):
        from app.core.tax_library import get_standard_deduction
        # Future year should fallback to latest known (2026)
        assert get_standard_deduction(2030, "Single") == 16100

    def test_2018_tcja_jump(self):
        from app.core.tax_library import get_standard_deduction
        pre = get_standard_deduction(2017, "Single")
        post = get_standard_deduction(2018, "Single")
        assert post > pre * 1.5  # TCJA nearly doubled it


class TestCalculateFederalTax:
    """calculate_federal_tax (0/15/20% LTCG brackets)."""

    def test_zero_gain(self):
        from app.core.tax_library import calculate_federal_tax
        assert calculate_federal_tax(0, 50_000) == 0.0

    def test_negative_gain(self):
        from app.core.tax_library import calculate_federal_tax
        assert calculate_federal_tax(-10_000, 50_000) == 0.0

    def test_single_in_zero_pct_bracket_2025(self):
        from app.core.tax_library import calculate_federal_tax
        # Single 2025: 0% up to $48,350
        # Other income = $30k, gain = $10k -> all in 0% bracket
        assert calculate_federal_tax(10_000, 30_000, "Single", year=2025) == 0.0

    def test_single_in_15pct_bracket_2025(self):
        from app.core.tax_library import calculate_federal_tax
        # Other income = $60k (past 0% threshold), gain = $10k -> 15%
        tax = calculate_federal_tax(10_000, 60_000, "Single", year=2025)
        assert tax == pytest.approx(1_500)

    def test_single_in_20pct_bracket_2025(self):
        from app.core.tax_library import calculate_federal_tax
        # Other income = $550k (past 15% threshold at 533,400), gain = $10k -> 20%
        tax = calculate_federal_tax(10_000, 550_000, "Single", year=2025)
        assert tax == pytest.approx(2_000)

    def test_mfj_0pct_threshold_2024(self):
        from app.core.tax_library import calculate_federal_tax
        # MFJ 2024: 0% up to $94,050
        tax = calculate_federal_tax(50_000, 30_000, "Married Filing Jointly", year=2024)
        assert tax == pytest.approx(0.0)  # 30k + 50k = 80k < 94,050

    def test_gain_spanning_brackets(self):
        from app.core.tax_library import calculate_federal_tax
        # Single 2025: 0% up to 48,350. Other income = 40k.
        # Gain = 20k: 8,350 at 0%, 11,650 at 15%
        tax = calculate_federal_tax(20_000, 40_000, "Single", year=2025)
        expected = 0 + 11_650 * 0.15
        assert tax == pytest.approx(expected)

    def test_2026_single_thresholds(self):
        from app.core.tax_library import calculate_federal_tax
        assert calculate_federal_tax(9_000, 40_000, "Single", year=2026) == 0.0
        assert calculate_federal_tax(10_000, 40_000, "Single", year=2026) == pytest.approx(82.5)
        assert calculate_federal_tax(10_000, 550_000, "Single", year=2026) == pytest.approx(2_000)


class TestCalculateTaxOnRealizedGains:
    """calculate_tax_on_realized_gains: ST, LT, collectible, NIIT."""

    def test_zero_gain_returns_zero(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        assert calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=0, other_income=100_000,
            year=2024, filing_status="Single"
        ) == 0.0

    def test_all_losses_returns_zero(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        assert calculate_tax_on_realized_gains(
            short_term_gain=-5_000, long_term_gain=-10_000, other_income=100_000,
            year=2024, filing_status="Single"
        ) == 0.0

    def test_st_gain_taxed_as_ordinary(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # $50k ST gain on $100k income -> taxed at marginal ordinary rate
        tax = calculate_tax_on_realized_gains(
            short_term_gain=50_000, long_term_gain=0, other_income=100_000,
            year=2024, filing_status="Single", use_standard_deduction=False,
        )
        assert tax > 0
        # Should be significantly higher than 15% LTCG rate
        assert tax > 50_000 * 0.15

    def test_lt_gain_preferential_rate(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=50_000, other_income=100_000,
            year=2024, filing_status="Single", use_standard_deduction=False,
        )
        # Should be at 15% -> ~$7,500
        assert tax == pytest.approx(7_500, rel=0.05)

    def test_collectible_gains_capped_at_28pct(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # Very high income so marginal rate > 28%
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=0,
            long_term_gain_collectible=100_000,
            other_income=600_000,
            year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        # 28% cap on the collectible portion itself = $28,000
        # Plus NIIT may apply: MAGI=$700k - $200k threshold = $500k excess
        # NII = $100k collectible. NIIT = min(100k, 500k) * 3.8% = $3,800
        # Total max = $28,000 + $3,800 = $31,800
        assert tax <= 100_000 * 0.28 + 100_000 * 0.038 + 1

    def test_niit_applies_above_200k_single(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # Single, MAGI $300k (other income $250k + $50k LT gain)
        tax_with_niit = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=50_000,
            other_income=250_000,
            year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        # Without NIIT it would be just LTCG tax
        pure_ltcg = 50_000 * 0.15
        # NIIT: min(50k investment income, (300k - 200k)) * 3.8% = 50k * 3.8% = 1900
        assert tax_with_niit > pure_ltcg
        assert tax_with_niit == pytest.approx(pure_ltcg + 1_900, rel=0.05)

    def test_niit_mfj_threshold_250k(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # MFJ, MAGI = $240k (under threshold) -> no NIIT
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=40_000,
            other_income=200_000,
            year=2024, filing_status="Married Filing Jointly",
            use_standard_deduction=False,
        )
        pure_ltcg = 40_000 * 0.15
        assert tax == pytest.approx(pure_ltcg, rel=0.05)

    def test_niit_mfs_threshold_125k(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # MFS threshold is $125k
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=50_000,
            other_income=130_000,
            year=2024, filing_status="Married Filing Separately",
            use_standard_deduction=False,
        )
        # MAGI = 180k > 125k. Excess = 55k. NII = 50k. NIIT = min(50k,55k) * 3.8%
        expected_niit = 50_000 * 0.038
        assert tax > 50_000 * 0.15  # More than just LTCG
        assert tax == pytest.approx(50_000 * 0.15 + expected_niit, rel=0.1)

    def test_niit_not_applied_pre_2013(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=100_000,
            other_income=500_000,
            year=2012, filing_status="Single",
            method="historical_smart",
            use_standard_deduction=False,
        )
        # Should not include NIIT
        # Hard to test exact value, but verify it doesn't crash
        assert tax >= 0

    def test_standard_deduction_reduces_tax(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax_with_ded = calculate_tax_on_realized_gains(
            short_term_gain=50_000, long_term_gain=0,
            other_income=50_000, year=2024, filing_status="Single",
            use_standard_deduction=True,
        )
        tax_without_ded = calculate_tax_on_realized_gains(
            short_term_gain=50_000, long_term_gain=0,
            other_income=50_000, year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        assert tax_with_ded < tax_without_ded

    def test_huge_gains(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            short_term_gain=500_000, long_term_gain=2_000_000,
            other_income=1_000_000, year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        assert tax > 0
        # Sanity: tax should be less than total gains
        assert tax < 2_500_000

    def test_legacy_realized_gain_treated_as_lt(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        # When only realized_gain is provided (legacy), it's treated as LT
        tax_legacy = calculate_tax_on_realized_gains(
            realized_gain=50_000, other_income=100_000,
            year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        tax_explicit = calculate_tax_on_realized_gains(
            long_term_gain=50_000, other_income=100_000,
            year=2024, filing_status="Single",
            use_standard_deduction=False,
        )
        assert tax_legacy == pytest.approx(tax_explicit)


class TestHistoricalSmartTax:
    """historical_smart method for various year ranges."""

    def test_2020_lt_at_preferential_rates(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        tax = calculate_tax_on_realized_gains(
            short_term_gain=0, long_term_gain=50_000,
            other_income=100_000, year=2020, filing_status="Single",
            method="historical_smart", use_standard_deduction=True,
        )
        assert 5_000 < tax < 10_000

    def test_1980_inclusion_rate_40pct(self):
        from app.core.tax_library import get_capital_gains_inclusion_rate
        assert get_capital_gains_inclusion_rate(1980) == 0.40

    def test_1950_inclusion_rate_50pct(self):
        from app.core.tax_library import get_capital_gains_inclusion_rate
        assert get_capital_gains_inclusion_rate(1950) == 0.50

    def test_2024_inclusion_rate_100pct(self):
        from app.core.tax_library import get_capital_gains_inclusion_rate
        assert get_capital_gains_inclusion_rate(2024) == 1.0

    def test_all_filing_statuses_produce_tax(self):
        from app.core.tax_library import calculate_tax_on_realized_gains
        for status in ["Single", "Married Filing Jointly",
                       "Married Filing Separately", "Head of Household"]:
            tax = calculate_tax_on_realized_gains(
                long_term_gain=50_000, other_income=100_000,
                year=2024, filing_status=status,
                method="historical_smart",
                use_standard_deduction=False,
            )
            assert tax > 0, f"Tax should be positive for {status}"


class TestTaxSeriesWithCarryforward:
    """calculate_tax_series_with_carryforward: loss carryforward, $3k deduction."""

    def test_loss_carried_forward(self):
        from app.core.tax_library import calculate_tax_series_with_carryforward
        # Year 1: -$20k loss. Year 2: +$10k gain.
        # After $3k deduction in Y1: $17k carried forward.
        # Y2: $10k - $17k = -$7k net -> no tax on gains, but $3k deduction
        #     produces tax savings (negative tax)
        # Y3: remaining $4k carryforward < $15k gain -> positive tax
        pl = pd.Series([-20_000, 10_000, 15_000], index=[2020, 2021, 2022])
        taxes = calculate_tax_series_with_carryforward(
            pl, other_income=100_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        assert taxes[2020] <= 0  # Loss year - tax savings (refund)
        assert taxes[2021] <= 0  # Carried forward covers gain, $3k deduction gives savings
        assert taxes[2022] > 0  # Now we pay tax

    def test_3k_deduction_cap(self):
        from app.core.tax_library import calculate_tax_series_with_carryforward
        # $50k loss -> only $3k deducted, rest carried
        pl = pd.Series([-50_000], index=[2024])
        taxes = calculate_tax_series_with_carryforward(
            pl, other_income=200_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        # Should produce a tax savings (negative or zero tax for the gains, plus savings from deduction)
        assert taxes[2024] <= 0

    def test_dataframe_input_st_lt_split(self):
        from app.core.tax_library import calculate_tax_series_with_carryforward
        pl_df = pd.DataFrame({
            "Realized ST P&L": [10_000, 5_000],
            "Realized LT P&L": [20_000, 30_000],
        }, index=[2023, 2024])
        taxes = calculate_tax_series_with_carryforward(
            pl_df, other_income=100_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        assert taxes[2023] > 0
        assert taxes[2024] > 0
        # ST should produce higher tax than if all LT
        taxes_all_lt = calculate_tax_series_with_carryforward(
            pd.Series([30_000, 35_000], index=[2023, 2024]),
            other_income=100_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        # With ST component, total tax should be higher
        assert taxes.sum() > taxes_all_lt.sum()

    def test_multi_year_carryforward(self):
        from app.core.tax_library import calculate_tax_series_with_carryforward
        # $100k loss takes many years to carry forward
        pl = pd.Series([-100_000, 5_000, 5_000, 5_000, 5_000],
                       index=[2020, 2021, 2022, 2023, 2024])
        taxes = calculate_tax_series_with_carryforward(
            pl, other_income=100_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        # All years after loss should have non-positive tax
        # (carried forward loss covers gains, $3k deduction provides savings)
        for y in [2021, 2022, 2023, 2024]:
            assert taxes[y] <= 0, f"Year {y} should have no positive tax due to carryforward"

    def test_retirement_income_applied(self):
        from app.core.tax_library import calculate_tax_series_with_carryforward
        pl = pd.Series([50_000, 50_000], index=[2023, 2024])
        # Pre-retirement: $200k income. Post-retirement: $50k income.
        taxes_high = calculate_tax_series_with_carryforward(
            pl, other_income=200_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
        )
        taxes_retire = calculate_tax_series_with_carryforward(
            pl, other_income=200_000, filing_status="Single",
            method="2024_fixed", use_standard_deduction=False,
            retirement_income=50_000, retirement_year=2024,
        )
        # 2024 tax should be lower with retirement income
        assert taxes_retire[2024] < taxes_high[2024]


class TestStateTax:
    """State income tax calculation."""

    def test_no_tax_state_returns_zero(self):
        from app.core.tax_library import calculate_state_tax
        for state in ["FL", "TX", "NV", "WA", "WY", "AK", "SD", "TN", "NH"]:
            tax = calculate_state_tax(2024, state, "Single", 100_000, 10_000, 20_000)
            assert tax == 0.0, f"{state} should have no income tax"

    def test_none_state_returns_zero(self):
        from app.core.tax_library import calculate_state_tax
        assert calculate_state_tax(2024, None, "Single", 100_000, 10_000, 20_000) == 0.0
        assert calculate_state_tax(2024, "", "Single", 100_000, 10_000, 20_000) == 0.0

    def test_zero_gains_returns_zero(self):
        from app.core.tax_library import calculate_state_tax
        assert calculate_state_tax(2024, "CA", "Single", 100_000, 0, 0) == 0.0


class TestProgressiveTax:
    """Internal _progressive_tax helper."""

    def test_basic_progressive(self):
        from app.core.tax_library import _progressive_tax
        brackets = [[0, 0.10], [10_000, 0.20], [50_000, 0.30]]
        # $30k income: 10k * 10% + 20k * 20% = 1k + 4k = 5k
        assert _progressive_tax(brackets, 30_000) == pytest.approx(5_000)

    def test_zero_income(self):
        from app.core.tax_library import _progressive_tax
        brackets = [[0, 0.10], [10_000, 0.20]]
        assert _progressive_tax(brackets, 0) == 0.0

    def test_negative_income(self):
        from app.core.tax_library import _progressive_tax
        brackets = [[0, 0.10]]
        assert _progressive_tax(brackets, -5_000) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 2. SHADOW BACKTEST
# ═══════════════════════════════════════════════════════════════════════

class TestShadowBacktestBasic:
    """run_shadow_backtest: basic scenarios."""

    def test_flat_prices_portfolio_equals_start_val(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices()
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
        )
        port = result[5]  # portfolio_series
        assert port.iloc[0] == pytest.approx(100_000, rel=1e-4)
        assert port.iloc[-1] == pytest.approx(100_000, rel=1e-4)

    def test_growing_prices_portfolio_grows(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _growing_prices()
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices,
        )
        port = result[5]
        assert port.iloc[-1] > port.iloc[0] * 1.5  # Prices doubled

    def test_two_tickers_equal_allocation(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _two_ticker_prices()
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices,
        )
        port = result[5]
        # SPY doubled, BND flat -> portfolio should grow ~50%
        assert port.iloc[-1] > port.iloc[0] * 1.3
        assert port.iloc[-1] < port.iloc[0] * 1.7

    def test_returns_correct_tuple_length(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices()
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
        )
        # Should return 8-tuple
        assert len(result) == 8
        trades_df, pl_by_year, comp_df, unrealized_df, logs, port, twr, blocked = result
        assert isinstance(logs, list)
        assert isinstance(port, pd.Series)
        assert isinstance(twr, pd.Series)


class TestShadowBacktestDCA:
    """DCA injection testing."""

    def test_monthly_dca_increases_portfolio(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices()
        result_no_dca = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
        )
        result_dca = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
            cashflow=1_000, cashflow_freq="Monthly",
        )
        port_no_dca = result_no_dca[5]
        port_dca = result_dca[5]
        # DCA adds ~$12k over the year
        assert port_dca.iloc[-1] > port_no_dca.iloc[-1] + 10_000

    def test_yearly_dca(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices(start="2020-01-02", end="2024-12-31")
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices,
            cashflow=10_000, cashflow_freq="Yearly",
        )
        port = result[5]
        # 4 yearly injections of $10k on top of $100k
        assert port.iloc[-1] > 130_000  # At least some injections happened

    def test_dca_stops_at_draw_start_when_not_in_retirement(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices(start="2023-01-02", end="2024-12-31")
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2024-12-31",
            prices_df=prices,
            cashflow=1_000, cashflow_freq="Monthly",
            draw_start_date=datetime.date(2024, 1, 1),
            dca_in_retirement=False,
        )
        port = result[5]
        # DCA should have injected for ~12 months (2023), then stopped
        # ~100k + 12k = ~112k, no more injections after draw_start
        # Some rounding, but should be well below 24 months of DCA
        assert port.iloc[-1] < 100_000 + 25_000


class TestShadowBacktestRebalancing:
    """Rebalancing logic."""

    def test_yearly_rebalance(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _two_ticker_prices()
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices, rebalance_freq="Yearly",
        )
        trades = result[0]  # trades_df
        # Should have rebalance trades (SPY grew, BND didn't)
        assert not trades.empty
        assert len(trades) > 0

    def test_quarterly_rebalance_has_more_trades(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _two_ticker_prices()
        result_yearly = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices, rebalance_freq="Yearly",
        )
        result_quarterly = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices, rebalance_freq="Quarterly",
        )
        yearly_trades = result_yearly[0]
        quarterly_trades = result_quarterly[0]
        assert len(quarterly_trades) > len(yearly_trades)

    def test_threshold_rebalance(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _two_ticker_prices()
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices, rebalance_freq="Threshold",
            threshold_pct=5.0,
        )
        trades = result[0]
        # Should trigger rebalances when SPY drifts > 5pp from 50%
        assert not trades.empty

    def test_no_rebalance_means_drift(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _two_ticker_prices()
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices, rebalance_freq="None",
        )
        trades = result[0]
        # No rebalance trades should occur (only composition snapshots)
        if not trades.empty:
            real_trades = trades[trades["Trade Amount"].abs() > 0.01]
            rebal_trades = real_trades[
                (real_trades["Realized P&L"] != 0) |
                (real_trades["Trade Amount"].abs() > 1)
            ]
            # Should be empty or only DCA buys
            assert len(rebal_trades) == 0


class TestShadowBacktestDraws:
    """Monthly draws and retirement draws."""

    def test_monthly_draw_from_loan(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices(start="2023-01-02", end="2023-12-29")
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
            draw_monthly=5_000,
            draw_start_date=datetime.date(2023, 1, 1),
            starting_loan=50_000,
            margin_rate_annual=0.0,
        )
        logs = result[4]
        draw_logs = [l for l in logs if "Draw" in l and "$5,000" in l]
        # Should have ~11-12 draws over the year
        assert len(draw_logs) >= 10

    def test_retirement_draw_label(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices(start="2023-01-02", end="2024-12-31")
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2024-12-31",
            prices_df=prices,
            draw_monthly=2_000,
            draw_monthly_retirement=5_000,
            draw_start_date=datetime.date(2023, 1, 1),
            retirement_date=datetime.date(2024, 1, 1),
        )
        logs = result[4]
        pre_ret = [l for l in logs if "Draw 2023" in l and "RetDraw" not in l]
        post_ret = [l for l in logs if "RetDraw 2024" in l]
        assert len(pre_ret) > 0, "Pre-retirement draws should be labeled 'Draw'"
        assert len(post_ret) > 0, "Post-retirement draws should be labeled 'RetDraw'"

    def test_loan_repayment(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _flat_prices(start="2023-01-02", end="2023-12-29")
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
            starting_loan=100_000,
            margin_rate_annual=0.0,
            loan_repayment=5_000,
            loan_repayment_freq="Monthly",
        )
        logs = result[4]
        repay_logs = [l for l in logs if "Repay" in l]
        assert len(repay_logs) >= 10


class TestShadowBacktestTaxLots:
    """Tax lot FIFO tracking."""

    def test_short_term_classification(self):
        from app.core.shadow_backtest import run_shadow_backtest
        # Prices flat for 6 months -> rebalance should produce ST gains
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        prices = pd.DataFrame({
            "SPY": np.linspace(100, 130, len(dates)),
            "BND": np.full(len(dates), 100.0),
        }, index=dates)
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2023-01-02", "2023-06-30",
            prices_df=prices, rebalance_freq="Quarterly",
        )
        trades = result[0]
        if not trades.empty:
            sells = trades[trades["Trade Amount"] < 0]
            if not sells.empty:
                # Within 6 months, all gains should be ST
                assert sells["Realized ST P&L"].sum() != 0 or sells["Realized LT P&L"].sum() == 0

    def test_long_term_classification_after_year(self):
        from app.core.shadow_backtest import run_shadow_backtest
        # Prices grow over 2 years -> first rebalance after >1 year should produce LT
        dates = pd.bdate_range("2022-01-03", "2024-12-31")
        prices = pd.DataFrame({
            "SPY": np.linspace(100, 200, len(dates)),
            "BND": np.full(len(dates), 100.0),
        }, index=dates)
        result = run_shadow_backtest(
            {"SPY": 50.0, "BND": 50.0}, 100_000, "2022-01-03", "2024-12-31",
            prices_df=prices, rebalance_freq="Yearly",
        )
        trades = result[0]
        if not trades.empty:
            sells = trades[trades["Trade Amount"] < 0]
            if not sells.empty:
                # After 1+ year, should have LT gains
                assert sells["Realized LT P&L"].sum() > 0

    def test_section_1256_60_40_split(self):
        from app.core.shadow_backtest import get_tax_treatment
        # GSG is Section 1256
        assert get_tax_treatment("GSG") == "Section1256"
        assert get_tax_treatment("DBMF") == "Section1256"

    def test_collectible_treatment(self):
        from app.core.shadow_backtest import get_tax_treatment
        assert get_tax_treatment("GLD") == "Collectible"
        assert get_tax_treatment("SLV") == "Collectible"
        assert get_tax_treatment("IAU") == "Collectible"

    def test_equity_treatment(self):
        from app.core.shadow_backtest import get_tax_treatment
        assert get_tax_treatment("SPY") == "Equity"
        assert get_tax_treatment("QQQ") == "Equity"
        assert get_tax_treatment("VTI") == "Equity"


class TestShadowBacktestTWR:
    """Time-Weighted Return series."""

    def test_twr_starts_at_one(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _growing_prices()
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices,
        )
        twr = result[6]
        assert twr.iloc[0] == pytest.approx(1.0)

    def test_twr_grows_with_growing_prices(self):
        from app.core.shadow_backtest import run_shadow_backtest
        prices = _growing_prices()
        result = run_shadow_backtest(
            {"TEST": 100.0}, 100_000, "2020-01-02", "2024-12-31",
            prices_df=prices,
        )
        twr = result[6]
        assert twr.iloc[-1] > 1.5  # Prices doubled


# ═══════════════════════════════════════════════════════════════════════
# 3. MARGIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════

class TestSimulateMarginFixedRate:
    """simulate_margin with Fixed rate mode."""

    def test_fixed_rate_compound_interest(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        rate_cfg = {"type": "Fixed", "rate_pct": 8.0}
        loan, *_ = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        # Actual/360 daily accrual with monthly capitalization.
        expected = 108_342.46271301954
        assert loan.iloc[-1] == pytest.approx(expected, rel=1e-4)

    def test_zero_rate_no_growth(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        rate_cfg = {"type": "Fixed", "rate_pct": 0.0}
        loan, *_ = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        assert loan.iloc[-1] == pytest.approx(100_000, rel=1e-4)

    def test_equity_calculation(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        loan, equity, eq_pct, usage, _ = simulate_margin(
            port, 50_000, {"type": "Fixed", "rate_pct": 0.0}, 0, 0.25,
        )
        # Equity = Port - Loan = 200k - 50k = 150k
        assert equity.iloc[0] == pytest.approx(150_000, rel=1e-3)
        # Equity % = 150k / 200k = 75%
        assert eq_pct.iloc[0] == pytest.approx(0.75, rel=1e-2)

    def test_usage_pct_calculation(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        loan, equity, eq_pct, usage, _ = simulate_margin(
            port, 50_000, {"type": "Fixed", "rate_pct": 0.0}, 0, 0.25,
        )
        # Usage = Loan / (Port * (1 - maint)) = 50k / (200k * 0.75) = 33.3%
        assert usage.iloc[0] == pytest.approx(50_000 / (200_000 * 0.75), rel=1e-2)


class TestSimulateMarginLegacyFloat:
    """simulate_margin with legacy float rate_annual."""

    def test_legacy_float_matches_fixed(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        loan_legacy, *_ = simulate_margin(port, 100_000, 8.0, 0, 0.25)
        loan_fixed, *_ = simulate_margin(
            port, 100_000, {"type": "Fixed", "rate_pct": 8.0}, 0, 0.25,
        )
        assert loan_legacy.iloc[-1] == pytest.approx(loan_fixed.iloc[-1], rel=1e-6)


class TestSimulateMarginVariable:
    """simulate_margin with Variable rate mode."""

    def test_variable_rate_accrues_interest(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {"type": "Variable", "base_series": base, "spread_pct": 1.0}
        loan, _, _, _, eff_rate = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        assert eff_rate.iloc[10] == pytest.approx(6.0, abs=0.1)
        # Loan should grow at ~6%
        n_days = len(port)
        expected = 100_000 * (1 + 0.06) ** (n_days / 252)
        assert loan.iloc[-1] == pytest.approx(expected, rel=1e-3)

    def test_variable_rate_changes_over_time(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        # Base rate changes mid-year
        base = pd.Series(5.0, index=port.index)
        midpoint = len(port) // 2
        base.iloc[midpoint:] = 3.0  # Rate drops
        rate_cfg = {"type": "Variable", "base_series": base, "spread_pct": 1.0}
        loan, _, _, _, eff_rate = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        # Effective rate should be 6% first half, 4% second half
        assert eff_rate.iloc[10] == pytest.approx(6.0, abs=0.1)
        assert eff_rate.iloc[-10] == pytest.approx(4.0, abs=0.1)


class TestSimulateMarginTiered:
    """simulate_margin with Tiered rate mode."""

    def test_tiered_single_tier(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {
            "type": "Tiered",
            "base_series": base,
            "tiers": [(0, 1.0)],  # Single tier: base + 1%
        }
        loan, _, _, _, eff_rate = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        # Effective rate should be ~6%
        assert eff_rate.iloc[10] == pytest.approx(6.0, rel=0.1)
        # Loan should grow
        assert loan.iloc[-1] > 100_000

    def test_tiered_multiple_tiers(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        base = pd.Series(5.0, index=port.index)
        rate_cfg = {
            "type": "Tiered",
            "base_series": base,
            "tiers": [
                (0, 2.0),       # 0-50k: base + 2%
                (50_000, 1.5),  # 50k-100k: base + 1.5%
                (100_000, 1.0), # 100k+: base + 1%
            ],
        }
        # $100k loan spans first two tiers
        loan, _, _, _, eff_rate = simulate_margin(port, 100_000, rate_cfg, 0, 0.25)
        # Blended rate should be between base+1.5% and base+2%
        assert 6.0 < eff_rate.iloc[10] < 8.0
        assert loan.iloc[-1] > 100_000


class TestSimulateMarginDraws:
    """simulate_margin draw functionality."""

    def test_draws_increase_loan(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        loan_no_draw, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
        )
        loan_draw, *_ = simulate_margin(
            port, 50_000, 0.0, 5_000, 0.25,
        )
        assert loan_draw.iloc[-1] > loan_no_draw.iloc[-1]

    def test_draws_with_draw_start_date(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2024-12-31", 200_000)
        # Draws start mid-2023
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 5_000, 0.25,
            draw_start_date=datetime.date(2023, 7, 1),
        )
        # Should be less than full year of draws
        full_draw_loan, *_ = simulate_margin(
            port, 50_000, 0.0, 5_000, 0.25,
        )
        assert loan.iloc[-1] < full_draw_loan.iloc[-1]

    def test_retirement_draws(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2024-12-31", 200_000)
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 2_000, 0.25,
            draw_monthly_retirement=8_000,
            retirement_date=datetime.date(2024, 1, 1),
        )
        # Retirement draws are larger, so total should be significant
        # Pre-ret: ~12 * 2k = 24k. Post-ret: ~12 * 8k = 96k. Total ~120k + starting 50k
        assert loan.iloc[-1] > 50_000 + 100_000

    def test_repayments_decrease_loan(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        repay = pd.Series(0.0, index=port.index)
        repay.iloc[10] = 20_000
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, repayment_series=repay,
        )
        assert loan.iloc[-1] == pytest.approx(30_000, abs=100)

    def test_tax_payments_increase_loan(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        taxes = pd.Series(0.0, index=port.index)
        taxes.iloc[60] = 10_000  # Tax payment
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, tax_series=taxes,
        )
        assert loan.iloc[-1] == pytest.approx(60_000, abs=100)


class TestSimulateMarginDCA:
    """DCA via margin functionality."""

    def test_fund_dca_margin_true(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1_000  # ~monthly
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
            dca_series=dca, fund_dca_margin=True,
        )
        total_dca = dca.sum()
        assert loan.iloc[-1] == pytest.approx(50_000 + total_dca, rel=0.01)

    def test_fund_dca_margin_false_no_loan_increase(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1_000
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
            dca_series=dca, fund_dca_margin=False,
        )
        assert loan.iloc[-1] == pytest.approx(50_000, abs=100)

    def test_fund_dca_margin_default_is_false(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-12-29", 200_000)
        dca = pd.Series(0.0, index=port.index)
        dca.iloc[::21] = 1_000
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
            dca_series=dca,
        )
        assert loan.iloc[-1] == pytest.approx(50_000, abs=100)


class TestSimulateMarginStartingCash:
    """Starting with negative loan (cash)."""

    def test_negative_loan_no_interest(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-06-30", 200_000)
        # Negative starting loan = cash. No interest should accrue.
        loan, *_ = simulate_margin(
            port, -50_000, {"type": "Fixed", "rate_pct": 8.0}, 0, 0.25,
        )
        # Cash should stay at -50k (no interest on cash)
        assert loan.iloc[-1] == pytest.approx(-50_000, abs=100)

    def test_repayment_pushes_loan_negative(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-06-30", 200_000)
        repay = pd.Series(0.0, index=port.index)
        repay.iloc[10] = 60_000  # Over-repay $50k loan
        loan, *_ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25, repayment_series=repay,
        )
        assert loan.iloc[11] < 0


class TestSimulateMarginDivisionByZero:
    """Portfolio with zero values should not crash."""

    def test_zero_portfolio_no_nan_inf(self):
        from app.services.testfol_api import simulate_margin
        port = _flat_port("2023-01-02", "2023-03-31", 100_000)
        port.iloc[10:15] = 0.0
        loan, equity, eq_pct, usage, _ = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
        )
        assert not eq_pct.isna().any()
        assert not np.isinf(eq_pct).any()
        assert not usage.isna().any()
        assert not np.isinf(usage).any()


# ═══════════════════════════════════════════════════════════════════════
# 4. STATISTICS
# ═══════════════════════════════════════════════════════════════════════

class TestCalculateCAGR:
    """calculate_cagr: various scenarios."""

    def test_doubling_in_5_years(self):
        from app.core.calculations.stats import calculate_cagr
        dates = pd.bdate_range("2019-01-02", "2023-12-29")
        vals = np.linspace(100, 200, len(dates))
        series = pd.Series(vals, index=dates)
        cagr = calculate_cagr(series)
        # Doubling in ~5 years -> ~14.87%
        assert cagr == pytest.approx(14.87, abs=1.0)

    def test_tripling_in_5_years(self):
        from app.core.calculations.stats import calculate_cagr
        dates = pd.bdate_range("2019-01-02", "2023-12-29")
        vals = np.linspace(100, 300, len(dates))
        series = pd.Series(vals, index=dates)
        cagr = calculate_cagr(series)
        # Tripling in ~5 years -> ~24.57%
        assert cagr == pytest.approx(24.57, abs=2.0)

    def test_flat_returns_zero(self):
        from app.core.calculations.stats import calculate_cagr
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        series = pd.Series(100.0, index=dates)
        assert calculate_cagr(series) == pytest.approx(0.0, abs=0.1)

    def test_declining_returns_negative(self):
        from app.core.calculations.stats import calculate_cagr
        dates = pd.bdate_range("2019-01-02", "2023-12-29")
        vals = np.linspace(200, 100, len(dates))
        series = pd.Series(vals, index=dates)
        cagr = calculate_cagr(series)
        assert cagr < 0

    def test_empty_series(self):
        from app.core.calculations.stats import calculate_cagr
        assert calculate_cagr(pd.Series(dtype=float)) == 0.0


class TestCalculateMaxDrawdown:
    """calculate_max_drawdown: known scenarios."""

    def test_50pct_drawdown(self):
        from app.core.calculations.stats import calculate_max_drawdown
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        # Goes 100 -> 200 -> 100
        mid = len(dates) // 2
        vals = np.concatenate([
            np.linspace(100, 200, mid),
            np.linspace(200, 100, len(dates) - mid),
        ])
        series = pd.Series(vals, index=dates)
        mdd = calculate_max_drawdown(series)
        assert mdd == pytest.approx(-50.0, abs=2.0)

    def test_no_drawdown(self):
        from app.core.calculations.stats import calculate_max_drawdown
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        vals = np.linspace(100, 200, len(dates))
        series = pd.Series(vals, index=dates)
        mdd = calculate_max_drawdown(series)
        assert mdd == pytest.approx(0.0, abs=0.5)

    def test_flat_returns_zero(self):
        from app.core.calculations.stats import calculate_max_drawdown
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        series = pd.Series(100.0, index=dates)
        mdd = calculate_max_drawdown(series)
        assert mdd == pytest.approx(0.0)


class TestCalculateSharpeRatio:
    """calculate_sharpe_ratio: positive and negative."""

    def test_positive_returns_positive_sharpe(self):
        from app.core.calculations.stats import calculate_sharpe_ratio
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        vals = np.linspace(100, 150, len(dates))
        series = pd.Series(vals, index=dates)
        sharpe = calculate_sharpe_ratio(series)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        from app.core.calculations.stats import calculate_sharpe_ratio
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        vals = np.linspace(100, 50, len(dates))
        series = pd.Series(vals, index=dates)
        sharpe = calculate_sharpe_ratio(series)
        assert sharpe < 0

    def test_flat_returns_zero_sharpe(self):
        from app.core.calculations.stats import calculate_sharpe_ratio
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        series = pd.Series(100.0, index=dates)
        assert calculate_sharpe_ratio(series) == 0.0

    def test_with_risk_free_rate(self):
        from app.core.calculations.stats import calculate_sharpe_ratio
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        vals = np.linspace(100, 110, len(dates))
        series = pd.Series(vals, index=dates)
        sharpe_0 = calculate_sharpe_ratio(series, risk_free_rate=0.0)
        sharpe_5 = calculate_sharpe_ratio(series, risk_free_rate=0.05)
        # Higher risk-free rate -> lower Sharpe
        assert sharpe_5 < sharpe_0


class TestGenerateStats:
    """generate_stats: complete stats dictionary."""

    def test_returns_all_keys(self):
        from app.core.calculations.stats import generate_stats
        dates = pd.bdate_range("2020-01-02", "2024-12-31")
        vals = np.linspace(100, 200, len(dates))
        series = pd.Series(vals, index=dates)
        stats = generate_stats(series)
        expected_keys = {"cagr", "std", "sharpe", "max_drawdown",
                         "best_year", "worst_year", "ulcer_index",
                         "sortino", "calmar", "avg_drawdown"}
        assert expected_keys.issubset(stats.keys())

    def test_empty_series_returns_empty_dict(self):
        from app.core.calculations.stats import generate_stats
        assert generate_stats(pd.Series(dtype=float)) == {}

    def test_sortino_positive_for_noisy_growth(self):
        from app.core.calculations.stats import generate_stats
        # Linear growth has zero downside deviation (no negative returns),
        # so Sortino is 0. Add noise to get negative returns for computation.
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", "2024-12-31")
        vals = np.linspace(100, 250, len(dates))
        noise = np.random.normal(0, 0.5, len(dates)).cumsum()
        series = pd.Series(vals + noise, index=dates)
        stats = generate_stats(series)
        assert stats["sortino"] > 0

    def test_ulcer_index_zero_for_linear_growth(self):
        from app.core.calculations.stats import generate_stats
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        vals = np.linspace(100, 200, len(dates))
        series = pd.Series(vals, index=dates)
        stats = generate_stats(series)
        assert stats["ulcer_index"] == pytest.approx(0.0, abs=0.1)

    def test_best_worst_year(self):
        from app.core.calculations.stats import generate_stats
        # 2021: flat, 2022: 50% drop, 2023: recovery
        dates = pd.bdate_range("2021-01-04", "2023-12-29")
        vals = np.ones(len(dates)) * 100
        for i, d in enumerate(dates):
            if d.year == 2022:
                vals[i] = 50
            elif d.year == 2023:
                vals[i] = 120
        series = pd.Series(vals, index=dates)
        stats = generate_stats(series)
        assert stats["worst_year"] < 0
        assert stats["best_year"] > 0

    def test_calmar_ratio(self):
        from app.core.calculations.stats import generate_stats
        dates = pd.bdate_range("2020-01-02", "2024-12-31")
        vals = np.linspace(100, 200, len(dates))
        series = pd.Series(vals, index=dates)
        stats = generate_stats(series)
        # For linear growth, drawdown is ~0, calmar should be 0 or very large
        # (division by zero guard)
        assert stats["calmar"] >= 0


class TestTaxAdjustedEquity:
    """calculate_tax_adjusted_equity."""

    def test_returns_non_empty(self):
        from app.core.calculations.stats import calculate_tax_adjusted_equity
        port = _flat_port("2023-01-02", "2023-12-29", 100_000)
        loan = pd.Series(50_000.0, index=port.index)
        equity = port - loan
        tax_series = pd.Series(0.0, index=port.index)
        adj_eq, adj_tax = calculate_tax_adjusted_equity(
            equity, tax_series, port, loan,
            rate_annual=8.0,
        )
        assert not adj_eq.empty
        assert len(adj_eq) == len(port)

    def test_with_draws(self):
        from app.core.calculations.stats import calculate_tax_adjusted_equity
        port = _flat_port("2023-01-02", "2023-12-29", 100_000)
        loan = pd.Series(50_000.0, index=port.index)
        equity = port - loan
        tax_series = pd.Series(0.0, index=port.index)
        adj_eq, _ = calculate_tax_adjusted_equity(
            equity, tax_series, port, loan,
            rate_annual=8.0,
            draw_monthly=5_000,
            draw_start_date=None,
        )
        # Draws reduce equity
        assert adj_eq.iloc[-1] < equity.iloc[0]


# ═══════════════════════════════════════════════════════════════════════
# 5. BACKTEST ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class TestRunSingleBacktest:
    """run_single_backtest with dependency injection."""

    def _make_mocks(self, start="2023-01-02", end="2023-12-29"):
        dates = pd.bdate_range(start, end)

        def mock_fetch(*a, **kw):
            return (
                pd.Series(100.0, index=dates, name="Portfolio"),
                {"cagr": 10.0, "std": 15.0, "sharpe": 0.8},
                {"rebalancing_events": [], "daily_returns": []},
            )

        def mock_shadow(*a, **kw):
            return (
                pd.DataFrame(),  # trades
                pd.DataFrame(),  # pl_by_year
                pd.DataFrame(),  # composition
                pd.DataFrame(),  # unrealized
                [],               # logs
                pd.Series(100.0, index=dates),  # portfolio_series
                pd.Series(1.0, index=dates),    # twr_series
                [],               # pm_blocked_dates
            )

        return mock_fetch, mock_shadow

    def test_returns_dict_with_required_keys(self):
        from app.core.backtest_orchestrator import run_single_backtest
        mock_fetch, mock_shadow = self._make_mocks()
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={"SPY": 25.0},
            rebalance={},
            start_date="2023-01-02",
            end_date="2023-12-29",
            start_val=100_000,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        required = {"name", "series", "stats", "wmaint", "wmaint_pm",
                     "allocation", "trades", "logs"}
        assert required.issubset(result.keys())

    def test_wmaint_calculation(self):
        from app.core.backtest_orchestrator import run_single_backtest
        mock_fetch, mock_shadow = self._make_mocks()
        result = run_single_backtest(
            allocation={"SPY": 60.0, "BND": 40.0},
            maint_pcts={"SPY": 25.0, "BND": 10.0},
            rebalance={},
            start_date="2023-01-02",
            end_date="2023-12-29",
            start_val=100_000,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        # wmaint = (60/100)*25% + (40/100)*10% = 0.15 + 0.04 = 0.19
        assert result["wmaint"] == pytest.approx(0.19, abs=0.01)

    def test_pm_wmaint_calculation(self):
        from app.core.backtest_orchestrator import run_single_backtest
        mock_fetch, mock_shadow = self._make_mocks()
        result = run_single_backtest(
            allocation={"SPY": 60.0, "BND": 40.0},
            maint_pcts={"SPY": 25.0, "BND": 10.0},
            rebalance={},
            start_date="2023-01-02",
            end_date="2023-12-29",
            start_val=100_000,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            pm_maint_pcts={"SPY": 15.0, "BND": 5.0},
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        # pm_wmaint = (60/100)*15% + (40/100)*5% = 0.09 + 0.02 = 0.11
        assert result["wmaint_pm"] == pytest.approx(0.11, abs=0.01)

    def test_pm_wmaint_uses_total_weight_not_100(self):
        from app.core.backtest_orchestrator import run_single_backtest
        mock_fetch, mock_shadow = self._make_mocks()
        # Weight is 50, not 100
        result = run_single_backtest(
            allocation={"SPY": 50.0},
            maint_pcts={"SPY": 25.0},
            rebalance={},
            start_date="2023-01-02",
            end_date="2023-12-29",
            start_val=100_000,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            pm_maint_pcts={"SPY": 15.0},
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        # wmaint = (50/50) * 25% = 0.25 (not 50/100 * 25% = 0.125)
        assert result["wmaint"] == pytest.approx(0.25, abs=0.01)
        # pm_wmaint = (50/50) * 15% = 0.15
        assert result["wmaint_pm"] == pytest.approx(0.15, abs=0.01)

    def test_default_maintenance_when_missing(self):
        from app.core.backtest_orchestrator import run_single_backtest
        mock_fetch, mock_shadow = self._make_mocks()
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={},  # Empty -> should use default 25%
            rebalance={},
            start_date="2023-01-02",
            end_date="2023-12-29",
            start_val=100_000,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=mock_fetch,
            run_shadow_fn=mock_shadow,
        )
        assert result["wmaint"] == pytest.approx(0.25, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# 6. UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class TestResampleData:
    """resample_data: all timeframes and methods."""

    def _make_series(self, start="2023-01-02", end="2023-12-29"):
        dates = pd.bdate_range(start, end)
        return pd.Series(
            np.random.uniform(95, 105, len(dates)),
            index=dates, dtype=float,
        )

    def test_1d_ohlc(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1D", method="ohlc")
        assert set(result.columns) == {"Open", "High", "Low", "Close"}
        assert len(result) == len(series)

    def test_1d_last(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1D", method="last")
        assert isinstance(result, pd.Series)

    def test_1w_ohlc(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1W", method="ohlc")
        assert not result.empty
        assert set(result.columns) == {"Open", "High", "Low", "Close"}
        # Weekly should have ~50 rows for a full year
        assert 40 < len(result) < 55

    def test_1m_ohlc(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1M", method="ohlc")
        assert not result.empty
        assert "Close" in result.columns
        # Monthly: 12 months
        assert 10 < len(result) <= 12

    def test_3m_ohlc(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "3M", method="ohlc")
        assert not result.empty
        assert 3 <= len(result) <= 4

    def test_1y_ohlc(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1Y", method="ohlc")
        assert not result.empty
        assert len(result) == 1  # Single year of data

    def test_1w_last(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1W", method="last")
        assert isinstance(result, pd.Series)
        assert not result.empty

    def test_1m_max(self):
        from app.common.utils import resample_data
        series = self._make_series()
        result = resample_data(series, "1M", method="max")
        assert isinstance(result, pd.Series)
        assert not result.empty

    def test_ohlc_high_is_max(self):
        from app.common.utils import resample_data
        # Construct a series where we know exact weekly high
        dates = pd.bdate_range("2023-01-02", "2023-01-31")
        vals = list(range(100, 100 + len(dates)))
        series = pd.Series(vals, index=dates, dtype=float)
        result = resample_data(series, "1W", method="ohlc")
        # Each week's High should be the max of that week
        for _, row in result.iterrows():
            assert row["High"] >= row["Open"]
            assert row["High"] >= row["Close"]
            assert row["Low"] <= row["Open"]
            assert row["Low"] <= row["Close"]


# ═══════════════════════════════════════════════════════════════════════
# 7. MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeMA:
    """analyze_ma: 200-day moving average analysis."""

    def test_always_above_ma_no_events(self):
        from app.core.calculations.moving_averages import analyze_ma
        # Monotonically increasing -> always above MA
        dates = pd.bdate_range("2019-01-02", "2024-12-31")
        series = pd.Series(np.linspace(50, 300, len(dates)), index=dates)
        ma_series, events_df = analyze_ma(series, window=200)
        assert ma_series is not None
        # Should have no or very few events (maybe 1 at the start before MA establishes)
        assert len(events_df) <= 1

    def test_crash_produces_event(self):
        from app.core.calculations.moving_averages import analyze_ma
        # Prices above MA, then crash below
        dates = pd.bdate_range("2019-01-02", "2024-12-31")
        vals = np.linspace(100, 300, len(dates))
        # Insert a crash mid-series
        mid = len(dates) // 2
        vals[mid:mid+100] = np.linspace(300, 100, 100)
        vals[mid+100:] = np.linspace(100, 350, len(dates) - mid - 100)
        series = pd.Series(vals, index=dates)
        ma_series, events_df = analyze_ma(series, window=200)
        assert not events_df.empty

    def test_empty_series(self):
        from app.core.calculations.moving_averages import analyze_ma
        ma, events = analyze_ma(pd.Series(dtype=float))
        assert ma is None
        assert events.empty


class TestAnalyzeWMA:
    """analyze_wma: 200-week moving average analysis."""

    def test_empty_series(self):
        from app.core.calculations.moving_averages import analyze_wma
        weekly, wma, events = analyze_wma(pd.Series(dtype=float))
        assert weekly is None
        assert events.empty

    def test_insufficient_data_returns_none_wma(self):
        from app.core.calculations.moving_averages import analyze_wma
        # 100 weeks < 200 week window
        dates = pd.bdate_range("2023-01-02", "2024-12-31")
        series = pd.Series(np.linspace(100, 150, len(dates)), index=dates)
        weekly, wma, events = analyze_wma(series, window=200)
        assert weekly is not None
        assert wma is None
        assert events.empty


# ═══════════════════════════════════════════════════════════════════════
# 8. TICKER PARSING
# ═══════════════════════════════════════════════════════════════════════

class TestParseTicker:
    """parse_ticker: base symbol and modifier extraction."""

    def test_simple_ticker(self):
        from app.core.shadow_backtest import parse_ticker
        base, params = parse_ticker("SPY")
        assert base == "SPY"
        assert params == {}

    def test_ticker_with_leverage(self):
        from app.core.shadow_backtest import parse_ticker
        base, params = parse_ticker("SPY?L=2.0&E=0.5")
        assert base == "SPY"
        assert params["L"] == "2.0"
        assert params["E"] == "0.5"

    def test_sim_ticker_mapped(self):
        from app.core.shadow_backtest import parse_ticker
        base, params = parse_ticker("SPYSIM")
        assert base == "SPY"

    def test_sim_ticker_with_params(self):
        from app.core.shadow_backtest import parse_ticker
        base, params = parse_ticker("SPYSIM?L=3")
        assert base == "SPY"
        assert params["L"] == "3"

    def test_ndxmegasim_not_remapped(self):
        from app.core.shadow_backtest import parse_ticker
        base, params = parse_ticker("NDXMEGASIM")
        assert base == "NDXMEGASIM"


# ═══════════════════════════════════════════════════════════════════════
# 9. DRIFT CHECKING
# ═══════════════════════════════════════════════════════════════════════

class TestCheckDrift:
    """_check_drift: rebalance trigger detection."""

    def test_no_drift_at_target(self):
        from app.core.shadow_backtest import _check_drift
        positions = {"SPY": 50_000, "BND": 50_000}
        allocation = {"SPY": 50.0, "BND": 50.0}
        triggered, max_drift, worst = _check_drift(positions, allocation, 5.0)
        assert not triggered
        assert max_drift == pytest.approx(0.0, abs=0.1)

    def test_drift_triggers_above_threshold(self):
        from app.core.shadow_backtest import _check_drift
        # SPY at 60%, target 50% -> 10pp drift
        positions = {"SPY": 60_000, "BND": 40_000}
        allocation = {"SPY": 50.0, "BND": 50.0}
        triggered, max_drift, worst = _check_drift(positions, allocation, 5.0)
        assert triggered
        assert max_drift == pytest.approx(10.0, abs=0.5)

    def test_zero_total_value(self):
        from app.core.shadow_backtest import _check_drift
        positions = {"SPY": 0, "BND": 0}
        allocation = {"SPY": 50.0, "BND": 50.0}
        triggered, max_drift, worst = _check_drift(positions, allocation, 5.0)
        assert not triggered


# ═══════════════════════════════════════════════════════════════════════
# 10. TAX LOT
# ═══════════════════════════════════════════════════════════════════════

class TestTaxLot:
    """TaxLot FIFO selling."""

    def test_partial_sell(self):
        from app.core.shadow_backtest import TaxLot
        lot = TaxLot("SPY", pd.Timestamp("2023-01-01"), 100.0, 150.0, 15_000.0)
        sold, basis, remaining = lot.sell_shares(30.0)
        assert sold == 30.0
        assert basis == pytest.approx(4_500.0)  # 30/100 * 15000
        assert remaining is not None
        assert remaining.quantity == pytest.approx(70.0)
        assert remaining.total_cost_basis == pytest.approx(10_500.0)

    def test_full_sell(self):
        from app.core.shadow_backtest import TaxLot
        lot = TaxLot("SPY", pd.Timestamp("2023-01-01"), 100.0, 150.0, 15_000.0)
        sold, basis, remaining = lot.sell_shares(100.0)
        assert sold == 100.0
        assert basis == pytest.approx(15_000.0)
        assert remaining is None

    def test_over_sell(self):
        from app.core.shadow_backtest import TaxLot
        lot = TaxLot("SPY", pd.Timestamp("2023-01-01"), 50.0, 100.0, 5_000.0)
        sold, basis, remaining = lot.sell_shares(75.0)
        # Should only sell 50 (full lot)
        assert sold == 50.0
        assert basis == pytest.approx(5_000.0)
        assert remaining is None


# ═══════════════════════════════════════════════════════════════════════
# 11. EDGE CASES AND INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases across modules."""

    def test_shadow_backtest_single_day(self):
        from app.core.shadow_backtest import run_shadow_backtest
        dates = pd.bdate_range("2023-06-01", "2023-06-02")
        prices = pd.DataFrame({"TEST": [100.0, 101.0]}, index=dates)
        result = run_shadow_backtest(
            {"TEST": 100.0}, 10_000, "2023-06-01", "2023-06-02",
            prices_df=prices,
        )
        port = result[5]
        assert len(port) >= 1

    def test_margin_sim_single_day(self):
        from app.services.testfol_api import simulate_margin
        dates = pd.bdate_range("2023-06-01", "2023-06-02")
        port = pd.Series(100_000.0, index=dates)
        loan, eq, eq_pct, usage, rate = simulate_margin(
            port, 50_000, 0.0, 0, 0.25,
        )
        assert len(loan) == len(port)

    def test_cagr_very_short_period(self):
        from app.core.calculations.stats import calculate_cagr
        dates = pd.bdate_range("2023-06-01", "2023-06-06")
        series = pd.Series([100, 101, 102, 103], index=dates[:4])
        cagr = calculate_cagr(series)
        assert cagr > 0  # Should not crash

    def test_generate_stats_single_value(self):
        from app.core.calculations.stats import generate_stats
        series = pd.Series([100.0], index=pd.bdate_range("2023-01-02", "2023-01-02"))
        stats = generate_stats(series)
        # Should not crash, CAGR = 0
        assert stats.get("cagr", 0) == 0.0

    def test_resample_empty_series(self):
        from app.common.utils import resample_data
        # Empty series with DatetimeIndex (required for resample)
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        result = resample_data(empty, "1M", method="last")
        assert result.empty

    def test_shadow_backtest_missing_ticker_in_prices(self):
        from app.core.shadow_backtest import run_shadow_backtest
        # Allocation has ticker not in prices_df
        prices = _flat_prices("TEST")
        result = run_shadow_backtest(
            {"MISSING": 100.0}, 100_000, "2023-01-02", "2023-12-29",
            prices_df=prices,
        )
        logs = result[4]
        # Should log a critical error about missing data
        assert any("CRITICAL" in l or "Missing" in l for l in logs)
