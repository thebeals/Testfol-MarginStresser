"""Results package — render() orchestrator with tab delegation."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import logging

from app.common import utils
from app.core import calculations, tax_library
from app.services import testfol_api as api
from app.reporting import report_generator
from app.ui import charts, xray_view

from app.ui.results.tabs_chart import render_chart_tab
from app.ui.results.tabs_tax import render_tax_impact_tab
from app.ui.results.tabs_monte_carlo import render_monte_carlo_tab
from app.ui.results.tabs_debug import render_debug_tab


def _extract_backtest_params(results: dict, config: dict | None) -> tuple[dict | None, dict | None, dict | None]:
    """Extract allocation, maint_pcts, rebalance from results (or config fallback)."""
    allocation = results.get("allocation")
    maint_pcts = results.get("maint_pcts")
    rebalance = results.get("rebalance")

    if (not allocation or maint_pcts is None) and config:
        port_name = results.get("name", "")
        for p_cfg in config.get("portfolios", []):
            if p_cfg.get("name") == port_name:
                adf = p_cfg.get("alloc_df")
                if adf is not None and not adf.empty:
                    allocation = dict(zip(adf["Ticker"], adf["Weight %"]))
                    maint_pcts = {
                        row["Ticker"].split("?")[0]: float(row.get("Maint %", 25.0))
                        for _, row in adf.iterrows()
                    }
                reb = p_cfg.get("rebalance", {})
                rebalance = dict(reb)
                break

    return allocation, maint_pcts, rebalance


def _clip_with_return_anchor(series: pd.Series | None, start_date) -> pd.Series | None:
    """Clip to a visible start while retaining one prior observation for returns."""
    if series is None or series.empty:
        return series
    start = pd.Timestamp(start_date)
    ordered = series.sort_index()
    prior = ordered[ordered.index < start].tail(1)
    return pd.concat([prior, ordered[ordered.index >= start]])


@st.cache_data(show_spinner=False)
def _compute_fresh_yearly_returns(
    allocation: dict, maint_pcts: dict, rebalance: dict,
    series_start: str, series_end: str, series_start_val: float,
    year_list: tuple[int, ...],
    cache_policy: str = "effective-start-years-v4",
) -> tuple[dict, pd.Series | None]:
    """Run a fresh 1-year backtest per calendar year to get drift-free yearly returns (cached)."""
    import time
    from app.core.backtest_orchestrator import run_single_backtest

    series_start_ts = pd.Timestamp(series_start)
    series_end_ts = pd.Timestamp(series_end)
    year_list = tuple(
        y for y in year_list
        if series_start_ts.year <= y <= series_end_ts.year
    )
    if not year_list:
        return {}, None

    _log = logging.getLogger("fresh_returns")
    _log.info("Fresh yearly returns: computing %d years (%d–%d, %s)...", len(year_list), year_list[0], year_list[-1], cache_policy)
    t0 = time.perf_counter()

    last_date = series_end_ts + pd.Timedelta(days=7)
    fresh = {}
    year_series = []

    for i, y in enumerate(year_list, 1):
        start = max(pd.Timestamp(f"{y}-01-01"), series_start_ts).strftime("%Y-%m-%d")
        end = last_date.strftime("%Y-%m-%d") if y == year_list[-1] else f"{y}-12-31"

        try:
            t_y = time.perf_counter()
            res = run_single_backtest(
                allocation=allocation,
                maint_pcts=maint_pcts,
                rebalance=rebalance,
                start_date=start,
                end_date=end,
                start_val=100000.0,
                cashflow_amount=0.0,
                cashflow_freq="Monthly",
                invest_div=True,
                pay_down_margin=False,
                tax_config={},
                bearer_token=None,
                name=f"Fresh_{y}",
            )
            s = res.get("series")
            if s is None:
                s = res.get("port_series")
            if s is not None and len(s) >= 2:
                fresh[y] = (s.iloc[-1] / s.iloc[0]) - 1
                year_series.append(s)
                _log.info("  [%d/%d] %d: %.2f%% (%.1fs)", i, len(year_list), y, fresh[y] * 100, time.perf_counter() - t_y)
            else:
                _log.warning("  [%d/%d] %d: no data", i, len(year_list), y)
        except Exception as e:
            _log.warning("  [%d/%d] %d: failed (%s)", i, len(year_list), y, e)

    stitched = None
    if year_series:
        pieces = []
        running_value = series_start_val
        for s in year_series:
            normalized = s / s.iloc[0] * running_value
            pieces.append(normalized)
            running_value = normalized.iloc[-1]
        stitched = pd.concat(pieces)
        stitched = stitched[~stitched.index.duplicated(keep='last')]
        stitched = stitched.sort_index()

    _log.info("Fresh yearly returns: done in %.1fs (%d years computed)", time.perf_counter() - t0, len(fresh))
    return fresh, stitched


@st.cache_data(show_spinner=False)
def _cached_tax_calculations(
    pl_by_year: pd.DataFrame, other_income: float, filing_status: str,
    tax_method: str, use_std_deduction: bool, state_code: str,
    retirement_income, retirement_year, retirement_date,
):
    """Cached federal + state tax calculation."""
    import time
    _log = logging.getLogger("results")
    _log.info("Tax calculations: computing federal + state (%s, %s)...", tax_method, filing_status)
    t0 = time.perf_counter()
    fed = tax_library.calculate_tax_series_with_carryforward(
        pl_by_year, other_income, filing_status,
        method=tax_method, use_standard_deduction=use_std_deduction,
        retirement_income=retirement_income, retirement_year=retirement_year,
        retirement_date=retirement_date,
    )
    state = tax_library.calculate_state_tax_series_with_carryforward(
        pl_by_year, other_income, state_code, filing_status,
        use_standard_deduction=use_std_deduction,
        retirement_income=retirement_income, retirement_year=retirement_year,
        retirement_date=retirement_date,
    )
    _log.info("Tax calculations: done in %.1fs (fed=$%.0f, state=$%.0f)", time.perf_counter() - t0, fed.sum(), state.sum())
    return fed, state


@st.cache_data(show_spinner=False)
def _cached_simulate_margin(
    port_series: pd.Series, starting_loan: float, rate_annual,
    draw_monthly: float, wmaint: float,
    tax_series: pd.Series | None, repayment_series: pd.Series | None,
    draw_start_date, draw_monthly_retirement: float, retirement_date,
    dca_series: pd.Series | None, fund_dca_margin: bool,
    accrual_start_date=None,
):
    """Cached margin simulation."""
    import time
    _log = logging.getLogger("results")
    _log.info("Margin simulation: loan=$%.0f rate=%s draw=$%.0f/mo...", starting_loan, rate_annual, draw_monthly)
    t0 = time.perf_counter()
    result = api.simulate_margin(
        port_series, starting_loan, rate_annual, draw_monthly, wmaint,
        tax_series=tax_series, repayment_series=repayment_series,
        draw_start_date=draw_start_date,
        draw_monthly_retirement=draw_monthly_retirement,
        retirement_date=retirement_date,
        dca_series=dca_series, fund_dca_margin=fund_dca_margin,
        accrual_start_date=accrual_start_date,
    )
    _log.info("Margin simulation: done in %.1fs", time.perf_counter() - t0)
    return result


@st.cache_data(show_spinner=False)
def _cached_resample_all(
    tax_adj_port_series: pd.Series, final_adj_series: pd.Series,
    loan_series: pd.Series, usage_series: pd.Series,
    equity_pct_series: pd.Series, rate_series: pd.Series,
    timeframe: str,
):
    """Cached batch resampling of all 6 display series."""
    import time
    _log = logging.getLogger("results")
    _log.info("Resampling: 6 series at %s...", timeframe)
    t0 = time.perf_counter()
    ohlc = utils.resample_data(tax_adj_port_series, timeframe, method="ohlc")
    equity = utils.resample_data(final_adj_series, timeframe, method="last")
    loan = utils.resample_data(loan_series, timeframe, method="last")
    usage = utils.resample_data(usage_series, timeframe, method="max")
    equity_pct = utils.resample_data(equity_pct_series, timeframe, method="last")
    rate = utils.resample_data(rate_series, timeframe, method="last")
    _log.info("Resampling: done in %.1fs", time.perf_counter() - t0)
    return ohlc, equity, loan, usage, equity_pct, rate


def render(results: dict, config: dict, portfolio_name: str = "", clip_start_date=None) -> None:
    """
    Renders the results section, including metrics and charts.
    """

    # Extract Data from Results
    port_series = results["port_series"]
    stats = results["stats"]
    portfolio_name = results.get("name", "Portfolio")
    component_prices = results.get("component_prices", pd.DataFrame())


    # --- Clip Data Logic (Sync with Chart) ---
    original_start_date = results.get("start_date")
    twr_series = results.get("twr_series")
    start_candidates = [
        pd.Timestamp(value)
        for value in (
            original_start_date,
            results.get("effective_start_date"),
            clip_start_date,
        )
        if value is not None
    ]
    analysis_start_date = (
        max(start_candidates)
        if start_candidates
        else (port_series.index[0] if not port_series.empty else None)
    )

    if analysis_start_date is not None and not port_series.empty:
        # Avoid clipping if clip_start is before port start
        if analysis_start_date > port_series.index[0]:
            port_series = _clip_with_return_anchor(port_series, analysis_start_date)

            # Recalculate Stats for clipped period
            if not port_series.empty:
                # Use TWR Series for stats if available (Correct for Cashflows)
                target_series = port_series
                if twr_series is not None and not twr_series.empty:
                    twr_clipped = _clip_with_return_anchor(twr_series, analysis_start_date)
                    if not twr_clipped.empty:
                        target_series = twr_clipped

                stats = calculations.generate_stats(target_series)

    # Initialize optional chart variables to prevent UnboundLocalError
    fig_tax_impact = None

    if port_series.empty or not isinstance(port_series.index, pd.DatetimeIndex):
        st.error("No valid simulation data generated (or data completely clipped). Check inputs or API connection.")
        return

    trades_df = results["trades_df"]
    pl_by_year = results["pl_by_year"]
    composition_df = results.get("composition_df", pd.DataFrame())
    raw_response = results["raw_response"]

    start_val = port_series.iloc[0] # RE-BASE start val to the clipped start

    wmaint = results.get("wmaint", config.get('wmaint', 0.25))

    # Tax/Margin Config
    pay_tax_margin = config.get('pay_tax_margin', False)
    pay_tax_cash = config.get('pay_tax_cash', False)
    other_income = config.get('other_income', 0.0)
    filing_status = config.get('filing_status', 'Single')
    tax_method = config.get('tax_method', '2026_fixed')
    use_std_deduction = config.get('use_std_deduction', True)
    state_code = config.get('state_code', '')

    rate_annual = config.get('rate_annual', 8.0)
    draw_monthly = config.get('draw_monthly', 0.0)
    draw_monthly_retirement = config.get('draw_monthly_retirement', 0.0)
    draw_start_date = config.get('draw_start_date', None)
    retirement_date = config.get('retirement_date', None)
    retirement_income = config.get('retirement_income', None)
    starting_loan = config.get('starting_loan', 0.0)
    starting_cash = config.get('starting_cash', 0.0)
    if starting_cash > 0 and starting_loan == 0:
        starting_loan = -starting_cash  # Cash is modeled as negative loan

    # Derive retirement year from retirement_date only (not draw_start_date)
    _retirement_year = None
    if retirement_income is not None and retirement_income > 0 and retirement_date is not None:
        if hasattr(retirement_date, 'year'):
            _retirement_year = retirement_date.year
        elif isinstance(retirement_date, str):
            _retirement_year = int(retirement_date[:4])

    _gcf = config.get('global_cashflow', {})
    cashflow = _gcf.get('amount', config.get('cashflow', 0.0))
    cashfreq = _gcf.get('freq', config.get('cashfreq', "Monthly"))
    pay_down_margin = _gcf.get('pay_down_margin', config.get('pay_down_margin', False))
    fund_dca_margin = _gcf.get('fund_dca_margin', False)
    dca_in_retirement = config.get('dca_in_retirement', True)

    chart_style = config.get('chart_style', "Classic (Combined)")
    timeframe = config.get('timeframe', "1M")
    # Candlestick chart has its own timeframe pills — let it override sidebar
    if chart_style == "Candlestick":
        timeframe = st.session_state.get('candlestick_tf', timeframe)
    log_scale = config.get('log_scale', True)
    show_range_slider = config.get('show_range_slider', True)
    show_volume = config.get('show_volume', True)

    pm_enabled = config.get('pm_enabled', False)
    pm_mode = config.get('pm_mode', 'Off')
    pm_threshold = config.get('pm_threshold', 110000.0)
    wmaint_pm = results.get("wmaint_pm", 0.0)
    # Recompute wmaint_pm from current allocation so PM reflects UI changes without re-run
    if pm_mode != 'Off':
        _computed_pm = 0.0
        _port_name = results.get("name", "")
        for _p_cfg in config.get("portfolios", []):
            if _p_cfg.get("name") == _port_name:
                _adf = _p_cfg.get("alloc_df")
                if _adf is not None and not _adf.empty and "PM Maint %" in _adf.columns:
                    for _, _r in _adf.iterrows():
                        _pm_v = float(_r.get("PM Maint %", 0))
                        _wt = float(_r.get("Weight %", 0))
                        if _pm_v > 0:
                            _computed_pm += (_wt / 100) * (_pm_v / 100)
                break
        if _computed_pm > 0:
            wmaint_pm = _computed_pm
        elif wmaint_pm == 0.0:
            # No PM rates configured — disable PM comparison
            pm_mode = 'Off'
    pm_blocked_dates = results.get("pm_blocked_dates", [])

    # Extract Benchmark (Standard Comparison or Custom)
    bench_series = results.get("bench_series", None)
    bench_stats = results.get("bench_stats")

    # Clip Benchmark if exists
    if bench_series is not None and analysis_start_date is not None:
         bench_series = _clip_with_return_anchor(bench_series, analysis_start_date)
         if not bench_series.empty:
             bench_stats = calculations.generate_stats(bench_series)

    logs = results.get("logs", [])

    # Calculate Tax Series for Margin Simulation (and Metrics)
    tax_payment_series = None
    total_tax_owed = 0.0

    # Initialize Tax Series (Defaults)
    fed_tax_series = pd.Series(dtype=float)
    state_tax_series = pd.Series(dtype=float)

    if not pl_by_year.empty:
        fed_tax_series, state_tax_series = _cached_tax_calculations(
            pl_by_year, other_income, filing_status, tax_method,
            use_std_deduction, state_code,
            retirement_income, _retirement_year, retirement_date,
        )
    else:
        # No realized P&L = No Tax
        fed_tax_series = pd.Series(0.0, index=[dt.date.today().year]) # Dummy index
        state_tax_series = pd.Series(0.0, index=[dt.date.today().year])

    total_tax_owed = fed_tax_series.sum() + state_tax_series.sum()

    # Create Payment Series (Unconditional for Sharpe Calc)
    tax_payment_series = pd.Series(0.0, index=port_series.index)
    annual_total_tax = fed_tax_series.add(state_tax_series, fill_value=0.0)

    for year, amount in annual_total_tax.items():
        if amount > 0:
            # Pay on April 15th of NEXT year
            pay_date = pd.Timestamp(year + 1, 4, 15)

            idx = port_series.index.searchsorted(pay_date)

            if idx < len(port_series.index):
                actual_date = port_series.index[idx]
                tax_payment_series[actual_date] += amount

    # Prepare Repayment Series (if Pay Down Margin is enabled)
    repayment_series = None
    if pay_down_margin and cashflow > 0:
        dates = port_series.index
        repayment_vals = pd.Series(0.0, index=dates)

        if cashfreq == "Monthly":
                months = dates.month
                changes = months != np.roll(months, -1)
                changes[-1] = False
                repayment_vals[changes] = cashflow
        elif cashfreq == "Quarterly":
                quarters = dates.quarter
                changes = quarters != np.roll(quarters, -1)
                changes[-1] = False
                repayment_vals[changes] = cashflow
        elif cashfreq == "Yearly":
                years = dates.year
                changes = years != np.roll(years, -1)
                changes[-1] = False
                repayment_vals[changes] = cashflow

        repayment_series = repayment_vals

    # Build DCA series for margin simulation
    # fund_dca_margin=True: DCA always adds to loan (margin-funded)
    # fund_dca_margin=False: DCA only depletes cash, then becomes "new money"
    dca_series = None
    if cashflow > 0 and not pay_down_margin and (fund_dca_margin or starting_cash > 0):
        dates = port_series.index
        dca_vals = pd.Series(0.0, index=dates)
        if cashfreq == "Monthly":
            months = dates.month
            changes = months != np.roll(months, -1)
            changes[-1] = False
            dca_vals[changes] = cashflow
        elif cashfreq == "Quarterly":
            quarters = dates.quarter
            changes = quarters != np.roll(quarters, -1)
            changes[-1] = False
            dca_vals[changes] = cashflow
        elif cashfreq == "Yearly":
            years = dates.year
            changes = years != np.roll(years, -1)
            changes[-1] = False
            dca_vals[changes] = cashflow
        # Apply retirement cutoff — stop DCA at draw_start_date or retirement_date
        if not dca_in_retirement:
            _dca_cutoff = draw_start_date or retirement_date
            if _dca_cutoff is not None:
                cutoff = pd.Timestamp(_dca_cutoff)
                dca_vals[dates >= cutoff] = 0.0
        dca_series = dca_vals

    # Store DCA sum for margin stats (total DCA funded via margin)
    config['_dca_series_sum'] = dca_series.sum() if dca_series is not None and fund_dca_margin else 0.0

    # Re-run margin sim
    sim_tax_series = tax_payment_series if pay_tax_margin else None

    _log = logging.getLogger("results")
    eff_loan = 0.0 if pay_tax_cash else starting_loan
    eff_rate = 0.0 if pay_tax_cash else rate_annual
    eff_draw = 0.0 if pay_tax_cash else draw_monthly
    eff_draw_ret = 0.0 if pay_tax_cash else draw_monthly_retirement
    _log.info("Margin sim inputs: loan=$%.0f draw=$%.0f/mo ret_draw=$%.0f/mo draw_start=%s ret_date=%s pay_tax_cash=%s",
              eff_loan, eff_draw, eff_draw_ret, draw_start_date, retirement_date, pay_tax_cash)

    loan_series, equity_series, equity_pct_series, usage_series, effective_rate_series = _cached_simulate_margin(
        port_series, eff_loan,
        eff_rate, eff_draw, wmaint,
        tax_series=sim_tax_series,
        repayment_series=repayment_series,
        draw_start_date=draw_start_date,
        draw_monthly_retirement=eff_draw_ret,
        retirement_date=retirement_date,
        dca_series=dca_series,
        fund_dca_margin=fund_dca_margin,
        accrual_start_date=original_start_date,
    )

    # Compute PM usage series post-hoc (loan is invariant to margin type)
    pm_usage_series = pd.Series(dtype=float)
    if pm_mode != 'Off' and wmaint_pm > 0 and not loan_series.empty:
        max_loan_pm = port_series * (1 - wmaint_pm)
        valid_pm = max_loan_pm > 0
        pm_usage_series = pd.Series(0.0, index=port_series.index)
        pm_usage_series[valid_pm] = loan_series[valid_pm] / max_loan_pm[valid_pm]
        pm_usage_series.name = "PM Usage %"

    # Update session state with latest margin results for reporting
    results.update({
        "loan_series": loan_series,
        "equity_series": equity_series,
        "usage_series": usage_series,
        "equity_pct_series": equity_pct_series,
        "effective_rate_series": effective_rate_series
    })


    # Calculate Tax-Adjusted Equity Curve (Global for Tabs)
    final_adj_series = pd.Series(dtype=float)
    final_tax_series = pd.Series(dtype=float) # Series of ACTUAL taxes paid (scaled if needed)

    if not equity_series.empty:
        if pay_tax_margin:
            final_adj_series = equity_series
            final_tax_series = tax_payment_series if tax_payment_series is not None else pd.Series(0.0, index=equity_series.index)
        elif pay_tax_cash:
            if tax_payment_series is not None and tax_payment_series.sum() > 0:
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=draw_monthly, draw_start_date=draw_start_date, draw_monthly_retirement=draw_monthly_retirement, retirement_date=retirement_date
                )
            else:
                empty_tax = pd.Series(0.0, index=equity_series.index)
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly, draw_start_date=draw_start_date, draw_monthly_retirement=draw_monthly_retirement, retirement_date=retirement_date
                )
        else: # None (Gross)
            final_adj_series = equity_series
            final_tax_series = pd.Series(0.0, index=equity_series.index)

    # --- Prepare Tax-Adjusted Data for Charts ---
    tax_adj_port_series = final_adj_series + loan_series

    # Use clipped bench_series if already set above, otherwise retrieve from results
    if bench_series is None:
        bench_series = results.get("bench_series")
    bench_resampled = None
    bench_aligned = None
    if bench_series is not None:
            bench_aligned = bench_series.reindex(tax_adj_port_series.index).ffill().bfill()
            bench_resampled = utils.resample_data(bench_aligned, timeframe, method="last")

    # Retrieve Comparison Benchmark (Standard Rebalance) if available
    comp_series = results.get("comparison_series")
    comp_resampled = None
    comp_aligned = None
    if comp_series is not None:
             comp_aligned = comp_series.reindex(tax_adj_port_series.index).ffill().bfill()
             comp_resampled = utils.resample_data(comp_aligned, timeframe, method="last")
             if comp_series.name is None:
                 comp_resampled.name = "Standard (Yearly)"
             else:
                 comp_resampled.name = comp_series.name

    # Recalculate Leverage Metrics based on Tax-Adjusted Equity
    tax_adj_equity_pct_series = pd.Series(0.0, index=tax_adj_port_series.index)
    valid_idx = tax_adj_port_series > 0
    tax_adj_equity_pct_series[valid_idx] = final_adj_series[valid_idx] / tax_adj_port_series[valid_idx]

    tax_adj_usage_series = pd.Series(0.0, index=tax_adj_port_series.index)
    if wmaint < 1.0:
        max_loan_series = tax_adj_port_series * (1 - wmaint)
        valid_loan = max_loan_series > 0
        tax_adj_usage_series[valid_loan] = loan_series[valid_loan] / max_loan_series[valid_loan]

    # PM tax-adjusted usage
    tax_adj_pm_usage_series = pd.Series(dtype=float)
    if pm_mode != 'Off' and wmaint_pm > 0:
        tax_adj_pm_usage_series = pd.Series(0.0, index=tax_adj_port_series.index)
        max_loan_pm = tax_adj_port_series * (1 - wmaint_pm)
        valid_pm = max_loan_pm > 0
        tax_adj_pm_usage_series[valid_pm] = loan_series[valid_pm] / max_loan_pm[valid_pm]

    ohlc_data, equity_resampled, loan_resampled, usage_resampled, equity_pct_resampled, effective_rate_resampled = _cached_resample_all(
        tax_adj_port_series, final_adj_series, loan_series,
        tax_adj_usage_series, tax_adj_equity_pct_series, effective_rate_series,
        timeframe,
    )



    # Update Range Caption to reflect actual displayed data
    if not port_series.empty:
        start_str = pd.Timestamp(analysis_start_date).strftime('%Y-%m-%d')
        end_str = port_series.index[-1].strftime('%Y-%m-%d')
        display_range = f"{start_str} to {end_str} (Synced)"
    else:
        display_range = results.get('sim_range', 'N/A')

    st.caption(f"📅 **Backtest Range:** {display_range}")
    m1, m2, m3, m4, m5 = st.columns(5)

    # Comparison Logic
    if bench_series is not None:
         try:
             aligned_pf = tax_adj_port_series.reindex(bench_series.index).ffill()
             diff_val = aligned_pf.iloc[-1] - bench_series.iloc[-1]
             diff_pct = (diff_val / bench_series.iloc[-1]) * 100
             st.info(f"**Comparison Active**: {portfolio_name} vs {bench_series.name if bench_series.name else 'Benchmark'} | Diff: ${diff_val:,.2f} ({diff_pct:+.2f}%)")
         except (IndexError, KeyError, ZeroDivisionError, TypeError):
             pass

    total_return = (tax_adj_port_series.iloc[-1] / start_val - 1) * 100

    # Use Stats reported by Testfol API (TWR)
    cagr = stats.get("cagr", 0.0)
    sharpe = stats.get("sharpe", 0.0)

    # Recalculate max drawdown from the displayed (clipped) series so it matches the chart
    max_dd_date = None
    ath_date = None
    current_dd_from_ath = 0.0
    avg_dd = 0.0
    med_dd = 0.0
    if not port_series.empty:
        _ps = port_series.astype(float)
        _dd_series = _ps / _ps.cummax() - 1.0
        max_dd = float(_dd_series.min()) * 100
        max_dd_date = _dd_series.idxmin()
        current_dd_from_ath = float(_dd_series.iloc[-1]) * 100
        ath_date = _ps.idxmax()
        _dd_negative = _dd_series[_dd_series < 0]
        avg_dd = float(_dd_negative.mean()) * 100 if not _dd_negative.empty else 0.0
        med_dd = float(_dd_negative.median()) * 100 if not _dd_negative.empty else 0.0
    else:
        max_dd = stats.get("max_drawdown", 0.0)

    # Retrieve Benchmark Stats (use clipped stats if already computed above)
    if bench_stats is None:
        bench_stats = results.get("bench_stats")

    # Calculate CAGR display value
    cagr_display = cagr * 100 if abs(cagr) <= 1 else cagr

    # Calculate Tax-Adjusted metrics
    if start_val > 0 and not port_series.empty:
        final_adj_val = final_adj_series.iloc[-1]
        start_equity = final_adj_series.iloc[0]
        days = (final_adj_series.index[-1] - final_adj_series.index[0]).days
        if days > 0 and start_equity > 0:
            years = days / 365.25
            tax_adj_cagr = (final_adj_val / start_equity) ** (1 / years) - 1
        else:
            tax_adj_cagr = 0.0
        tax_adj_sharpe = calculations.calculate_sharpe_ratio(final_adj_series)
    else:
        final_adj_val = 0.0
        tax_adj_cagr = 0.0
        tax_adj_sharpe = 0.0

    # Calculate Differences if Benchmark Exists
    active_bench_series = bench_series if bench_series is not None else results.get("comparison_series")
    active_bench_stats = bench_stats if bench_stats is not None else results.get("comparison_stats")

    if active_bench_stats:
        b_cagr = active_bench_stats.get("cagr", 0.0)
        b_sharpe = active_bench_stats.get("sharpe", 0.0)
        b_dd = active_bench_stats.get("max_drawdown", 0.0)

        b_cagr_display = b_cagr * 100 if abs(b_cagr) <= 1 else b_cagr
        diff_cagr_display = cagr_display - b_cagr_display
        diff_sharpe = sharpe - b_sharpe
        diff_dd = max_dd - b_dd

        bench_label = "Bench"
        if active_bench_series is not None and hasattr(active_bench_series, 'name') and active_bench_series.name:
            bench_label = active_bench_series.name.replace("Benchmark", "").replace("Standard", "").strip(" ()")
            if not bench_label: bench_label = "Bench"

        # Primary Metrics Row - Gross (Pre-Tax) from API
        m1.metric("Portfolio Value", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("Gross CAGR", f"{cagr_display:.2f}%", f"{diff_cagr_display:+.2f}% vs {bench_label}", help="Pre-tax return from Testfol API")
        m3.metric("Gross Sharpe", f"{sharpe:.2f}", f"{diff_sharpe:+.2f} vs {bench_label}", help="Pre-tax risk-adjusted return")
        _dd_help = f"Trough: {max_dd_date.strftime('%b %d, %Y')}" if max_dd_date is not None else None
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", f"{diff_dd:+.2f}% vs {bench_label}", delta_color="inverse", help=_dd_help)
        _dd_parts = []
        if max_dd_date is not None:
            _dd_parts.append(max_dd_date.strftime('%b %Y'))
        if avg_dd != 0.0:
            _dd_parts.append(f"Avg {avg_dd:.1f}% · Med {med_dd:.1f}%")
        if current_dd_from_ath != 0.0:
            _dd_parts.append(f"Now {current_dd_from_ath:.1f}% (peak {ath_date.strftime('%b %Y')})")
        else:
            _dd_parts.append("Now ATH")
        m4.caption(" · ".join(_dd_parts))
        if not final_adj_series.empty and final_adj_series.iloc[-1] != 0:
            m5.metric("Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")
        else:
            m5.metric("Leverage", "N/A")

        # Detailed Comparison Table
        bench_end_val = active_bench_series.iloc[-1] if active_bench_series is not None and not active_bench_series.empty else 0
        strategy_end_val = tax_adj_port_series.iloc[-1]

        comp_label = active_bench_series.name if active_bench_series is not None and hasattr(active_bench_series, 'name') and active_bench_series.name else "Benchmark"

        comp_data = {
            "Metric": ["Ending Value", "CAGR", "Sharpe", "Max Drawdown", "Std Dev"],
            portfolio_name: [f"${strategy_end_val:,.0f}", f"{cagr_display:.2f}%", f"{sharpe:.2f}", f"{max_dd:.2f}%", f"{stats.get('std', stats.get('volatility',0))*100:.2f}%"],
            comp_label: [f"${bench_end_val:,.0f}", f"{b_cagr_display:.2f}%", f"{b_sharpe:.2f}", f"{b_dd:.2f}%", f"{active_bench_stats.get('std', active_bench_stats.get('volatility',0))*100:.2f}%"],
            "Diff": [f"${strategy_end_val - bench_end_val:+,.0f}", f"{diff_cagr_display:+.2f}%", f"{diff_sharpe:+.2f}", f"{diff_dd:+.2f}%", ""]
        }
        comp_df = pd.DataFrame(comp_data)
        with st.expander(f"📊 Detailed {portfolio_name} vs {comp_label} Statistics", expanded=False):
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    else:
        # Standard View (No Benchmark) - Gross (Pre-Tax) from API
        m1.metric("Portfolio Value", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("Gross CAGR", f"{cagr_display:.2f}%", help="Pre-tax return from Testfol API")
        m3.metric("Gross Sharpe", f"{sharpe:.2f}", help="Pre-tax risk-adjusted return")
        _dd_help2 = f"Trough: {max_dd_date.strftime('%b %d, %Y')}" if max_dd_date is not None else None
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse", help=_dd_help2)
        _dd_parts2 = []
        if max_dd_date is not None:
            _dd_parts2.append(max_dd_date.strftime('%b %Y'))
        if avg_dd != 0.0:
            _dd_parts2.append(f"Avg {avg_dd:.1f}% · Med {med_dd:.1f}%")
        if current_dd_from_ath != 0.0:
            _dd_parts2.append(f"Now {current_dd_from_ath:.1f}% (peak {ath_date.strftime('%b %Y')})")
        else:
            _dd_parts2.append("Now ATH")
        m4.caption(" · ".join(_dd_parts2))
        if not final_adj_series.empty and final_adj_series.iloc[-1] != 0:
            m5.metric("Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")
        else:
            m5.metric("Leverage", "N/A")

    # --- Secondary Metrics in Expander (Only show when tax simulation is active) ---
    tax_sim_active = pay_tax_margin or pay_tax_cash

    if tax_sim_active:
        with st.expander("💰 Tax & Post-Tax Details", expanded=False):
            st.caption(f"📅 **Shadow Data Range:** {results.get('shadow_range', 'N/A')}")

            sm1, sm2, sm3, sm4 = st.columns(4)

            # Tax Info
            tax_label = "Total Tax Paid" if pay_tax_margin else "Est. Tax Owed"
            if total_tax_owed > 0:
                if not final_tax_series.empty and final_tax_series.sum() > 0:
                    display_tax = final_tax_series.sum()
                else:
                    display_tax = total_tax_owed
            else:
                display_tax = 0.0

            sm1.metric(tax_label, f"${display_tax:,.0f}")
            sm2.metric("Post-Tax Equity", f"${final_adj_val:,.0f}", help="Equity after tax simulation")
            sm3.metric("Post-Tax CAGR", f"{tax_adj_cagr * 100:.2f}%", help="Growth rate after taxes")
            sm4.metric("Post-Tax Sharpe", f"{tax_adj_sharpe:.2f}", help="Risk-adjusted return after taxes")

            unpaid_liability = total_tax_owed - display_tax
            if unpaid_liability > 1:
                st.caption(f"ℹ️ **Timing Difference:** Total Tax Paid (\${display_tax:,.0f}) is lower than Total Tax Owed (\${total_tax_owed:,.0f}) because taxes are typically paid on **April 15th of the following year**. The tax bill for the final simulation year (\${unpaid_liability:,.0f}) is technically owed (Accrued) but the payment date falls **after** the simulation ends, so it was never deducted from your cash.")

    st.markdown("---")

    # =========================================================================
    # Results View
    # =========================================================================
    results_views = ["📈 Chart", "📊 Returns Analysis", "⚖️ Rebalancing", "💸 Tax Analysis", "🔍 X-Ray", "🔮 Monte Carlo", "🔧 Debug", "💰 Withdrawals"]
    selected_results_view = st.segmented_control(
        "Results View",
        results_views,
        default=results_views[0],
        key=f"results_view_{portfolio_name or 'default'}",
        label_visibility="collapsed",
    )

    if selected_results_view == "📈 Chart":
        render_chart_tab(
            st.container(),
            chart_style=chart_style,
            tax_adj_port_series=tax_adj_port_series,
            final_adj_series=final_adj_series,
            loan_series=loan_series,
            tax_adj_equity_pct_series=tax_adj_equity_pct_series,
            tax_adj_usage_series=tax_adj_usage_series,
            equity_series=equity_series,
            usage_series=usage_series,
            equity_pct_series=equity_pct_series,
            effective_rate_series=effective_rate_series,
            ohlc_data=ohlc_data,
            equity_resampled=equity_resampled,
            loan_resampled=loan_resampled,
            usage_resampled=usage_resampled,
            equity_pct_resampled=equity_pct_resampled,
            effective_rate_resampled=effective_rate_resampled,
            bench_resampled=bench_resampled,
            comp_resampled=comp_resampled,
            port_series=port_series,
            component_prices=component_prices,
            portfolio_name=portfolio_name,
            log_scale=log_scale,
            show_range_slider=show_range_slider,
            show_volume=show_volume,
            timeframe=timeframe,
            wmaint=wmaint,
            stats=stats,
            config=config,
            pay_tax_cash=pay_tax_cash,
            draw_monthly=draw_monthly,
            draw_monthly_retirement=draw_monthly_retirement,
            draw_start_date=draw_start_date,
            retirement_date=retirement_date,
            logs=logs,
            final_tax_series=final_tax_series,
            tax_payment_series=tax_payment_series,
            start_val=start_val,
            rate_annual=rate_annual,
            pm_enabled=pm_enabled,
            pm_mode=pm_mode,
            pm_usage_series=tax_adj_pm_usage_series,
            wmaint_pm=wmaint_pm,
            pm_threshold=pm_threshold,
            pm_blocked_dates=pm_blocked_dates,
        )

    elif selected_results_view == "📊 Returns Analysis":
        if pay_tax_cash:
            st.info("ℹ️ **Note:** Returns are **Net of Tax** (simulated as cash withdrawals).")
        elif pay_tax_margin:
            st.info("ℹ️ **Note:** Returns are **Gross** (taxes paid via margin loan).")
        else:
            st.info("ℹ️ **Note:** Returns are **Gross** (Pre-Tax).")

        # Compute fresh per-year returns for drift-free yearly column
        fresh_yearly, fresh_series = {}, None
        _alloc, _maint, _rebal = _extract_backtest_params(results, config)
        if _alloc and _maint is not None:
            _reb_mode = (_rebal or {}).get("mode", "Standard")
            if _reb_mode != "None":
                fresh_yearly, fresh_series = _compute_fresh_yearly_returns(
                    allocation=_alloc, maint_pcts=_maint, rebalance=_rebal,
                    series_start=pd.Timestamp(analysis_start_date).strftime("%Y-%m-%d"),
                    series_end=tax_adj_port_series.index[-1].strftime("%Y-%m-%d"),
                    series_start_val=float(tax_adj_port_series.iloc[0]),
                    year_list=tuple(range(
                        pd.Timestamp(analysis_start_date).year,
                        tax_adj_port_series.index[-1].year + 1,
                    )),
                )

        charts.render_returns_analysis(
            tax_adj_port_series,
            bench_series=bench_aligned if bench_series is not None else None,
            comparison_series=comp_aligned if comp_series is not None else None,
            unique_id=portfolio_name,
            portfolio_name=portfolio_name,
            component_data=component_prices,
            raw_port_series=port_series,
            stats=stats,
            raw_response=raw_response,
            fresh_yearly=fresh_yearly,
            fresh_series=fresh_series,
            allocation=_alloc,
            composition_df=composition_df,
            analysis_start_date=analysis_start_date,
        )

    elif selected_results_view == "⚖️ Rebalancing":
        st.warning("⚠️ **Note:** These trade calculations assume **Gross** portfolio values. Tax payments are NOT deducted by selling shares in this view (assumes taxes paid via margin or external cash).")
        rebal_freq_for_chart = config.get('rebalance', 'Yearly')

        if not composition_df.empty:
            composition_df = composition_df.copy()
            for d in composition_df["Date"].unique():
                if d in tax_adj_port_series.index:
                    target_total = tax_adj_port_series.loc[d]
                    current_total = composition_df[composition_df["Date"] == d]["Value"].sum()
                    if current_total > 0:
                        ratio = target_total / current_total
                        composition_df.loc[composition_df["Date"] == d, "Value"] *= ratio

        # Clip component prices to match the common start date
        alloc_component_prices = component_prices
        if not component_prices.empty and not port_series.empty:
            alloc_component_prices = component_prices[component_prices.index >= port_series.index[0]]

        charts.render_rebalancing_analysis(
            trades_df, pl_by_year, composition_df,
            tax_method, other_income, filing_status, state_code,
            rebalance_freq=rebal_freq_for_chart,
            use_standard_deduction=use_std_deduction,
            unrealized_pl_df=results.get("unrealized_pl_df", pd.DataFrame()),
            custom_freq=config.get('custom_freq', 'Yearly'),
            unique_id=portfolio_name,
            component_prices=alloc_component_prices,
            allocation=results.get("allocation", {}),
            start_val=start_val,
            retirement_income=retirement_income,
            retirement_year=_retirement_year,
            port_series=port_series,
            rebal_config=results.get("_reb", {}),
        )

    elif selected_results_view == "💸 Tax Analysis":
        tax_container = st.container()
        render_tax_impact_tab(
            tax_container,
            pl_by_year=pl_by_year,
            config=config,
            port_series=port_series,
            equity_series=equity_series,
            loan_series=loan_series,
            final_adj_series=final_adj_series,
            annual_total_tax=annual_total_tax,
            tax_payment_series=tax_payment_series,
            pay_tax_margin=pay_tax_margin,
            pay_tax_cash=pay_tax_cash,
            rate_annual=rate_annual,
            draw_monthly=draw_monthly,
            starting_loan=starting_loan,
            wmaint=wmaint,
            repayment_series=repayment_series,
            twr_series=results.get("twr_series"),
            stats=stats,
            draw_start_date=draw_start_date,
            accrual_start_date=original_start_date,
        )
        with tax_container:
            charts.render_tax_analysis(
                pl_by_year, other_income, filing_status, state_code,
                tax_method=tax_method,
                use_standard_deduction=use_std_deduction,
                unrealized_pl_df=results.get("unrealized_pl_df", pd.DataFrame()),
                trades_df=trades_df,
                pay_tax_cash=pay_tax_cash,
                pay_tax_margin=pay_tax_margin,
                retirement_income=retirement_income,
                retirement_year=_retirement_year,
            )

    elif selected_results_view == "🔍 X-Ray":
        if not composition_df.empty:
            latest_date = composition_df['Date'].max()
            latest_df = composition_df[composition_df['Date'] == latest_date]

            total_val = latest_df['Value'].sum()
            if total_val > 0:
                alloc_map = dict(zip(latest_df['Ticker'], latest_df['Value'] / total_val))
                xray_view.render_xray(alloc_map, portfolio_name=portfolio_name)
            else:
                st.info("No allocation data found for X-Ray.")
        else:
            st.info("No composition data available for X-Ray.")

    elif selected_results_view == "🔮 Monte Carlo":
        extended_series = results.get("port_series")
        tax_twr_series = results.get("twr_series")

        daily_rets = pd.Series(dtype=float)
        source_label = "Unknown"
        use_extended = False

        if extended_series is not None and not extended_series.empty:
            if tax_twr_series is not None and not tax_twr_series.empty:
                if extended_series.index[0] < tax_twr_series.index[0]:
                    use_extended = True
                    source_label = "Extended Chart Data (Simulated)"
                else:
                    daily_rets = tax_twr_series.pct_change()
                    source_label = "Tax TWR Data (Real)"
            else:
                use_extended = True
                source_label = "Extended Chart Data (Simulated)"
        elif tax_twr_series is not None and not tax_twr_series.empty:
            daily_rets = tax_twr_series.pct_change()
            source_label = "Tax TWR Data (Real)"

        if use_extended:
            daily_rets = extended_series.pct_change()

        render_monte_carlo_tab(
            st.container(),
            results=results,
            config=config,
            portfolio_name=portfolio_name,
            daily_rets=daily_rets,
            source_label=source_label,
        )

    elif selected_results_view == "🔧 Debug":
        render_debug_tab(st.container(), logs, raw_response, portfolio_name)

    else:
        from app.ui.results.tabs_withdrawals import render_withdrawals_tab
        render_withdrawals_tab(st.container(), logs, draw_monthly, draw_start_date, loan_series=loan_series)

    # -------------------------------------------------------------------------
    # Post-Tabs (Charts Controls & Report)
    # -------------------------------------------------------------------------
    st.divider()
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])

    # Report Generation (Sidebar)
    if results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Report")

        try:
            report_data = st.session_state.get('results_list', results)
            if isinstance(report_data, list) and not report_data:
                report_data = results

            report_html = report_generator.generate_html_report(report_data)

            btn_label = "Download Combined Report (HTML)" if isinstance(report_data, list) and len(report_data) > 1 else "Download HTML Report"

            st.sidebar.download_button(
                label=btn_label,
                data=report_html,
                file_name=f"testfol_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                key=f"dl_report_{portfolio_name}",
                help="Download a standalone HTML report with charts and stats."
            )
        except Exception as e:
            st.sidebar.error(f"Report Gen Failed: {e}")
