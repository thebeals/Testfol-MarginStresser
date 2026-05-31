"""Chart tab with margin calcs, MA analysis, and cheat sheet subtabs."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from app.ui import charts
from app.ui.results.tabs_withdrawals import _parse_events


def render_chart_tab(
    tab,
    chart_style: str,
    tax_adj_port_series: pd.Series,
    final_adj_series: pd.Series,
    loan_series: pd.Series,
    tax_adj_equity_pct_series: pd.Series,
    tax_adj_usage_series: pd.Series,
    equity_series: pd.Series,
    usage_series: pd.Series,
    equity_pct_series: pd.Series,
    effective_rate_series: pd.Series,
    ohlc_data,
    equity_resampled: pd.Series,
    loan_resampled: pd.Series,
    usage_resampled: pd.Series,
    equity_pct_resampled: pd.Series,
    effective_rate_resampled: pd.Series,
    bench_resampled: pd.Series | None,
    comp_resampled: pd.Series | None,
    port_series: pd.Series,
    component_prices: pd.DataFrame,
    portfolio_name: str,
    log_scale: bool,
    show_range_slider: bool,
    show_volume: bool,
    timeframe: str,
    wmaint: float,
    stats: dict,
    config: dict,
    pay_tax_cash: bool,
    draw_monthly: float,
    draw_monthly_retirement: float,
    draw_start_date,
    retirement_date,
    logs: list | None,
    final_tax_series: pd.Series,
    tax_payment_series,
    start_val: float,
    rate_annual,
    pm_enabled: bool,
    pm_mode: str = "Off",
    pm_usage_series: pd.Series | None = None,
    wmaint_pm: float = 0.0,
    pm_threshold: float = 110000.0,
    pm_blocked_dates: list | None = None,
) -> None:
    with tab:
        chart_views = ["🧮 Margin Calcs", "📉 200DMA", "📉 150MA", "📊 Munger200WMA", "📜 Cheat Sheet"]
        selected_view = st.segmented_control(
            "Chart View",
            chart_views,
            default=chart_views[0],
            key=f"chart_view_{portfolio_name}",
            label_visibility="collapsed",
        )

        if selected_view == chart_views[0]:
            draw_events_df = _parse_events(logs) if logs else pd.DataFrame()
            draw_dates = list(draw_events_df.loc[draw_events_df["Type"] == "Draw", "Date"]) if not draw_events_df.empty else []
            ret_draw_dates = list(draw_events_df.loc[draw_events_df["Type"] == "RetDraw", "Date"]) if not draw_events_df.empty else []

            if chart_style == "Classic (Combined)":
                series_opts = ["Portfolio", "Equity", "Loan", "Margin usage %"]
                charts.render_classic_chart(
                    tax_adj_port_series, final_adj_series, loan_series,
                    tax_adj_equity_pct_series, tax_adj_usage_series,
                    series_opts, log_scale,
                    bench_series=bench_resampled,
                    comparison_series=comp_resampled,
                    effective_rate_series=effective_rate_resampled,
                    pm_usage_series=pm_usage_series if pm_mode != 'Off' else None,
                    pm_mode=pm_mode,
                    pm_blocked_dates=pm_blocked_dates,
                    draw_start_date=draw_start_date,
                    draw_monthly=draw_monthly,
                    draw_monthly_retirement=draw_monthly_retirement,
                    retirement_date=retirement_date,
                    draw_dates=draw_dates,
                    ret_draw_dates=ret_draw_dates,
                    tax_payment_series=tax_payment_series,
                )
            else:  # Candlestick
                charts.render_candlestick_chart(
                    ohlc_data,
                    equity_resampled,
                    loan_resampled,
                    usage_resampled,
                    equity_pct_resampled,
                    timeframe,
                    log_scale,
                    show_range_slider=show_range_slider,
                    show_volume=show_volume,
                    bench_series=bench_resampled,
                    comparison_series=comp_resampled,
                    effective_rate_series=effective_rate_resampled,
                    pm_usage_series=pm_usage_series if pm_mode != 'Off' else None,
                    pm_mode=pm_mode,
                    daily_close=port_series,
                )

            if pay_tax_cash:
                _render_cash_statistics(final_adj_series, final_tax_series, draw_monthly, equity_series, draw_start_date=draw_start_date)
            else:
                _gcf = config.get('global_cashflow', {})
                _fund_dca = _gcf.get('fund_dca_margin', False)
                _pay_down = _gcf.get('pay_down_margin', False)
                _dca_total = 0.0
                if _fund_dca and not _pay_down:
                    _dca_total = config.get('_dca_series_sum', 0.0)
                _render_margin_statistics(
                    tax_adj_port_series, final_adj_series, loan_series,
                    equity_series, usage_series, equity_pct_series,
                    final_tax_series, draw_monthly, wmaint, pm_enabled,
                    draw_start_date=draw_start_date,
                    pm_mode=pm_mode,
                    pm_usage_series=pm_usage_series,
                    wmaint_pm=wmaint_pm,
                    pm_threshold=pm_threshold,
                    pm_blocked_dates=pm_blocked_dates,
                    total_dca_margin=_dca_total,
                )
        elif selected_view == chart_views[1]:
            charts.render_ma_analysis_tab(port_series, portfolio_name, portfolio_name, window=200, show_stage_analysis=False)
        elif selected_view == chart_views[2]:
            charts.render_ma_analysis_tab(port_series, portfolio_name, portfolio_name, window=150, show_stage_analysis=True)
        elif selected_view == chart_views[3]:
            charts.render_munger_wma_tab(port_series, portfolio_name, portfolio_name, window=200)
        else:
            charts.render_cheat_sheet(
                port_series,
                portfolio_name,
                portfolio_name,
                component_data=component_prices
            )


def _render_cash_statistics(
    final_adj_series: pd.Series,
    final_tax_series: pd.Series,
    draw_monthly: float,
    equity_series: pd.Series,
    draw_start_date=None,
) -> None:
    with st.expander("Detailed Cash Statistics", expanded=True):
        total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0

        total_draws = 0.0
        if draw_monthly > 0 and not equity_series.empty:
            _ds = pd.Timestamp(draw_start_date) if draw_start_date is not None else None
            prev_m = equity_series.index[0].month
            for d in equity_series.index[1:]:
                if d.month != prev_m:
                    if _ds is None or d >= _ds:
                        total_draws += draw_monthly
                    prev_m = d.month

        total_withdrawn = total_tax_paid + total_draws

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Balance", f"${final_adj_series.iloc[-1]:,.2f}")
        c2.metric("Total Withdrawn", f"${total_withdrawn:,.2f}", help="Includes Taxes + Monthly Draws")
        c3.metric("Total Tax Paid", f"${total_tax_paid:,.2f}")

        st.markdown("##### Withdrawal Breakdown")
        import pandas as _pd
        w_df = _pd.DataFrame({
            "Category": ["Monthly Draws", "Taxes Paid"],
            "Amount": [total_draws, total_tax_paid]
        })
        st.dataframe(w_df.style.format({"Amount": "${:,.2f}"}), use_container_width=True, hide_index=True)


def _render_margin_statistics(
    tax_adj_port_series: pd.Series,
    final_adj_series: pd.Series,
    loan_series: pd.Series,
    equity_series: pd.Series,
    usage_series: pd.Series,
    equity_pct_series: pd.Series,
    final_tax_series: pd.Series,
    draw_monthly: float,
    wmaint: float,
    pm_enabled: bool,
    draw_start_date=None,
    pm_mode: str = "Off",
    pm_usage_series: pd.Series | None = None,
    wmaint_pm: float = 0.0,
    pm_threshold: float = 110000.0,
    pm_blocked_dates: list | None = None,
    total_dca_margin: float = 0.0,
) -> None:
    with st.expander("Detailed Margin Statistics", expanded=True):
        if usage_series.empty:
            return

        max_usage_idx = usage_series.idxmax()
        max_usage_val = usage_series.max()
        equity_at_max = equity_series.loc[max_usage_idx]

        min_equity_idx = final_adj_series.idxmin()
        min_equity_val = final_adj_series.min()

        total_draws = 0.0
        if draw_monthly > 0 and not equity_series.empty:
            _ds = pd.Timestamp(draw_start_date) if draw_start_date is not None else None
            prev_m = equity_series.index[0].month
            for d in equity_series.index[1:]:
                if d.month != prev_m:
                    if _ds is None or d >= _ds:
                        total_draws += draw_monthly
                    prev_m = d.month

        total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0
        start_loan_val = loan_series.iloc[0]
        final_loan_val = loan_series.iloc[-1]
        total_interest = (final_loan_val - start_loan_val) - total_draws - total_tax_paid - total_dca_margin

        avg_usage = usage_series.mean()
        current_usage = usage_series.iloc[-1]
        safety_buffer = 1.0 - current_usage

        current_port_val = tax_adj_port_series.iloc[-1]
        max_loan_allowed = current_port_val * (1 - wmaint)
        avail_withdraw = max_loan_allowed - final_loan_val

        # Row 1: Current Status
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Equity", f"${final_adj_series.iloc[-1]:,.2f}")
        c2.metric("Final Loan", f"${loan_series.iloc[-1]:,.2f}")
        c3.metric("Final Usage", f"{usage_series.iloc[-1]*100:.2f}%")

        # Row 2: Extremes & Capacity
        c4, c5, c6 = st.columns(3)
        c4.metric(
            "Lowest Equity",
            f"${min_equity_val:,.2f}",
            f"{min_equity_idx.date()}",
            delta_color="off"
        )
        c5.metric(
            "Avail. to Withdraw",
            f"${avail_withdraw:,.2f}",
            help="Additional amount that can be drawn before reaching maintenance threshold"
        )
        c6.metric(
            "Max Margin Usage",
            f"{max_usage_val*100:.2f}%",
            f"{max_usage_idx.date()} (Eq: ${equity_at_max:,.0f})",
            delta_color="inverse"
        )

        st.markdown("##### Survival Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Interest Paid", f"${total_interest:,.2f}", help="Cost of Leverage")
        m2.metric("Avg Margin Usage", f"{avg_usage*100:.2f}%", help="Chronic Stress Level")
        m3.metric("Safety Buffer", f"{safety_buffer*100:.2f}%", help="% Market Drop allowed before Margin Call")

        # PM Metrics (side by side with Reg-T)
        if pm_mode != 'Off' and pm_usage_series is not None and not pm_usage_series.empty:
            st.markdown("##### Portfolio Margin (PM) Metrics")
            pm1, pm2, pm3 = st.columns(3)
            pm_final = pm_usage_series.iloc[-1]
            pm_max_val = pm_usage_series.max()
            pm_max_idx = pm_usage_series.idxmax()
            pm_safety = 1.0 - pm_final
            pm_max_loan_allowed = current_port_val * (1 - wmaint_pm) if wmaint_pm > 0 else 0
            pm_avail = pm_max_loan_allowed - final_loan_val if pm_max_loan_allowed > 0 else 0

            pm1.metric("Final PM Usage", f"{pm_final*100:.2f}%")
            pm2.metric("Max PM Usage", f"{pm_max_val*100:.2f}%", f"{pm_max_idx.date()}", delta_color="off")
            pm3.metric("PM Safety Buffer", f"{pm_safety*100:.2f}%")

        # Standard Margin Breaches (Usage >= 100%)
        breach_episodes = []

        def _group_episodes(mask, usage_src, eq_pct_src, breach_type):
            """Group consecutive True days into episodes with summary stats."""
            if not mask.any():
                return
            dates = mask.index[mask]
            # Split into contiguous runs (gap > 3 trading days = new episode)
            groups, current = [], [dates[0]]
            for d in dates[1:]:
                if (d - current[-1]).days <= 5:  # allow weekends/holidays
                    current.append(d)
                else:
                    groups.append(current)
                    current = [d]
            groups.append(current)

            for grp in groups:
                start, end = grp[0], grp[-1]
                usage_slice = usage_src.loc[grp] * 100
                eq_slice = eq_pct_src.loc[grp] * 100
                breach_episodes.append({
                    "Start": start.strftime("%Y-%m-%d"),
                    "End": end.strftime("%Y-%m-%d"),
                    "Days": len(grp),
                    "Peak Usage %": round(usage_slice.max(), 1),
                    "Min Equity %": round(eq_slice.min(), 1),
                    "Type": breach_type,
                })

        reg_t_mask = usage_series >= 1
        _group_episodes(reg_t_mask, usage_series, equity_pct_series, "Reg-T Call")

        if pm_mode != 'Off' and pm_usage_series is not None and not pm_usage_series.empty:
            pm_call_mask = pm_usage_series >= 1.0
            _group_episodes(pm_call_mask, pm_usage_series, equity_pct_series, "PM Call")

        if pm_enabled or pm_mode != 'Off':
            pm_eq_mask = equity_series < 100000
            if pm_eq_mask.any():
                _group_episodes(pm_eq_mask, usage_series, equity_pct_series, "PM Min Eq < $100k")

        st.markdown("##### Margin & PM Breaches")
        if not breach_episodes:
            st.success("No breaches triggered! 🎉")
        else:
            episodes_df = pd.DataFrame(breach_episodes)

            reg_t_eps = episodes_df[episodes_df["Type"] == "Reg-T Call"]
            pm_eps = episodes_df[episodes_df["Type"] != "Reg-T Call"]
            reg_t_days = int(reg_t_eps["Days"].sum()) if not reg_t_eps.empty else 0
            pm_days = int(pm_eps["Days"].sum()) if not pm_eps.empty else 0

            msg_parts = []
            if not reg_t_eps.empty:
                msg_parts.append(f"{len(reg_t_eps)} Margin Call episode(s) ({reg_t_days} days)")
            if not pm_eps.empty:
                msg_parts.append(f"{len(pm_eps)} PM Breach episode(s) ({pm_days} days)")

            full_msg = f"⚠️ {' + '.join(msg_parts)}"

            if not reg_t_eps.empty:
                st.error(full_msg)
            else:
                st.warning(full_msg)

            st.dataframe(episodes_df, use_container_width=True, hide_index=True)

        # PM Buy-Blocked Dates
        if pm_blocked_dates:
            st.markdown("##### PM Buy-Blocked Rebalance Dates")
            st.warning(f"{len(pm_blocked_dates)} rebalance(s) had buys blocked due to PM equity < ${pm_threshold:,.0f}")
            blocked_df = pd.DataFrame({"Date": [d.date() if hasattr(d, 'date') else d for d in pm_blocked_dates]})
            st.dataframe(blocked_df, use_container_width=True, hide_index=True)
