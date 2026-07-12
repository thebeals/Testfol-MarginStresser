import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots

from app.core import calculations

# --- Multi-Portfolio Chart ---
def _format_markdown_currency(value):
    """Format currency for Streamlit Markdown without triggering math rendering."""
    return f"\\${value:,.0f}"


def _format_cashflow_caption(cashflow_config):
    """Return a short chart caption fragment for active DCA cashflows."""
    cfg = cashflow_config or {}
    try:
        amount = float(cfg.get("amount", cfg.get("cashflow", 0.0)) or 0.0)
    except (TypeError, ValueError):
        amount = 0.0

    if amount <= 0:
        return ""

    freq = str(cfg.get("freq", cfg.get("cashflow_freq", cfg.get("cashfreq", "Monthly"))) or "Monthly")
    label = "Margin repayment" if cfg.get("pay_down_margin", False) else "DCA"
    suffix = " (margin-funded)" if label == "DCA" and cfg.get("fund_dca_margin", False) else ""
    return f"{label}: {_format_markdown_currency(amount)} {freq}{suffix}."


def render_multi_portfolio_chart(results_list, benchmarks=[], log_scale=True, cashflow_config=None):
    """
    Renders a performance chart for multiple portfolios.
    Clips all series to common start date and rebases to $10k for fair comparison.
    """
    st.markdown("### Multi-Portfolio Performance Comparison")

    fig = go.Figure()

    # Colors for portfolios
    colors = ['#2E86C1', '#28B463', '#D35400', '#884EA0', '#F1C40F', '#1F618D', '#148F77', '#B03A2E']

    # Determine Common Start Date (latest start among all portfolios)
    start_dates = []
    for res in results_list:
        series = res.get('series')
        if series is not None and not series.empty:
            start_dates.append(series.index.min())
    if benchmarks:
        for bench in benchmarks:
            if bench is not None and not bench.empty:
                start_dates.append(bench.index.min())
    
    common_start = max(start_dates) if start_dates else None
    
    # Use the first portfolio's start_val from user's Global Capital config
    rebase_target = 10000.0
    if results_list:
        rebase_target = results_list[0].get('start_val', 10000.0)
    
    if common_start:
        caption_parts = [
            f"Chart aligned to common start date: **{common_start.date()}**.",
            f"All values rebased to {_format_markdown_currency(rebase_target)}.",
        ]
        cashflow_caption = _format_cashflow_caption(cashflow_config)
        if cashflow_caption:
            caption_parts.append(cashflow_caption)
        st.caption("ℹ️ " + " ".join(caption_parts))

    # Sort by total return (highest first) for legend/tooltip ordering
    def _total_return(idx):
        s = results_list[idx].get('series')
        if s is None or s.empty:
            return 0
        if common_start:
            s = s[s.index >= common_start]
        if s.empty or s.iloc[0] == 0:
            return 0
        return s.iloc[-1] / s.iloc[0]

    sorted_indices = sorted(range(len(results_list)), key=_total_return, reverse=True)

    # Add Portfolios (clipped to common start, rebased to $10k)
    for i in sorted_indices:
        res = results_list[i]
        name = res['name']
        series = res.get('series')
        stats = res.get('stats', {})
        
        if series is None or series.empty:
            continue
        
        # Clip to common start date
        if common_start:
            series = series[series.index >= common_start]
            if series.empty: continue
            
        color = colors[i % len(colors)]
        
        # Recalculate stats from the clipped series so legend matches the chart
        cagr = stats.get('cagr', 0.0)
        cagr_display = cagr * 100 if abs(cagr) <= 1 else cagr

        # Max drawdown from the displayed (clipped) series, not the original
        clipped_float = series.astype(float)
        running_max = clipped_float.cummax()
        max_dd = float((clipped_float / running_max - 1.0).min())
        max_dd_display = max_dd * 100

        label = f"{name} (CAGR: {cagr_display:.2f}%, DD: {max_dd_display:.2f}%)"
        
        # Rebase to $10k at common start for fair visual comparison
        plot_values = series.values
        if not series.empty and series.iloc[0] != 0:
            plot_values = (series.values / series.iloc[0]) * rebase_target
        
        fig.add_trace(go.Scatter(
            x=series.index, 
            y=plot_values, 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2),
            hovertemplate="<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>"
        ))

    # Add Benchmarks (clipped to common start, rebased to $10k)
    if benchmarks:
        for i, bench in enumerate(benchmarks):
            if bench is None or bench.empty: continue

            # Clip to common start date
            if common_start:
                bench = bench[bench.index >= common_start]
                if bench.empty: continue

            b_name = bench.name if hasattr(bench, 'name') and bench.name else f"Benchmark {i+1}"
            b_stats = calculations.generate_stats(bench)
            b_cagr = b_stats.get('cagr', 0.0)
            b_mdd = b_stats.get('max_drawdown', 0.0)
            b_label = f"{b_name} (CAGR: {b_cagr:.2f}%, DD: {b_mdd:.2f}%)"

            # Rebase to $10k
            b_plot = bench.values
            if not bench.empty and bench.iloc[0] != 0:
                b_plot = (bench.values / bench.iloc[0]) * rebase_target

            fig.add_trace(go.Scatter(
                x=bench.index,
                y=b_plot,
                mode='lines',
                name=b_label,
                line=dict(color='#BDC3C7', width=1.5, dash='dash'),
                hovertemplate="<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>"
            ))
            
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Portfolio Value ($)",
        yaxis_type="log" if log_scale else "linear",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown Chart ---
    st.markdown("### Drawdowns")
    fig_dd = go.Figure()

    # 1. Portfolios Drawdown (clipped to common start, same sort as performance chart)
    for i in sorted_indices:
        res = results_list[i]
        series = res.get('series')
        if series is None or series.empty: continue
        
        # Clip to common start date
        if common_start:
            series = series[series.index >= common_start]
            if series.empty: continue
            
        # Drawdown Calc
        series = series.astype(float)
        running_max = series.cummax()
        drawdown = (series / running_max) - 1.0
        
        name = res['name']
        color = colors[i % len(colors)]
        
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown, 
            mode='lines', 
            name=name,
            line=dict(color=color, width=1),
            fill='tozeroy', 
            hovertemplate="<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>"
        ))

    # 2. Benchmarks Drawdown (clipped to common start)
    if benchmarks:
        for i, bench in enumerate(benchmarks):
            if bench is None or bench.empty: continue
            
            # Clip to common start date
            if common_start:
                bench = bench[bench.index >= common_start]
                if bench.empty: continue
            
            bench = bench.astype(float)
            running_max = bench.cummax()
            dd = (bench / running_max) - 1.0
            
            b_name = bench.name if hasattr(bench, 'name') and bench.name else f"Benchmark {i+1}"
            
            fig_dd.add_trace(go.Scatter(
                x=dd.index, 
                y=dd,
                mode='lines',
                name=b_name,
                line=dict(color='#BDC3C7', width=1.0, dash='dash'),
                hovertemplate="<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>"
            ))
            
    fig_dd.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".0%",
        height=350,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=30, b=40)
    )
        
    st.plotly_chart(fig_dd, use_container_width=True)

    # Comparison Table (Stats are already accurate - refetched or calculated in main app)
    if results_list:
        st.markdown("### Statistics")
        
        stats_data = []
        for i in sorted_indices:
            res = results_list[i]
            series = res.get('series')
            stats = res.get('stats', {})
            if series is None or series.empty:
                continue
            
            # Clip to common start (same as chart) for End Balance calculation
            if common_start:
                series = series[series.index >= common_start]
                if series.empty: continue
            
            # Use stats directly - they're already accurate for the displayed period
            cagr_raw = stats.get('cagr', 0)
            cagr_display = cagr_raw * 100 if abs(cagr_raw) <= 1 else cagr_raw
            
            # Handle volatility/std - API may return 'std' or 'volatility'
            vol_raw = stats.get('std', stats.get('volatility', 0))
            vol_display = vol_raw * 100 if abs(vol_raw) <= 1 else vol_raw
            
            row = {
                "Name": res['name'],
                "CAGR": f"{cagr_display:.2f}%",
                "Stdev": f"{vol_display:.2f}%",
                "Sharpe": f"{stats.get('sharpe', 0):.2f}",
                "Max DD": f"{stats.get('max_drawdown', 0):.2f}%",
                "End Balance": f"${series.iloc[-1]:,.2f}" if not series.empty else "$0"
            }
            stats_data.append(row)
            
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

def render_classic_chart(port_series, final_adj_series, loan_series,
                        equity_pct_series, usage_series,
                        series_opts, log_scale,
                        bench_series=None, comparison_series=None, effective_rate_series=None,
                        pm_usage_series=None, pm_mode="Off", pm_blocked_dates=None,
                        draw_start_date=None, draw_monthly=0,
                        draw_monthly_retirement=0, retirement_date=None,
                        draw_dates=None, ret_draw_dates=None, tax_payment_series=None):
    """
    Renders the classic line chart with toggleable traces.
    """
    fig = go.Figure()
    
    # 1. Total Portfolio Value
    if "Portfolio" in series_opts:
        fig.add_trace(go.Scatter(
            x=port_series.index, y=port_series,
            name="Portfolio Value (Gross)",
            line=dict(color='#2E86C1', width=2),
            hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>"
        ))
        
    # 2. Net Equity
    if "Equity" in series_opts:
        # If paying taxes from cash, final_adj_series is Net Equity
        # If paying with margin, final_adj_series is also Net Equity (simulated)
        fig.add_trace(go.Scatter(
            x=final_adj_series.index, y=final_adj_series,
            name="Net Liquidity",
            line=dict(color='#28B463', width=2),
            fill='tozeroy',
            fillcolor='rgba(40, 180, 99, 0.1)',
            hovertemplate="Net Liquidity: $%{y:,.0f}<extra></extra>"
        ))

    # 3. Margin Loan
    if "Loan" in series_opts:
        fig.add_trace(go.Scatter(
            x=loan_series.index, y=loan_series,
            name="Margin Loan",
            line=dict(color='#E74C3C', width=1.5, dash='dot'),
            hovertemplate="Loan: $%{y:,.0f}<extra></extra>"
        ))
        
    # 4. Benchmarks
    if bench_series is not None and not bench_series.empty:
        fig.add_trace(go.Scatter(
            x=bench_series.index, y=bench_series,
            name=bench_series.name or "Benchmark",
            line=dict(color='#F1C40F', width=1.5, dash='dash'),
            hovertemplate=f"{bench_series.name or 'Benchmark'}: $%{{y:,.0f}}<extra></extra>"
        ))

    if comparison_series is not None and not comparison_series.empty:
        fig.add_trace(go.Scatter(
            x=comparison_series.index, y=comparison_series,
            name=comparison_series.name or "Comparison",
            line=dict(color='#9B59B6', width=1.5, dash='dash'),
            hovertemplate=f"{comparison_series.name or 'Comparison'}: $%{{y:,.0f}}<extra></extra>"
        ))
        
    # Secondary Axis for Percentages
    if "Margin usage %" in series_opts or "Equity %" in series_opts:
        y2_max = max(
            usage_series.max() if not usage_series.empty else 0,
            equity_pct_series.max() if not equity_pct_series.empty else 0,
        )
        fig.update_layout(yaxis2=dict(
            title="Percentage (%)",
            overlaying="y",
            side="right",
            range=[0, max(y2_max * 1.2, 1.5)]
        ))

    if "Margin usage %" in series_opts:
        _usage_label = "Reg-T Usage" if pm_mode != "Off" and pm_usage_series is not None and not pm_usage_series.empty else "Usage"
        fig.add_trace(go.Scatter(
            x=usage_series.index, y=usage_series,
            name="Margin Usage %" if pm_mode == "Off" else "Reg-T Usage %",
            line=dict(color='#FFD700', width=1),
            yaxis="y2",
            hovertemplate=f"{_usage_label}: %{{y:.1%}}<extra></extra>"
        ))
        
    if "Equity %" in series_opts:
        fig.add_trace(go.Scatter(
            x=equity_pct_series.index, y=equity_pct_series,
            name="Equity %",
            line=dict(color='#148F77', width=1),
            yaxis="y2",
            hovertemplate="Equity %: %{y:.1%}<extra></extra>"
        ))

    # PM Usage overlay (Compare mode)
    if pm_usage_series is not None and not pm_usage_series.empty and "Margin usage %" in series_opts and pm_mode == "Compare":
        fig.add_trace(go.Scatter(
            x=pm_usage_series.index, y=pm_usage_series,
            name="PM Usage %",
            line=dict(color='#00CED1', width=1.5, dash='dash'),
            yaxis="y2",
            hovertemplate="PM Usage: %{y:.1%}<extra></extra>"
        ))

    # PM buy-blocked rebalance markers
    if pm_blocked_dates:
        blocked_ts = pd.to_datetime(pm_blocked_dates)
        for bd in blocked_ts:
            fig.add_vline(
                x=bd, line_width=2, line_dash="dash", line_color="rgba(255, 99, 71, 0.7)",
                annotation_text="⛔", annotation_position="top right",
                annotation_font_size=14,
            )
        # Add scatter markers on the portfolio line at blocked dates
        if "Portfolio" in series_opts and not port_series.empty:
            matched_vals = []
            matched_dates = []
            for bd in blocked_ts:
                # Find nearest date in series
                idx = port_series.index.get_indexer([bd], method="nearest")[0]
                if 0 <= idx < len(port_series):
                    matched_dates.append(port_series.index[idx])
                    matched_vals.append(port_series.iloc[idx])
            if matched_dates:
                fig.add_trace(go.Scatter(
                    x=matched_dates, y=matched_vals,
                    mode="markers",
                    name="PM Buy Block",
                    marker=dict(symbol="x", size=12, color="#FF6347", line=dict(width=2, color="#FF6347")),
                    hovertemplate="⛔ Rebalance blocked<br>%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>",
                ))

    # Margin Call Threshold Line (100% Usage)
    if "Margin usage %" in series_opts:
        fig.add_trace(go.Scatter(
            x=[port_series.index[0], port_series.index[-1]], 
            y=[1.0, 1.0],
            name="Margin Call Threshold",
            line=dict(color='#FF0000', width=1.5, dash='dash'), # Red dashed line
            yaxis="y2",
            mode="lines",
            hoverinfo="skip" # Don't clutter hover
        ))
        
    if effective_rate_series is not None and not effective_rate_series.empty:
        # Check if rate is in % (e.g. 5.0) or decimal (0.05). API returns %.
        # Convert to decimal for consistent Y-axis with other percentages
        rate_decimal = effective_rate_series / 100.0
        fig.add_trace(go.Scatter(
            x=effective_rate_series.index, y=rate_decimal,
            name="Margin Rate %",
            line=dict(color='#FF00FF', width=1, dash='dot'), # Magenta
            yaxis="y2",

            hovertemplate="Rate: %{y:.2%}<extra></extra>" 
        ))

    # Withdrawal vertical lines (thin red, capped at Reg-T usage curve)
    if draw_dates and not usage_series.empty:
        for dd in draw_dates:
            idx = usage_series.index.get_indexer([dd], method="nearest")[0]
            if 0 <= idx < len(usage_series):
                draw_ts = pd.Timestamp(dd)
                fig.add_shape(
                    type="line", x0=draw_ts, x1=draw_ts, y0=0, y1=usage_series.iloc[idx],
                    xref="x", yref="y2",
                    line=dict(color="rgba(255,0,0,0.35)", width=1),
                )

    # Retirement draw vertical lines (orange, capped at Reg-T usage curve)
    if ret_draw_dates and not usage_series.empty:
        for dd in ret_draw_dates:
            idx = usage_series.index.get_indexer([dd], method="nearest")[0]
            if 0 <= idx < len(usage_series):
                draw_ts = pd.Timestamp(dd)
                fig.add_shape(
                    type="line", x0=draw_ts, x1=draw_ts, y0=0, y1=usage_series.iloc[idx],
                    xref="x", yref="y2",
                    line=dict(color="rgba(255,165,0,0.45)", width=1),
                )

    # Tax payment vertical lines (cyan, capped at Reg-T usage value) + hover markers
    if tax_payment_series is not None and not usage_series.empty:
        t_dates, t_vals, t_texts = [], [], []
        for td, tax_amt in tax_payment_series[tax_payment_series > 0].items():
            idx = usage_series.index.get_indexer([td], method="nearest")[0]
            if 0 <= idx < len(usage_series):
                d = usage_series.index[idx]
                v = usage_series.iloc[idx]
                tax_ts = pd.Timestamp(td)
                fig.add_shape(
                    type="line", x0=tax_ts, x1=tax_ts, y0=0, y1=v,
                    xref="x", yref="y2",
                    line=dict(color="rgba(0,255,255,0.5)", width=1),
                )
                t_dates.append(d)
                t_vals.append(v)
                t_texts.append(f"🏛 Tax Payment<br>{d.strftime('%b %d, %Y')}<br>${tax_amt:,.0f}")
        if t_dates:
            fig.add_trace(go.Scatter(
                x=t_dates, y=t_vals, mode="markers", yaxis="y2",
                name="Tax Payment", showlegend=False,
                marker=dict(size=1, opacity=0),
                hovertemplate="%{text}<extra></extra>",
                text=t_texts,
            ))

    # Withdrawal start date vertical line (only if within chart range)
    _chart_start = port_series.index[0] if not port_series.empty else None
    _chart_end = port_series.index[-1] if not port_series.empty else None
    _effective_draw_start = draw_start_date if draw_start_date is not None else (_chart_start if draw_monthly else None)
    if _effective_draw_start is not None and _chart_start is not None:
        draw_ts = pd.Timestamp(_effective_draw_start)
        # Snap to nearest chart date if draw_ts falls on a non-trading day
        if draw_ts < _chart_start:
            draw_ts = _chart_start  # Clamp to first trading day
        if draw_ts <= _chart_end:
            date_label = draw_ts.strftime("%b %d, %Y")
            amt_label = f"${draw_monthly:,.0f}/mo" if draw_monthly else ""
            annotation = f"Withdrawals Start<br>{date_label}"
            if amt_label:
                annotation += f"<br>{amt_label}"
            fig.add_shape(
                type="line", x0=draw_ts, x1=draw_ts, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="red", width=2),
            )
            fig.add_annotation(
                x=draw_ts, y=1, xref="x", yref="paper",
                text=annotation, showarrow=False,
                font=dict(color="red", size=11),
                xanchor="left", xshift=6, yshift=10,
            )

    # Retirement date vertical line (only if within chart range)
    if retirement_date is not None and draw_monthly_retirement > 0 and _chart_start is not None:
        ret_ts = pd.Timestamp(retirement_date)
        # Snap to nearest chart date if on a non-trading day
        if ret_ts < _chart_start:
            ret_ts = _chart_start
        if not (ret_ts <= _chart_end):
            ret_ts = None
    else:
        ret_ts = None
    if ret_ts is not None:
        ret_label = ret_ts.strftime("%b %d, %Y")
        ret_annotation = f"Retirement<br>{ret_label}<br>${draw_monthly_retirement:,.0f}/mo"
        fig.add_shape(
            type="line", x0=ret_ts, x1=ret_ts, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="#00BFFF", width=2),
        )
        fig.add_annotation(
            x=ret_ts, y=1, xref="x", yref="paper",
            text=ret_annotation, showarrow=False,
            font=dict(color="#00BFFF", size=11),
            xanchor="left", xshift=6, yshift=10,
        )

    # Pad x-axis left if a vertical marker sits at the chart start
    _x_range_start = None
    if _effective_draw_start is not None and _chart_start is not None:
        _draw_ts_check = pd.Timestamp(_effective_draw_start)
        if _draw_ts_check <= _chart_start:
            _x_range_start = _chart_start - pd.Timedelta(days=15)

    fig.update_layout(
        template="plotly_dark",
        title="Portfolio History",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        xaxis_range=[_x_range_start, None] if _x_range_start else None,
        yaxis_title="Value ($)",
        yaxis_type="log" if log_scale else "linear",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard_view(port, equity, loan, equity_pct, usage_pct, maint_pct, stats, log_opts, bench_series=None, comparison_series=None, start_val=10000, rate_annual=8.0):
    """Render dashboard-style separate charts"""
    
    # Get log scale settings
    log_portfolio = log_opts.get("portfolio", False)
    log_leverage = log_opts.get("leverage", False)
    log_margin = log_opts.get("margin", False)
    
    # Configure dark theme
    dark_theme = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(30,35,45,1)",
        "plot_bgcolor": "rgba(30,35,45,1)",
        "font": {"color": "#E0E0E0"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.2)"
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.2)"
        }
    }
    
    # Row 1: Portfolio Value Chart
    st.markdown("### Portfolio Value Over Time")
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=port.index, y=port,
        name="Margin Portfolio",
        line=dict(color="#1DB954", width=2),
        hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Equity (Net of Loan)",
        line=dict(color="#4A90E2", width=2),
        hovertemplate="Equity: $%{y:,.0f}<extra></extra>"
    ))

    # Add Benchmark Trace if available (Moved to end)
    if bench_series is not None:
         bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
         fig1.add_trace(go.Scatter(
             x=bench_series.index, y=bench_series,
             name=bench_name,
             line=dict(color="#FFD700", width=2, dash="dash"),
             hovertemplate=f"{bench_name}: $%{{y:,.0f}}<extra></extra>"
         ))

    # Add Comparison Trace if available
    if comparison_series is not None:
         comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
         fig1.add_trace(go.Scatter(
             x=comparison_series.index, y=comparison_series,
             name=comp_name,
             line=dict(color="#00FFFF", width=2, dash="dot"),
             hovertemplate=f"{comp_name}: $%{{y:,.0f}}<extra></extra>"
         ))
    
    fig1.update_layout(
        **dark_theme,
        height=400,
        xaxis_title="Month",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Portfolio Value ($)",
        yaxis_type="log" if log_portfolio else "linear",
        legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Row 2: Two columns for Leverage and Margin Debt
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Leverage Over Time")
        fig2 = go.Figure()
        
        # Calculate leverage metrics
        current_leverage = port / equity
        target_leverage = 1 / (start_val / 100) if start_val < 100 else 1
        max_allowed = 1 / (1 - maint_pct)
        
        fig2.add_trace(go.Scatter(
            x=port.index, y=current_leverage,
            name="Current Leverage",
            line=dict(color="#1DB954", width=2),
            hovertemplate="Current: %{y:.2f}x<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[target_leverage]*len(port),
            name="Target Leverage",
            line=dict(color="#FFD700", width=2, dash="dot"),
            hovertemplate="Target: %{y:.2f}x<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[max_allowed]*len(port),
            name="Max Allowed",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            hovertemplate="Max: %{y:.2f}x<extra></extra>"
        ))
        
        fig2.update_layout(
            **dark_theme,
            height=350,
            xaxis_title="Date",
            xaxis_hoverformat="%b %d, %Y",
            yaxis_title="Leverage Ratio",
            yaxis_type="log" if log_leverage else "linear",
            legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Margin Debt Evolution")
        fig3 = go.Figure()
        
        # Add area chart for margin debt
        fig3.add_trace(go.Scatter(
            x=loan.index, y=loan,
            name="Margin Debt",
            fill='tozeroy',
            line=dict(color="#FF6B6B", width=2),
            fillcolor="rgba(255,107,107,0.3)",
            hovertemplate="Debt: $%{y:,.0f}<extra></extra>"
        ))
        
        # Portfolio value line
        fig3.add_trace(go.Scatter(
            x=port.index, y=port,
            name="Portfolio Value",
            line=dict(color="#4A90E2", width=2),
            yaxis="y",
            hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>"
        ))
        
        # Net liquidating value
        fig3.add_trace(go.Scatter(
            x=equity.index, y=equity,
            name="Net Liquidating Value",
            line=dict(color="#1DB954", width=2),
            yaxis="y",
            hovertemplate="Net Liq: $%{y:,.0f}<extra></extra>"
        ))
        
        # Actual/360 estimate for the calendar month shown on each row.
        month_day_fraction = pd.Series(loan.index.days_in_month / 360.0, index=loan.index)
        monthly_interest = loan * (rate_annual / 100) * month_day_fraction
        fig3.add_trace(go.Scatter(
            x=loan.index, y=monthly_interest,
            name="Monthly Interest",
            line=dict(color="#FFD700", width=1),
            yaxis="y2"
        ))
        
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(30,35,45,1)",
            plot_bgcolor="rgba(30,35,45,1)",
            font={"color": "#E0E0E0"},
            height=350,
            xaxis=dict(
                title="Date",
                hoverformat="%b %d, %Y",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                title="Value ($)", 
                side="left",
                type="log" if log_margin else "linear",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis2=dict(
                title="Monthly Interest ($)", 
                overlaying="y", 
                side="right",
                type="log" if log_margin else "linear",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Row 3: Final Margin Status - Enhanced display
    st.markdown("### Final Margin Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Margin Utilization gauge
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=usage_pct.iloc[-1] * 100,
            title={'text': "Margin Utilization"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#FFD700"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0,255,0,0.3)"},
                    {'range': [50, 80], 'color': "rgba(255,255,0,0.3)"},
                    {'range': [80, 100], 'color': "rgba(255,0,0,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            number={'suffix': "%", 'font': {'size': 40}},
            delta={'reference': 50, 'increasing': {'color': "red"}}
        ))
        fig4.update_layout(
            **dark_theme,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: #888;'>{'Moderate Risk' if usage_pct.iloc[-1] < 0.8 else 'High Risk'}</p>", unsafe_allow_html=True)
    
    with col2:
        # Leverage gauge
        final_leverage = (port.iloc[-1] / equity.iloc[-1])
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_leverage,
            title={'text': "Leverage"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [1, max_allowed], 'tickwidth': 1},
                'bar': {'color': "#4A90E2"},
                'steps': [
                    {'range': [1, 2], 'color': "rgba(0,255,0,0.3)"},
                    {'range': [2, 3], 'color': "rgba(255,255,0,0.3)"},
                    {'range': [3, max_allowed], 'color': "rgba(255,0,0,0.3)"}
                ],
            },
            number={'suffix': "x", 'font': {'size': 40}}
        ))
        fig5.update_layout(
            **dark_theme,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: #888;'>Peak {max_allowed:.2f}x</p>", unsafe_allow_html=True)
    
    with col3:
        # Available Margin display
        available_margin = port.iloc[-1] * (1 - maint_pct) - loan.iloc[-1]
        st.markdown(
            f"""
            <div style='background-color: rgba(30,35,45,1); padding: 20px; border-radius: 10px; text-align: center;'>
                <h4 style='color: #888; margin-bottom: 10px;'>Available Margin</h4>
                <h1 style='color: #1DB954; margin: 10px 0;'>${available_margin:,.2f}</h1>
                <p style='color: #888; margin-top: 10px;'>Additional Borrowing Power</p>
            </div>
            """,
            unsafe_allow_html=True
        )
