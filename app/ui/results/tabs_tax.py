"""Tax analysis tab rendering."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.core import calculations
from app.services import testfol_api as api


def render_tax_impact_tab(
    tab,
    pl_by_year: pd.Series | pd.DataFrame,
    config: dict,
    port_series: pd.Series,
    equity_series: pd.Series,
    loan_series: pd.Series,
    final_adj_series: pd.Series,
    annual_total_tax: pd.Series,
    tax_payment_series: pd.Series,
    pay_tax_margin: bool,
    pay_tax_cash: bool,
    rate_annual,
    draw_monthly: float,
    starting_loan: float,
    wmaint: float,
    repayment_series: pd.Series | None,
    twr_series: pd.Series | None,
    stats: dict,
    draw_start_date=None,
    accrual_start_date=None,
) -> None:
    with tab:
        st.markdown("### Annual Tax Impact Analysis")

        st.info("""
        **Methodology: Tax-Adjusted Returns**

        *   **None (Gross):** No tax simulation. Showing raw pre-tax returns.
        *   **Pay with Margin:** Taxes paid via loan. Assets preserved. Cost = Interest.
        *   **Pay from Cash:** Taxes paid via asset sales. Assets reduced. Cost = Lost Compounding.
        """)

        # --- Data Integrity Check (Chart vs Tax Data) ---
        if twr_series is not None and not twr_series.empty:
            _render_data_validation(twr_series, port_series, stats, config)

        if (pay_tax_margin or pay_tax_cash) and not final_adj_series.empty and not pl_by_year.empty:
            _render_tax_impact_chart(
                final_adj_series, annual_total_tax, equity_series,
                loan_series, port_series, rate_annual, draw_monthly,
                starting_loan, wmaint, pay_tax_cash, repayment_series,
                draw_start_date=draw_start_date,
                accrual_start_date=accrual_start_date,
            )
        elif not (pay_tax_margin or pay_tax_cash):
            st.warning("Tax Simulation is set to **None (Gross)**. Enable 'Pay from Cash' or 'Pay with Margin' to see tax impact analysis.")
        else:
            st.info("No data available for tax analysis.")


def _render_data_validation(
    twr_series: pd.Series,
    port_series: pd.Series,
    stats: dict,
    config: dict,
) -> None:
    # 1. Determine Common Timeframe
    start_dt = twr_series.index[0]
    end_dt = twr_series.index[-1]
    years = (end_dt - start_dt).days / 365.25

    # 2. Calculate Shadow CAGR
    shadow_cagr = (twr_series.iloc[-1] / twr_series.iloc[0]) ** (1/years) - 1

    # 3. Calculate "Subset" API CAGR (Aligned to Shadow Dates)
    api_cagr = 0.0
    if not port_series.empty:
        try:
            p_start = port_series.asof(start_dt)
            p_end = port_series.asof(end_dt)
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                api_cagr = (p_end / p_start) ** (1/years) - 1
        except (IndexError, KeyError, ZeroDivisionError, TypeError):
            api_cagr = stats.get("cagr", 0.0)
            if api_cagr > 1.0: api_cagr /= 100.0

    # 4. Comparison
    cf_amt = config.get('cashflow', 0.0)
    diff = abs(shadow_cagr - api_cagr)

    with st.expander("🔎 Data Validation (Chart vs Tax Data)", expanded=False):
        st.caption(f"Comparing performance over overlapped period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Chart CAGR (Testfol)", f"{api_cagr:.2%}")
        c2.metric("Tax CAGR (yFinance)", f"{shadow_cagr:.2%}")

        delta_color = "normal"
        if diff < 0.002: delta_color = "normal"
        elif diff < 0.005: delta_color = "off"
        else: delta_color = "inverse"

        c3.metric("Drift", f"{diff:.2%}", delta="Match" if diff < 0.005 else "Mismatch", delta_color=delta_color)

        if abs(cf_amt) > 10:
            st.caption("Note: Small drift is expected when Cashflows are active (MWR vs TWR).")
        elif diff > 0.01:
            st.warning("Significant drift detected! 'yfinance' price data may differ from Testfol's source (e.g., missing dividends, bad splits). Tax calculations may be less accurate.")

        sim_engine = config.get('sim_engine', 'standard')
        if sim_engine == 'hybrid':
            st.info("""
**Hybrid Mode: Why Drift Occurs**

Your chart uses Testfol's extended simulated data (e.g., SPYSIM for SPY back to the 1980s). However, tax calculations use **real yFinance data only**, which may start later (e.g., real ZROZ data starts in 2009).

When yFinance data starts later than your chart, the tax engine initializes your portfolio at that later date using the chart's value at that time as your cost basis. This ensures taxes are grounded in reality rather than synthetic models.

**Example:** If your chart shows $500k in 2009 but uses simulated data for 1980–2009, your taxes will treat 2009 as your acquisition date with a $500k cost basis.
            """.strip())


def _render_tax_impact_chart(
    final_adj_series: pd.Series,
    annual_total_tax: pd.Series,
    equity_series: pd.Series,
    loan_series: pd.Series,
    port_series: pd.Series,
    rate_annual,
    draw_monthly: float,
    starting_loan: float,
    wmaint: float,
    pay_tax_cash: bool,
    repayment_series: pd.Series | None,
    draw_start_date=None,
    accrual_start_date=None,
) -> None:
    # Annual Ending Balance (Tax Adjusted)
    annual_bal = final_adj_series.resample("YE").last()
    annual_bal.index = annual_bal.index.year

    annual_tax_aligned = annual_total_tax.reindex(annual_bal.index, fill_value=0.0)

    # Market Value (Gross Assets)
    if pay_tax_cash:
        empty_tax = pd.Series(0.0, index=equity_series.index)
        gross_cash_series, _ = calculations.calculate_tax_adjusted_equity(
            equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly, draw_start_date=draw_start_date
        )
        market_val_series = gross_cash_series
    else:
        gross_margin_loan, gross_margin_equity, _, _, _ = api.simulate_margin(
            port_series, starting_loan, rate_annual, draw_monthly, wmaint,
            tax_series=None, repayment_series=repayment_series,
            draw_start_date=draw_start_date,
            accrual_start_date=accrual_start_date,
        )
        market_val_series = gross_margin_equity

    annual_mv = market_val_series.resample("YE").last()
    annual_mv.index = annual_mv.index.year

    tax_impact_df = pd.DataFrame({
        "Market Value": annual_mv,
        "Ending Balance": annual_bal,
        "Tax Paid": annual_tax_aligned
    })

    fig_tax_impact = go.Figure()

    fig_tax_impact.add_trace(go.Scatter(
        x=tax_impact_df.index,
        y=tax_impact_df["Market Value"],
        name="Pre-Tax Wealth (Baseline)",
        line=dict(color="#636EFA", width=3),
        mode='lines+markers',
        hovertemplate="%{y:$,.0f}<extra></extra>"
    ))

    fig_tax_impact.add_trace(go.Bar(
        x=tax_impact_df.index,
        y=tax_impact_df["Ending Balance"],
        name="Ending Balance (Net)",
        marker_color="#00CC96",
        texttemplate="%{y:$.2s}",
        textposition="auto",
        hovertemplate="%{y:$,.0f}<extra></extra>"
    ))

    fig_tax_impact.add_trace(go.Bar(
        x=tax_impact_df.index,
        y=tax_impact_df["Tax Paid"],
        name="Tax Paid",
        marker_color="#EF553B",
        texttemplate="%{y:$.2s}",
        textposition="auto",
        hovertemplate="%{y:$,.0f}<extra></extra>"
    ))

    fig_tax_impact.update_layout(
        title="Annual Tax Impact: Net Wealth vs. Pre-Tax Baseline",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        barmode='stack',
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_tax_impact, use_container_width=True)

    st.markdown("### Detailed Data")
    st.dataframe(tax_impact_df.style.format("${:,.2f}"), use_container_width=True)
