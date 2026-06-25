"""Rolling Metrics charts — sourced from Testfol API or computed locally."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.charts.stats_helpers import excess_kurtosis, skewness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

ROLLING_METRICS = {
    "cagr":        {"label": "CAGR",              "yaxis": "CAGR (%)",              "fmt": ".2f", "suffix": "%", "is_pct": True},
    "std":         {"label": "Volatility",        "yaxis": "Volatility (%)",        "fmt": ".2f", "suffix": "%", "is_pct": True},
    "sharpe":      {"label": "Sharpe",            "yaxis": "Sharpe Ratio",          "fmt": ".2f", "suffix": "",  "is_pct": False},
    "sortino":     {"label": "Sortino",           "yaxis": "Sortino Ratio",         "fmt": ".2f", "suffix": "",  "is_pct": False},
    "excess_cagr": {"label": "Excess CAGR",       "yaxis": "Excess CAGR (%)",       "fmt": ".2f", "suffix": "%", "is_pct": True},
    "cum_return":  {"label": "Cumulative Return", "yaxis": "Cumulative Return (%)", "fmt": ".2f", "suffix": "%", "is_pct": True},
    "max_dd":      {"label": "Max Drawdown",      "yaxis": "Max Drawdown (%)",      "fmt": ".2f", "suffix": "%", "is_pct": True},
    "skewness":    {"label": "Skewness",          "yaxis": "Skewness",              "fmt": ".2f", "suffix": "",  "is_pct": False},
    "kurtosis":    {"label": "Kurtosis",          "yaxis": "Excess Kurtosis",       "fmt": ".2f", "suffix": "",  "is_pct": False},
    "full_kelly":  {"label": "Full Kelly",        "yaxis": "Full Kelly Leverage",   "fmt": ".2f", "suffix": "x", "is_pct": False},
    "ulcer":       {"label": "Ulcer Index",       "yaxis": "Ulcer Index",           "fmt": ".2f", "suffix": "",  "is_pct": False},
    "calmar":      {"label": "Calmar",            "yaxis": "Calmar Ratio",          "fmt": ".2f", "suffix": "",  "is_pct": False},
    "beta":        {"label": "Beta",              "yaxis": "Beta",                  "fmt": ".2f", "suffix": "",  "is_pct": False},
}


# ---------------------------------------------------------------------------
# Benchmark fetching (SPYSIM — cached via disk)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=None)
def _fetch_spysim_series() -> pd.Series:
    """Fetch SPYSIM benchmark series via Testfol API (disk-cached)."""
    from app.services import testfol_api as api
    try:
        series, _, _ = api.fetch_backtest(
            start_date="1900-01-01",
            end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            start_val=10000,
            cashflow=0,
            cashfreq="Monthly",
            rolling=1,
            invest_div=True,
            rebalance="Yearly",
            allocation={"SPYSIM": 100.0},
        )
        return series
    except Exception as e:
        logger.warning(f"Failed to fetch SPYSIM benchmark: {e}")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Local computation (for NDXMEGASIM / no API path)
# ---------------------------------------------------------------------------

def _compute_rolling_local(
    port_series: pd.Series,
    window_months: int,
    bench_series: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Compute rolling metrics locally from a daily value series.

    Args:
        port_series: Portfolio daily value series.
        window_months: Rolling window in months.
        bench_series: Optional benchmark series (SPYSIM) for Beta & Excess CAGR.
    """
    daily_rets = port_series.pct_change().dropna()
    if daily_rets.empty:
        return {}

    window_days = window_months * 21  # ~21 trading days per month
    if len(daily_rets) < window_days:
        return {}

    # Align benchmark returns if available
    bench_rets: pd.Series | None = None
    if bench_series is not None and not bench_series.empty:
        bench_rets = bench_series.pct_change().dropna()
        # Align to common dates
        common_idx = daily_rets.index.intersection(bench_rets.index)
        if len(common_idx) >= window_days:
            bench_rets = bench_rets.reindex(common_idx)
        else:
            bench_rets = None

    results = {}

    # Rolling CAGR
    start_vals = port_series.shift(window_days)
    valid = start_vals.notna() & (start_vals > 0)
    years = window_months / 12.0
    cagr = pd.Series(np.nan, index=port_series.index)
    cagr[valid] = ((port_series[valid] / start_vals[valid]) ** (1 / years) - 1) * 100
    results["cagr"] = cagr.dropna()

    # Rolling Volatility (annualized)
    roll_std = daily_rets.rolling(window_days).std() * np.sqrt(252) * 100
    results["std"] = roll_std.dropna()

    # Rolling Sharpe
    roll_mean = daily_rets.rolling(window_days).mean()
    roll_std_raw = daily_rets.rolling(window_days).std()
    sharpe = pd.Series(np.nan, index=daily_rets.index)
    nonzero = roll_std_raw > 0
    sharpe[nonzero] = (roll_mean[nonzero] / roll_std_raw[nonzero]) * np.sqrt(252)
    results["sharpe"] = sharpe.dropna()

    # Rolling Sortino
    def _rolling_sortino(rets, w):
        vals = np.full(len(rets), np.nan)
        r = rets.values
        for i in range(w, len(r)):
            chunk = r[i - w:i]
            neg = np.minimum(chunk, 0)
            dd = np.sqrt(np.mean(neg ** 2))
            if dd > 0:
                vals[i] = (np.mean(chunk) / dd) * np.sqrt(252)
        return pd.Series(vals, index=rets.index).dropna()

    results["sortino"] = _rolling_sortino(daily_rets, window_days)

    # Rolling Cumulative Return
    cum_ret = pd.Series(np.nan, index=port_series.index)
    cum_ret[valid] = ((port_series[valid] / start_vals[valid]) - 1) * 100
    results["cum_return"] = cum_ret.dropna()

    # Rolling Max Drawdown
    def _rolling_max_dd(series, w):
        vals = np.full(len(series), np.nan)
        s = series.values
        for i in range(w, len(s)):
            chunk = s[i - w:i + 1]
            cummax = np.maximum.accumulate(chunk)
            dd = (chunk - cummax) / cummax
            vals[i] = np.min(dd) * 100
        return pd.Series(vals, index=series.index).dropna()

    results["max_dd"] = _rolling_max_dd(port_series, window_days)

    # Rolling Skewness
    roll_skew = daily_rets.rolling(window_days).apply(skewness, raw=True)
    results["skewness"] = roll_skew.dropna()

    # Rolling Excess Kurtosis
    roll_kurt = daily_rets.rolling(window_days).apply(excess_kurtosis, raw=True)
    results["kurtosis"] = roll_kurt.dropna()

    # Rolling Full Kelly: f* = mean / var (daily), then annualized leverage
    # Kelly fraction = mean_excess / variance, but simplified: mean/var
    def _rolling_kelly(rets, w):
        vals = np.full(len(rets), np.nan)
        r = rets.values
        for i in range(w, len(r)):
            chunk = r[i - w:i]
            var = np.var(chunk, ddof=1)
            if var > 0:
                vals[i] = np.mean(chunk) / var
        return pd.Series(vals, index=rets.index).dropna()

    results["full_kelly"] = _rolling_kelly(daily_rets, window_days)

    # Rolling Ulcer Index
    def _rolling_ulcer(series, w):
        vals = np.full(len(series), np.nan)
        s = series.values
        for i in range(w, len(s)):
            chunk = s[i - w:i + 1]
            cummax = np.maximum.accumulate(chunk)
            dd_pct = (chunk - cummax) / cummax * 100
            vals[i] = np.sqrt(np.mean(dd_pct ** 2))
        return pd.Series(vals, index=series.index).dropna()

    results["ulcer"] = _rolling_ulcer(port_series, window_days)

    # Rolling Calmar = rolling CAGR / |rolling max DD|
    if "cagr" in results and "max_dd" in results:
        aligned = pd.DataFrame({"cagr": results["cagr"], "max_dd": results["max_dd"]}).dropna()
        calmar = pd.Series(np.nan, index=aligned.index)
        nonzero_dd = aligned["max_dd"].abs() > 0
        calmar[nonzero_dd] = aligned["cagr"][nonzero_dd] / aligned["max_dd"][nonzero_dd].abs()
        results["calmar"] = calmar.dropna()

    # --- Benchmark-relative metrics (require bench_series) ---
    if bench_rets is not None and bench_series is not None:
        # Align portfolio returns to common index with benchmark
        common_idx = daily_rets.index.intersection(bench_rets.index)
        port_aligned = daily_rets.reindex(common_idx)
        bench_aligned = bench_rets.reindex(common_idx)

        # Rolling Beta = Cov(port, bench) / Var(bench)
        def _rolling_beta(p_rets, b_rets, w):
            vals = np.full(len(p_rets), np.nan)
            p = p_rets.values
            b = b_rets.values
            for i in range(w, len(p)):
                p_chunk = p[i - w:i]
                b_chunk = b[i - w:i]
                b_var = np.var(b_chunk, ddof=1)
                if b_var > 0:
                    vals[i] = np.cov(p_chunk, b_chunk, ddof=1)[0, 1] / b_var
            return pd.Series(vals, index=p_rets.index).dropna()

        results["beta"] = _rolling_beta(port_aligned, bench_aligned, window_days)

        # Rolling Excess CAGR = rolling portfolio CAGR - rolling benchmark CAGR
        if bench_series is not None:
            bench_start = bench_series.shift(window_days)
            bench_valid = bench_start.notna() & (bench_start > 0)
            years = window_months / 12.0
            bench_cagr = pd.Series(np.nan, index=bench_series.index)
            bench_cagr[bench_valid] = (
                (bench_series[bench_valid] / bench_start[bench_valid]) ** (1 / years) - 1
            ) * 100
            bench_cagr = bench_cagr.dropna()

            if "cagr" in results and not bench_cagr.empty:
                excess = pd.DataFrame({
                    "port": results["cagr"],
                    "bench": bench_cagr,
                }).dropna()
                if not excess.empty:
                    results["excess_cagr"] = (excess["port"] - excess["bench"]).dropna()

    return results


# ---------------------------------------------------------------------------
# Parse API rolling data
# ---------------------------------------------------------------------------

def _parse_api_rolling(raw_response: dict | None) -> dict[str, pd.Series]:
    """Extract rolling chart data from the API response."""
    if not raw_response:
        return {}
    api_resp = raw_response.get("raw_response")
    if not api_resp:
        return {}
    charts = api_resp.get("charts", {})
    rolling_data = charts.get("rolling", {})
    if not isinstance(rolling_data, dict):
        return {}

    results = {}
    for key, meta in ROLLING_METRICS.items():
        if key not in rolling_data:
            continue
        pair = rolling_data[key]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        ts, vals = pair
        if not ts or not vals or len(ts) != len(vals):
            continue
        dates = pd.to_datetime(ts, unit="s")
        results[key] = pd.Series(vals, index=dates, name=key)

    return results


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_rolling_metrics(
    port_series: pd.Series,
    raw_response: dict | None = None,
    unique_id: str = "",
    window_months: int = 60,
):
    """Render rolling metrics section with selectable metric and window."""
    st.subheader("Rolling Metrics")

    # Window selector
    window_options = [12, 24, 36, 48, 60, 84, 120]
    window_labels = {w: f"{w} months ({w // 12}y)" if w >= 12 else f"{w} months" for w in window_options}

    col_window, col_metric = st.columns([1, 1])
    with col_window:
        selected_window = st.selectbox(
            "Rolling Window",
            options=window_options,
            index=window_options.index(window_months) if window_months in window_options else 4,
            format_func=lambda w: window_labels[w],
            key=f"roll_window_{unique_id}",
        )

    # Try API data first (only valid for the default 60-month window the API was called with)
    api_data = _parse_api_rolling(raw_response) if selected_window == 60 else {}

    # Fall back to local computation
    if not api_data:
        # Fetch SPYSIM benchmark for Beta & Excess CAGR
        bench_series = _fetch_spysim_series()
        with st.spinner("Computing rolling metrics..."):
            local_data = _compute_rolling_local(
                port_series, selected_window,
                bench_series=bench_series if not bench_series.empty else None,
            )
    else:
        # Even with API data, compute locally-only metrics (Beta, Excess CAGR, etc.)
        # that the API may not provide
        bench_series = _fetch_spysim_series()
        local_only = _compute_rolling_local(
            port_series, selected_window,
            bench_series=bench_series if not bench_series.empty else None,
        )
        # Only keep metrics not already in API data
        local_data = {k: v for k, v in local_only.items() if k not in api_data}

    # Merge: API takes priority, then local
    all_data = {**local_data, **api_data}

    if not all_data:
        st.info("Insufficient data for rolling metrics with this window.")
        return

    # Available metrics
    available = [k for k in ROLLING_METRICS if k in all_data]
    if not available:
        st.info("No rolling metrics available.")
        return

    with col_metric:
        selected_metric = st.selectbox(
            "Rolling Metric",
            options=available,
            format_func=lambda k: ROLLING_METRICS[k]["label"],
            key=f"roll_metric_{unique_id}",
        )

    meta = ROLLING_METRICS[selected_metric]
    series = all_data[selected_metric]

    if series.empty:
        st.info(f"No data for {meta['label']}.")
        return

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(color="#42A5F5", width=1.5),
        name=meta["label"],
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{meta['label']}: %{{y:{meta['fmt']}}}{meta['suffix']}<extra></extra>",
    ))

    # Add zero line for metrics that center around zero
    if selected_metric in ("excess_cagr", "skewness", "kurtosis"):
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

    # Add 1.0 line for Kelly
    if selected_metric == "full_kelly":
        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,200,0,0.5)", line_width=1,
                      annotation_text="1x Leverage", annotation_position="bottom right")

    # Add 1.0 line for Beta (market beta)
    if selected_metric == "beta":
        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,200,0,0.5)", line_width=1,
                      annotation_text="Market β", annotation_position="bottom right")

    suffix = meta["suffix"]
    tick_fmt = f"{meta['fmt']}" if not suffix else f"{meta['fmt']}"
    y_ticksuffix = suffix if suffix != "x" else "x"

    fig.update_layout(
        title=f"{selected_window}-Month Rolling {meta['label']}",
        yaxis_title=meta["yaxis"],
        yaxis_tickformat=tick_fmt,
        yaxis_ticksuffix=y_ticksuffix if meta["is_pct"] or suffix == "x" else "",
        template="plotly_dark",
        height=400,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"roll_chart_{unique_id}")

    # Summary stats for the selected rolling metric
    with st.expander(f"Rolling {meta['label']} Summary Statistics", expanded=False):
        vals = series.dropna().values
        if len(vals) > 0:
            s = suffix
            stats_data = {
                "Statistic": [
                    "Current", "Mean", "Median", "Std Dev",
                    "Min", "5th Percentile", "25th Percentile",
                    "75th Percentile", "95th Percentile", "Max",
                ],
                "Value": [
                    f"{vals[-1]:{meta['fmt']}}{s}",
                    f"{np.mean(vals):{meta['fmt']}}{s}",
                    f"{np.median(vals):{meta['fmt']}}{s}",
                    f"{np.std(vals, ddof=1):{meta['fmt']}}{s}",
                    f"{np.min(vals):{meta['fmt']}}{s}",
                    f"{np.percentile(vals, 5):{meta['fmt']}}{s}",
                    f"{np.percentile(vals, 25):{meta['fmt']}}{s}",
                    f"{np.percentile(vals, 75):{meta['fmt']}}{s}",
                    f"{np.percentile(vals, 95):{meta['fmt']}}{s}",
                    f"{np.max(vals):{meta['fmt']}}{s}",
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
