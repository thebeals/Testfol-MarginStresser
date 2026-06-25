"""Risk & Return Metrics table — sourced from Testfol API or computed locally."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.ui.charts.stats_helpers import excess_kurtosis, skewness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val, decimals=2):
    if val is None:
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(v):
        return ""
    return f"{v:,.{decimals}f}%"


def _fmt_num(val, decimals=2):
    if val is None:
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(v):
        return ""
    return f"{v:,.{decimals}f}"


def _downside_dev(returns):
    """sqrt(mean(min(0, r)^2))"""
    neg = np.minimum(returns, 0)
    return np.sqrt(np.mean(neg ** 2))


def _ulcer_index(series):
    cummax = series.cummax()
    dd = (series - cummax) / cummax * 100
    return np.sqrt(np.mean(dd ** 2))


def _avg_drawdown(series):
    cummax = series.cummax()
    dd_pct = (series - cummax) / cummax * 100
    below = dd_pct[dd_pct < 0]
    return below.mean() if not below.empty else 0.0


def _longest_drawdown_years(series):
    cummax = series.cummax()
    in_dd = series < cummax
    if not in_dd.any():
        return 0.0
    groups = (~in_dd).cumsum()
    max_days = 0
    for _, group in in_dd.groupby(groups):
        if group.all() and len(group) > 0:
            duration = (group.index[-1] - group.index[0]).days
            max_days = max(max_days, duration)
    return max_days / 365.25


def _sortino(series):
    returns = series.pct_change().dropna()
    if returns.empty:
        return 0.0
    dd = _downside_dev(returns.values)
    if dd == 0:
        return 0.0
    return (returns.mean() / dd) * np.sqrt(252)


def _cvar(vals, pct):
    """Conditional Value-at-Risk: mean of values <= percentile cutoff."""
    cutoff = np.nanpercentile(vals, pct)
    tail = vals[vals <= cutoff]
    return np.mean(tail) if len(tail) > 0 else np.nan


def _compute_benchmark_metrics(port_series, bench_series):
    """Compute benchmark-relative metrics locally using SPYSIM.

    Returns a dict with: beta, alpha_daily, alpha_annualized, benchmark_corr,
    upside/downside capture ratios, active_return, tracking_error, information_ratio, m2.
    """
    if bench_series is None or bench_series.empty or port_series.empty:
        return {}

    # Daily returns aligned to common dates
    p_daily = port_series.pct_change().dropna()
    b_daily = bench_series.pct_change().dropna()
    common = p_daily.index.intersection(b_daily.index)
    if len(common) < 60:
        return {}
    p_d = p_daily.reindex(common).dropna()
    b_d = b_daily.reindex(common).dropna()
    common = p_d.index.intersection(b_d.index)
    p_d = p_d.reindex(common)
    b_d = b_d.reindex(common)

    result = {}

    # Correlation
    result["benchmark_corr"] = float(np.corrcoef(p_d.values, b_d.values)[0, 1])

    # Beta = Cov(p,b) / Var(b)
    cov_matrix = np.cov(p_d.values, b_d.values, ddof=1)
    b_var = cov_matrix[1, 1]
    beta = cov_matrix[0, 1] / b_var if b_var > 0 else 0.0
    result["beta"] = beta

    # Alpha (daily): mean(p - beta * b)
    alpha_d = float(np.mean(p_d.values - beta * b_d.values))
    result["alpha_daily"] = alpha_d * 100
    result["alpha_annualized"] = ((1 + alpha_d) ** 252 - 1) * 100

    # Active Return = portfolio CAGR - benchmark CAGR
    from app.core.calculations import calculate_cagr
    port_cagr = calculate_cagr(port_series)
    bench_cagr = calculate_cagr(bench_series.reindex(port_series.index).dropna())
    result["active_return"] = port_cagr - bench_cagr

    # Tracking Error = annualized std of excess daily returns
    excess = p_d - b_d
    te = float(excess.std() * np.sqrt(252) * 100)
    result["tracking_error"] = te

    # Information Ratio = Active Return / Tracking Error
    result["information_ratio"] = result["active_return"] / te if te > 0 else 0.0

    # M2 = Sharpe_port * Std_bench + Rf (assume Rf=0 for simplicity)
    p_std = p_d.std()
    b_std = b_d.std()
    if p_std > 0 and b_std > 0:
        sharpe_p = (p_d.mean() / p_std) * np.sqrt(252)
        result["m2"] = sharpe_p * (b_std * np.sqrt(252) * 100)
    else:
        result["m2"] = 0.0

    # Capture Ratios (daily, monthly, annual)
    for tag, freq in [("", None), ("_monthly", "ME"), ("_annual", "YE")]:
        if freq:
            p_r = port_series.resample(freq).last().pct_change().dropna()
            b_r = bench_series.resample(freq).last().pct_change().dropna()
            ci = p_r.index.intersection(b_r.index)
            p_r = p_r.reindex(ci)
            b_r = b_r.reindex(ci)
        else:
            p_r, b_r = p_d, b_d

        # Upside capture: mean(p when b>0) / mean(b when b>0)
        up_mask = b_r > 0
        down_mask = b_r < 0

        b_up = b_r[up_mask]
        p_up = p_r[up_mask]
        uc = (p_up.mean() / b_up.mean() * 100) if len(b_up) > 0 and b_up.mean() != 0 else None
        result[f"upside_capture{tag}"] = uc

        b_down = b_r[down_mask]
        p_down = p_r[down_mask]
        dc = (p_down.mean() / b_down.mean() * 100) if len(b_down) > 0 and b_down.mean() != 0 else None
        result[f"downside_capture{tag}"] = dc

    return result


# ---------------------------------------------------------------------------
# Build metrics list
# ---------------------------------------------------------------------------

def _build_metrics(port_series, stats, raw_response):
    """Return list of (metric_name, value_str) tuples."""
    rows = []

    def add(name, val):
        rows.append((name, val))

    # --- Extract API data ---
    api_resp = raw_response.get("raw_response") if isinstance(raw_response, dict) else None

    api_stats = None
    if api_resp:
        s = api_resp.get("stats", {})
        api_stats = s[0] if isinstance(s, list) and s else (s if isinstance(s, dict) else None)

    daily_rs = monthly_rs = annual_rs = None
    if api_resp:
        d = api_resp.get("daily_return_statistics", [])
        daily_rs = d[0] if d else None
        m = api_resp.get("monthly_return_statistics", [])
        monthly_rs = m[0] if m else None
        a = api_resp.get("annual_return_statistics", [])
        annual_rs = a[0] if a else None

    # --- Local returns (always computed as fallback) ---
    daily_r = port_series.pct_change().dropna()
    monthly_r = port_series.resample("ME").last().pct_change().dropna()
    annual_r = port_series.resample("YE").last().pct_change().dropna()

    # Convenience: local values in pct
    d_vals = daily_r.values * 100
    m_vals = monthly_r.values * 100
    a_vals = annual_r.values * 100

    # --- CAGR ---
    cagr = api_stats.get("cagr", 0.0) if api_stats else stats.get("cagr", 0.0)
    cagr = cagr if cagr is not None else 0.0
    cagr_d = cagr * 100 if abs(cagr) <= 1 else cagr
    add("CAGR", _fmt_pct(cagr_d))

    # --- Average returns ---
    add("Average Annual Return", _fmt_pct(annual_rs["mean"] if annual_rs else np.mean(a_vals)))
    add("Average Monthly Return", _fmt_pct(monthly_rs["mean"] if monthly_rs else np.mean(m_vals)))
    add("Average Daily Return", _fmt_pct(daily_rs["mean"] if daily_rs else np.mean(d_vals), 3))

    # --- Standard deviations ---
    # Annualized
    ann_daily_std = api_stats.get("std") if api_stats else stats.get("std", np.std(d_vals, ddof=1) * np.sqrt(252))
    add("Standard Deviation of Daily Returns (Annualized)", _fmt_pct(ann_daily_std))

    m_std_raw = monthly_rs["std"] if monthly_rs else np.std(m_vals, ddof=1)
    add("Standard Deviation of Monthly Returns (Annualized)", _fmt_pct(m_std_raw * np.sqrt(12)))

    add("Standard Deviation of Annual Returns", _fmt_pct(annual_rs["std"] if annual_rs else np.std(a_vals, ddof=1)))

    # Raw
    add("Standard Deviation of Daily Returns", _fmt_pct(daily_rs["std"] if daily_rs else np.std(d_vals, ddof=1)))
    add("Standard Deviation of Monthly Returns", _fmt_pct(m_std_raw))

    # --- Downside deviations ---
    dd_d = daily_rs["downside_dev"] if daily_rs else _downside_dev(d_vals)
    dd_m = monthly_rs["downside_dev"] if monthly_rs else _downside_dev(m_vals)
    dd_a = annual_rs["downside_dev"] if annual_rs else _downside_dev(a_vals)

    add("Downside Deviation of Daily Returns (Annualized)", _fmt_pct(dd_d * np.sqrt(252)))
    add("Downside Deviation of Monthly Returns (Annualized)", _fmt_pct(dd_m * np.sqrt(12)))
    add("Downside Deviation of Annual Returns", _fmt_pct(dd_a))
    add("Downside Deviation of Daily Returns", _fmt_pct(dd_d))
    add("Downside Deviation of Monthly Returns", _fmt_pct(dd_m))

    # --- Benchmark-relative metrics ---
    # Use API stats when available, otherwise compute locally vs SPYSIM
    bench_m = {}
    if not api_stats:
        from app.ui.charts.rolling import _fetch_spysim_series
        bench_series = _fetch_spysim_series()
        if not bench_series.empty:
            bench_m = _compute_benchmark_metrics(port_series, bench_series)

    bsrc = api_stats if api_stats else bench_m
    if bsrc:
        add("Benchmark Correlation", _fmt_num(bsrc.get("benchmark_corr")))
        add("Beta", _fmt_num(bsrc.get("beta")))
        add("Alpha (daily)", _fmt_pct(bsrc.get("alpha_daily"), 3))
        add("Alpha (annualized)", _fmt_pct(bsrc.get("alpha_annualized")))

        for tag, suffix in [("Daily", ""), ("Monthly", "_monthly"), ("Annual", "_annual")]:
            uc = bsrc.get(f"upside_capture{suffix}")
            dc = bsrc.get(f"downside_capture{suffix}")
            add(f"Upside Capture Ratio ({tag})", _fmt_pct(uc))
            add(f"Downside Capture Ratio ({tag})", _fmt_pct(dc))
            spread = (uc - dc) if uc is not None and dc is not None else None
            add(f"Capture Spread ({tag})", _fmt_pct(spread))

        add("Active Return", _fmt_pct(bsrc.get("active_return")))
        add("Tracking Error", _fmt_pct(bsrc.get("tracking_error")))
        add("Information Ratio", _fmt_num(bsrc.get("information_ratio")))
    else:
        for label in [
            "Benchmark Correlation", "Beta", "Alpha (daily)", "Alpha (annualized)",
            "Upside Capture Ratio (Daily)", "Downside Capture Ratio (Daily)", "Capture Spread (Daily)",
            "Upside Capture Ratio (Monthly)", "Downside Capture Ratio (Monthly)", "Capture Spread (Monthly)",
            "Upside Capture Ratio (Annual)", "Downside Capture Ratio (Annual)", "Capture Spread (Annual)",
            "Active Return", "Tracking Error", "Information Ratio",
        ]:
            add(label, "—")

    # --- Drawdown metrics ---
    max_dd = api_stats.get("max_drawdown") if api_stats else stats.get("max_drawdown", 0.0)
    add("Maximum Drawdown", _fmt_pct(max_dd))
    add("Average Drawdown", _fmt_pct(api_stats["avg_drawdown"] if api_stats and api_stats.get("avg_drawdown") is not None else _avg_drawdown(port_series)))

    longest = api_stats.get("max_drawdown_years") if api_stats else _longest_drawdown_years(port_series)
    add("Longest Drawdown", f"{longest:.2f}y" if longest is not None else "")

    # --- Risk-adjusted metrics ---
    sharpe = api_stats.get("sharpe") if api_stats else stats.get("sharpe", 0.0)
    add("Sharpe Ratio", _fmt_num(sharpe))

    m2_val = api_stats.get("m2") if api_stats else bench_m.get("m2")
    if m2_val is not None:
        add("Modigliani\u2013Modigliani Measure (M\u00b2)", _fmt_pct(m2_val))
    else:
        add("Modigliani\u2013Modigliani Measure (M\u00b2)", "—")

    sortino = api_stats.get("sortino") if api_stats else _sortino(port_series)
    add("Sortino Ratio", _fmt_num(sortino))

    calmar = api_stats.get("calmar") if api_stats else (abs(cagr_d / max_dd) if max_dd and max_dd != 0 else 0.0)
    add("Calmar Ratio", _fmt_num(calmar))

    ulcer = api_stats.get("ulcer_index") if api_stats else _ulcer_index(port_series)
    ulcer = ulcer if ulcer is not None else 0.0
    add("Ulcer Index", _fmt_num(ulcer))

    upi = api_stats.get("upi") if api_stats else (cagr_d / ulcer if ulcer > 0 else 0.0)
    add("Ulcer Performance Index (UPI)", _fmt_num(upi))

    if api_stats and api_stats.get("diversification_ratio") is not None:
        add("Diversification Ratio", _fmt_num(api_stats["diversification_ratio"]))
    else:
        add("Diversification Ratio", "—")

    # --- % Positive ---
    add("% of Positive Days", _fmt_pct(daily_rs["positive_pct"] if daily_rs else (daily_r > 0).mean() * 100))
    add("% of Positive Months", _fmt_pct(monthly_rs["positive_pct"] if monthly_rs else (monthly_r > 0).mean() * 100))
    add("% of Positive Years", _fmt_pct(annual_rs["positive_pct"] if annual_rs else (annual_r > 0).mean() * 100))

    # --- VaR / CVaR / Max / Min  per period ---
    for tag, rs, vals, dec in [("Daily", daily_rs, d_vals, 3), ("Monthly", monthly_rs, m_vals, 2), ("Annual", annual_rs, a_vals, 2)]:
        if rs:
            add(f"{tag} Return Value-at-Risk (1%)", _fmt_pct(rs.get("p1"), dec))
            add(f"{tag} Return Value-at-Risk (5%)", _fmt_pct(rs.get("p5"), dec))
            add(f"{tag} Return Value-at-Risk (10%)", _fmt_pct(rs.get("p10"), dec))
            add(f"{tag} Return Conditional Value-at-Risk (1%)", _fmt_pct(rs.get("cvar_1"), dec))
            add(f"{tag} Return Conditional Value-at-Risk (5%)", _fmt_pct(rs.get("cvar_5"), dec))
            add(f"{tag} Return Conditional Value-at-Risk (10%)", _fmt_pct(rs.get("cvar_10"), dec))
            add(f"Maximum {tag} Return", _fmt_pct(rs.get("max"), dec))
            add(f"Minimum {tag} Return", _fmt_pct(rs.get("min"), dec))
        elif len(vals) > 0:
            add(f"{tag} Return Value-at-Risk (1%)", _fmt_pct(np.nanpercentile(vals, 1), dec))
            add(f"{tag} Return Value-at-Risk (5%)", _fmt_pct(np.nanpercentile(vals, 5), dec))
            add(f"{tag} Return Value-at-Risk (10%)", _fmt_pct(np.nanpercentile(vals, 10), dec))
            add(f"{tag} Return Conditional Value-at-Risk (1%)", _fmt_pct(_cvar(vals, 1), dec))
            add(f"{tag} Return Conditional Value-at-Risk (5%)", _fmt_pct(_cvar(vals, 5), dec))
            add(f"{tag} Return Conditional Value-at-Risk (10%)", _fmt_pct(_cvar(vals, 10), dec))
            add(f"Maximum {tag} Return", _fmt_pct(np.nanmax(vals), dec))
            add(f"Minimum {tag} Return", _fmt_pct(np.nanmin(vals), dec))

    # --- Skewness / Kurtosis ---
    for tag, rs, vals in [("Daily", daily_rs, daily_r.values), ("Monthly", monthly_rs, monthly_r.values), ("Annual", annual_rs, annual_r.values)]:
        if rs:
            add(f"Skewness of {tag} Returns", _fmt_num(rs.get("skewness")))
        elif len(vals) > 2:
            add(f"Skewness of {tag} Returns", _fmt_num(skewness(vals)))
        else:
            add(f"Skewness of {tag} Returns", "")

    for tag, rs, vals in [("Daily", daily_rs, daily_r.values), ("Monthly", monthly_rs, monthly_r.values), ("Annual", annual_rs, annual_r.values)]:
        if rs:
            add(f"Excess Kurtosis of {tag} Returns", _fmt_num(rs.get("kurtosis")))
        elif len(vals) > 2:
            add(f"Excess Kurtosis of {tag} Returns", _fmt_num(excess_kurtosis(vals)))
        else:
            add(f"Excess Kurtosis of {tag} Returns", "")

    # --- Avg Gain / Loss / Ratio ---
    for tag, rs, rets_pct in [("Daily", daily_rs, d_vals), ("Monthly", monthly_rs, m_vals), ("Annual", annual_rs, a_vals)]:
        if rs:
            add(f"Average {tag} Gain", _fmt_pct(rs.get("avg_gain")))
            add(f"Average {tag} Loss", _fmt_pct(rs.get("avg_loss")))
            add(f"Gain/Loss Ratio ({tag})", _fmt_num(rs.get("gain_loss_ratio")))
        elif len(rets_pct) > 0:
            pos = rets_pct[rets_pct > 0]
            neg = rets_pct[rets_pct < 0]
            avg_g = np.mean(pos) if len(pos) > 0 else 0.0
            avg_l = np.mean(neg) if len(neg) > 0 else 0.0
            ratio = abs(avg_g / avg_l) if avg_l != 0 else 0.0
            add(f"Average {tag} Gain", _fmt_pct(avg_g))
            add(f"Average {tag} Loss", _fmt_pct(avg_l))
            add(f"Gain/Loss Ratio ({tag})", _fmt_num(ratio))

    # --- Withdrawal Rates (API only) ---
    for horizon in [10, 20, 30, 40]:
        swr_val = pwr_val = ""
        if api_resp:
            swr_d = api_resp.get(f"swr_stats_{horizon}", [])
            if swr_d and isinstance(swr_d, list) and swr_d[0]:
                v = swr_d[0].get("wr_100")
                if v is not None:
                    swr_val = _fmt_pct(v)
            pwr_d = api_resp.get(f"pwr_stats_{horizon}", [])
            if pwr_d and isinstance(pwr_d, list) and pwr_d[0]:
                v = pwr_d[0].get("wr_100")
                if v is not None:
                    pwr_val = _fmt_pct(v)
        add(f"{horizon}Y Safe Withdrawal Rate", swr_val)
        add(f"{horizon}Y Perpetual Withdrawal Rate", pwr_val)

    return rows


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_risk_return_metrics(port_series, stats, raw_response=None, unique_id=""):
    """Render the full Risk & Return Metrics table under Summary."""
    if port_series.empty:
        return

    metrics = _build_metrics(port_series, stats, raw_response or {})
    if not metrics:
        return

    st.subheader("Risk and Return Metrics")

    df = pd.DataFrame(metrics, columns=["Metric", "Value"])

    def _color(val):
        if isinstance(val, str) and val.strip().startswith("-"):
            return "color: #EF553B"
        return ""

    styled = df.style.map(_color, subset=["Value"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(len(df) * 35 + 40, 2800),
    )
