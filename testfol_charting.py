#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# Streamlit App for Testfol Backtesting with Multi-Portfolio Support
# Uses FastAPI backend when available, falls back to in-process computation.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import pandas as pd
import streamlit as st
import requests
from app.common import utils
from app.core.result_validation import has_stale_local_cashflow_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
from app.ui import render_sidebar, render_config, render_results, asset_explorer, charts

BACKEND_URL = "http://localhost:8100"

# ─────────────────────────────────────────────────────────────────────────────
# Backend Detection
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _backend_available():
    """Check if FastAPI backend is reachable (cached 30s)."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=1)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Serialization Helpers (API mode)
# ─────────────────────────────────────────────────────────────────────────────

def _deser_series(json_str):
    """Deserialize a JSON string (split orient) back to a pandas Series."""
    if not json_str:
        return pd.Series(dtype=float)
    from io import StringIO
    s = pd.read_json(StringIO(json_str), orient="split", typ="series")
    s.index = pd.to_datetime(s.index)
    return s


def _deser_df(json_str, datetime_index=True):
    """Deserialize a JSON string (split orient) back to a pandas DataFrame."""
    if not json_str:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_json(StringIO(json_str), orient="split")
    if not df.empty and datetime_index:
        df.index = pd.to_datetime(df.index)
    return df


def _deserialize_result(item):
    """Convert a BacktestResult dict from JSON response into the format render_results expects."""
    port_name = item.get("name", "Portfolio")
    series = _deser_series(item.get("series_json"))
    series.name = port_name
    twr = _deser_series(item.get("twr_series_json"))
    dr_df = _deser_df(item.get("daily_returns_df_json"))
    trades = _deser_df(item.get("trades_df_json"), datetime_index=False)
    pl = _deser_df(item.get("pl_by_year_json"), datetime_index=False)
    comp = _deser_df(item.get("composition_df_json"), datetime_index=False)
    unrealized = _deser_df(item.get("unrealized_pl_df_json"))
    prices = _deser_df(item.get("component_prices_json"))

    return {
        "name": port_name,
        "series": series,
        "port_series": series,
        "stats": item.get("stats", {}),
        "twr_series": twr if not twr.empty else None,
        "daily_returns_df": dr_df if not dr_df.empty else None,
        "is_local": item.get("is_local", False),
        "api_failover": item.get("api_failover", False),
        "trades": trades,
        "trades_df": trades,
        "pl_by_year": pl,
        "unrealized_pl_df": unrealized,
        "component_prices": prices,
        "allocation": item.get("allocation", {}),
        "logs": item.get("logs", []),
        "composition": comp,
        "composition_df": comp,
        "raw_response": item.get("raw_response", {}),
        "start_val": item.get("start_val", 10000.0),
        "sim_range": item.get("sim_range", ""),
        "shadow_range": item.get("shadow_range", ""),
        "wmaint": item.get("wmaint", 0.25),
        "wmaint_pm": item.get("wmaint_pm", 0.0),
        "pm_blocked_dates": item.get("pm_blocked_dates", []),
        "_dca_config": item.get("_dca_config", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Payload Builder (API mode)
# ─────────────────────────────────────────────────────────────────────────────

_TAX_KEYS = {"pay_tax_margin", "pay_tax_cash", "other_income", "filing_status", "state_code", "tax_method"}


def _build_payload(config, start_date, end_date, bearer_token):
    """Build a MultiBacktestRequest payload from Streamlit config."""
    portfolios_cfg = _get_portfolios_cfg(config)

    api_portfolios = []
    d_maint = config.get("default_maint", 25.0)

    for p in portfolios_cfg:
        alloc_df = p.get("alloc_df", pd.DataFrame())
        if alloc_df.empty:
            continue

        alloc_map = dict(zip(alloc_df["Ticker"], alloc_df["Weight %"]))
        maint_pcts = {}
        pm_maint_pcts = {}
        for _, row in alloc_df.iterrows():
            ticker = row["Ticker"].split("?")[0]
            maint_pcts[ticker] = float(row.get("Maint %", d_maint))
            pm_val = float(row.get("PM Maint %", 0))
            if pm_val > 0:
                pm_maint_pcts[ticker] = pm_val

        reb = p.get("rebalance", {})
        dca = p.get("dca", {})
        api_portfolios.append({
            "name": p.get("name", "Portfolio"),
            "allocation": alloc_map,
            "maint_pcts": maint_pcts,
            "pm_maint_pcts": pm_maint_pcts,
            "rebalance": {
                "mode": reb.get("mode", "Standard"),
                "freq": reb.get("freq", "Yearly"),
                "month": reb.get("month", 1),
                "day": reb.get("day", 1),
                "compare_std": reb.get("compare_std", False),
                "threshold_pct": reb.get("threshold_pct", 5.0),
            },
            "dca": {
                "mode": dca.get("mode", "Proportional"),
                "target_ticker": dca.get("target_ticker", ""),
            },
        })

    gcf = config.get("global_cashflow", {})

    # Approximate effective margin rate for buy-block loan estimation
    _rate_cfg = config.get("rate_annual", 8.0)
    if isinstance(_rate_cfg, dict):
        _rtype = _rate_cfg.get("type", "Fixed")
        if _rtype == "Fixed":
            _approx_rate = _rate_cfg.get("rate_pct", 5.0)
        else:
            _tiers = _rate_cfg.get("tiers", [])
            _spread = _tiers[0][1] if _tiers else _rate_cfg.get("spread_pct", 1.0)
            _approx_rate = 4.5 + _spread
    else:
        _approx_rate = float(_rate_cfg)

    _draw_start = config.get("draw_start_date", None)
    pm_config = {
        "pm_buy_block": config.get("pm_buy_block", False),
        "pm_buy_block_threshold": config.get("pm_buy_block_threshold", 100000.0),
        "starting_loan": config.get("starting_loan", 0.0) - config.get("starting_cash", 0.0),
        "draw_monthly": config.get("draw_monthly", 0.0),
        "draw_monthly_retirement": config.get("draw_monthly_retirement", 0.0),
        "draw_start_date": str(_draw_start) if _draw_start is not None else None,
        "retirement_date": str(config["retirement_date"]) if config.get("retirement_date") else None,
        "dca_in_retirement": config.get("dca_in_retirement", True),
        "margin_rate_annual": _approx_rate,
        "cashflow_for_loan": gcf.get("amount", 0.0),
        "cashflow_freq": gcf.get("freq", "Monthly"),
        "fund_dca_margin": gcf.get("fund_dca_margin", False),
        "retirement_income": config.get("retirement_income", None),
    }

    payload = {
        "portfolios": api_portfolios,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "cashflow": {
            "start_val": gcf.get("start_val", 10000.0),
            "amount": gcf.get("amount", 0.0),
            "freq": gcf.get("freq", "Monthly"),
            "invest_div": gcf.get("invest_div", True),
            "pay_down_margin": gcf.get("pay_down_margin", False),
        },
        "tax_config": {k: v for k, v in config.items() if k in _TAX_KEYS},
        "bearer_token": bearer_token,
        "pm_config": pm_config,
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# In-Process Fallback (for Streamlit Cloud / no backend)
# ─────────────────────────────────────────────────────────────────────────────

def _get_portfolios_cfg(config):
    """Normalize portfolio config (handles legacy single-portfolio format)."""
    portfolios_cfg = config.get("portfolios", [])
    if not portfolios_cfg:
        portfolios_cfg = [{
            "id": "legacy",
            "name": "Portfolio 1",
            "alloc_df": config.get("edited_df", pd.DataFrame([{"Ticker": "SPY", "Weight %": 100, "Maint %": 25}])),
            "rebalance": {
                "mode": "Custom" if config.get("sim_engine") == "hybrid" else "Standard",
                "freq": config.get("custom_freq", "Yearly"),
                "month": config.get("rebalance_month", 1),
                "day": config.get("rebalance_day", 1),
                "compare_std": config.get("compare_standard", False),
            },
            "cashflow": {
                "start_val": config.get("start_val", 10000),
                "amount": config.get("cashflow", 0),
                "freq": config.get("cashfreq", "Monthly"),
                "invest_div": config.get("invest_div", True),
                "pay_down_margin": config.get("pay_down_margin", False),
                "fund_dca_margin": config.get("fund_dca_margin", False),
            },
            "dca": {
                "mode": "Proportional",
                "target_ticker": "",
            },
        }]
    return portfolios_cfg


@st.cache_data(show_spinner="Fetching data from Testfol API...", ttl=3600)
def _cached_fetch_backtest(*args, **kwargs):
    from app.services import fetch_backtest
    return fetch_backtest(*args, **kwargs)


@st.cache_data(show_spinner="Running Shadow Backtest...", ttl=3600)
def _cached_run_shadow_backtest(*args, _v="local-dca-routing-v4", **kwargs):
    from app.core import run_shadow_backtest
    return run_shadow_backtest(*args, **kwargs)


def _run_inprocess(config, start_date, end_date, bearer_token):
    """
    Run backtests in-process (no FastAPI). Delegates to the shared orchestrator
    with @st.cache_data-wrapped functions for Streamlit caching.
    """
    from app.core.backtest_orchestrator import run_multi_backtest

    portfolios_cfg = _get_portfolios_cfg(config)
    d_maint = config.get("default_maint", 25.0)

    # Convert Streamlit portfolio config to plain dicts for the orchestrator
    portfolios_plain = []
    for p in portfolios_cfg:
        alloc_df = p.get("alloc_df", pd.DataFrame())
        if alloc_df.empty:
            continue

        alloc_map = dict(zip(alloc_df["Ticker"], alloc_df["Weight %"]))
        maint_pcts = {}
        pm_maint_pcts = {}
        for _, row in alloc_df.iterrows():
            ticker = row["Ticker"].split("?")[0]
            maint_pcts[ticker] = float(row.get("Maint %", d_maint))
            pm_val = float(row.get("PM Maint %", 0))
            if pm_val > 0:
                pm_maint_pcts[ticker] = pm_val

        reb = p.get("rebalance", {})
        dca = p.get("dca", {})
        portfolios_plain.append({
            "name": p.get("name", "Portfolio"),
            "allocation": alloc_map,
            "maint_pcts": maint_pcts,
            "pm_maint_pcts": pm_maint_pcts,
            "rebalance": {
                "mode": reb.get("mode", "Standard"),
                "freq": reb.get("freq", "Yearly"),
                "month": reb.get("month", 1),
                "day": reb.get("day", 1),
                "compare_std": reb.get("compare_std", False),
                "threshold_pct": reb.get("threshold_pct", 5.0),
            },
            "dca": {
                "mode": dca.get("mode", "Proportional"),
                "target_ticker": dca.get("target_ticker", ""),
            },
        })

    if not portfolios_plain:
        return [], []

    gcf = config.get("global_cashflow", {})
    tax_config = {k: v for k, v in config.items() if k in _TAX_KEYS}

    # Approximate effective margin rate for buy-block loan estimation
    _rate_cfg = config.get("rate_annual", 8.0)
    if isinstance(_rate_cfg, dict):
        _rtype = _rate_cfg.get("type", "Fixed")
        if _rtype == "Fixed":
            _approx_rate = _rate_cfg.get("rate_pct", 5.0)
        else:
            # Variable/Tiered: base ~current fed + lowest spread
            _tiers = _rate_cfg.get("tiers", [])
            _spread = _tiers[0][1] if _tiers else _rate_cfg.get("spread_pct", 1.0)
            _approx_rate = 4.5 + _spread  # conservative fed funds estimate
    else:
        _approx_rate = float(_rate_cfg)

    pm_config = {
        "pm_buy_block": config.get("pm_buy_block", False),
        "pm_buy_block_threshold": config.get("pm_buy_block_threshold", 100000.0),
        "starting_loan": config.get("starting_loan", 0.0) - config.get("starting_cash", 0.0),
        "draw_monthly": config.get("draw_monthly", 0.0),
        "draw_monthly_retirement": config.get("draw_monthly_retirement", 0.0),
        "draw_start_date": config.get("draw_start_date", None),
        "retirement_date": config.get("retirement_date", None),
        "dca_in_retirement": config.get("dca_in_retirement", True),
        "margin_rate_annual": _approx_rate,
        "cashflow_for_loan": gcf.get("amount", 0.0),
        "cashflow_freq": gcf.get("freq", "Monthly"),
        "fund_dca_margin": gcf.get("fund_dca_margin", False),
        "retirement_income": config.get("retirement_income", None),
    }

    results_list, bench_series_list = run_multi_backtest(
        portfolios=portfolios_plain,
        start_date=str(start_date),
        end_date=str(end_date),
        start_val=gcf.get("start_val", 10000.0),
        cashflow_amount=gcf.get("amount", 0.0),
        cashflow_freq=gcf.get("freq", "Monthly"),
        invest_div=gcf.get("invest_div", True),
        pay_down_margin=gcf.get("pay_down_margin", False),
        tax_config=tax_config,
        bearer_token=bearer_token,
        fetch_backtest_fn=_cached_fetch_backtest,
        run_shadow_fn=_cached_run_shadow_backtest,
        pm_config=pm_config,
    )

    return results_list, bench_series_list


# ─────────────────────────────────────────────────────────────────────────────
# API Mode Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_via_api(config, start_date, end_date, bearer_token):
    """Run backtests via FastAPI backend. Returns (results_list, bench_series_list)."""
    payload = _build_payload(config, start_date, end_date, bearer_token)

    if not payload.get("portfolios"):
        st.warning("No valid portfolios configured. Add at least one portfolio with allocations.")
        return None, None

    resp = requests.post(
        f"{BACKEND_URL}/api/backtest/multi",
        json=payload,
        timeout=120,
    )

    if not resp.ok:
        st.error(f"Backend error ({resp.status_code}): {resp.text[:500]}")
        return None, None

    data = resp.json()
    results_list = [_deserialize_result(r) for r in data.get("results", [])]
    bench_series_list = []
    for b in data.get("bench_results", []):
        s = _deser_series(b.get("series_json"))
        if not s.empty:
            s.name = b.get("name", "Benchmark")
            bench_series_list.append(s)

    if has_stale_local_cashflow_series(results_list, payload.get("cashflow", {})):
        st.warning(
            "Backend cache returned a stale local portfolio curve without DCA deposits. "
            "Recomputing locally with fresh cashflow-aware series."
        )
        return _run_inprocess(config, start_date, end_date, bearer_token)

    return results_list, bench_series_list


# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Testfol Charting", layout="wide", page_icon="📈")

# --- Navigation ---
mode = st.sidebar.radio(
    "Navigation",
    ["Simulator", "Docs", "Changelog"],
    horizontal=True,
)

if mode == "Docs":
    utils.render_documentation()
    st.stop()
if mode == "Changelog":
    utils.render_changelog()
    st.stop()

# --- Sidebar ---
start_date, end_date, bearer_token, run_placeholder = render_sidebar()

# --- Main Area ---
config = render_config()

# --- Validation & Run ---
auto_run = st.session_state.pop("_auto_run_backtest", False)

if run_placeholder.button("🚀 Run Backtest", type="primary", use_container_width=True) or auto_run:
    st.divider()
    with st.spinner("Running Simulations..."):
        try:
            use_api = _backend_available()

            if use_api:
                results_list, bench_series_list = _run_via_api(config, start_date, end_date, bearer_token)
            else:
                results_list, bench_series_list = _run_inprocess(config, start_date, end_date, bearer_token)

            if results_list is not None:
                st.session_state.results_list = results_list
                st.session_state.bench_series_list = bench_series_list or []
            else:
                st.warning("Backtest returned no results. Check your portfolio configuration and date range.")

        except requests.exceptions.ConnectionError:
            # API was detected but went down mid-request — retry in-process
            try:
                results_list, bench_series_list = _run_inprocess(config, start_date, end_date, bearer_token)
                st.session_state.results_list = results_list
                st.session_state.bench_series_list = bench_series_list or []
            except Exception as e2:
                st.error(f"Error running backtest: {e2}")
                import traceback
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- Render Results (Outside Button Logic to Persist) ---
if "results_list" in st.session_state and st.session_state.results_list:
    results_list = st.session_state.results_list
    bench_series_list = st.session_state.get("bench_series_list", [])

    # Notify user if any portfolio fell back from Testfol API to local engine
    failover_names = [r["name"] for r in results_list if r.get("api_failover")]
    if failover_names:
        st.warning(
            f"Testfol API was unavailable. The following portfolios were computed locally "
            f"using {'Polygon.io' if os.environ.get('POLYGON_API_KEY') else 'yfinance'} price data: "
            f"**{', '.join(failover_names)}**. Results may differ slightly from Testfol."
        )

    if has_stale_local_cashflow_series(results_list, config.get("global_cashflow", {})):
        st.warning(
            "Detected stale local portfolio results where the value curve matches TWR "
            "instead of DCA-funded balance. Recomputing locally now."
        )
        with st.spinner("Refreshing stale local portfolio curves..."):
            try:
                results_list, bench_series_list = _run_inprocess(config, start_date, end_date, bearer_token)
                st.session_state.results_list = results_list
                st.session_state.bench_series_list = bench_series_list or []
            except Exception as stale_exc:
                st.error(f"Error refreshing stale local results: {stale_exc}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    charts.render_multi_portfolio_chart(
        results_list,
        benchmarks=bench_series_list,
        log_scale=config.get('log_scale', True),
        cashflow_config=config.get("global_cashflow", {}),
    )

    st.divider()

    active_idx = st.session_state.get('active_tab_idx', 0)
    if active_idx >= len(results_list):
        active_idx = 0

    start_dates = []
    for r in results_list:
        s = r.get('series')
        if s is not None and not s.empty:
            start_dates.append(s.index[0])
    common_start = max(start_dates) if start_dates else None

    res = results_list[active_idx]
    # Attach benchmark to individual result for per-portfolio comparison
    if bench_series_list and active_idx < len(bench_series_list):
        res["bench_series"] = bench_series_list[active_idx]
    st.markdown(f"### 📋 {res['name']} Details")
    render_results(res, config, portfolio_name=res['name'], clip_start_date=common_start)
