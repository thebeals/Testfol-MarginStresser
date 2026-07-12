from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import requests

from app.common.cache import cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)

API_URL = "https://testfol.io/api/backtest"

def table_to_dicts(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """
    Converts the allocation dataframe to dictionaries for allocation and maintenance.
    """
    df = df.dropna(subset=["Ticker"]).copy()
    alloc = {r["Ticker"].strip(): float(r["Weight %"]) for _,r in df.iterrows()}
    maint = {r["Ticker"].split("?")[0].strip(): float(r["Maint %"]) for _,r in df.iterrows()}
    return alloc, maint

def fetch_backtest(
    start_date,
    end_date,
    start_val: float,
    cashflow: float,
    cashfreq: str,
    rolling: int,
    invest_div: bool,
    rebalance: str,
    allocation: dict[str, float],
    return_raw: bool = False,
    include_raw: bool = False,
    rebalance_offset: int = 0,
    cashflow_offset: int = 0,
    **kwargs,
) -> tuple[pd.Series, dict, dict] | dict:
    """
    Fetches backtest data from testfol.io API with universal disk caching.
    """
    # 1. Build cache key (excludes bearer_token/kwargs for deterministic hashing)
    cache_payload = {
        "start_date": str(start_date), "end_date": str(end_date),
        "start_val": start_val, "cashflow": cashflow, "cashfreq": cashfreq,
        "rolling": rolling, "invest_div": invest_div, "rebalance": rebalance,
        "allocation": allocation, "return_raw": return_raw,
        "include_raw": include_raw, "rebalance_offset": rebalance_offset,
        "cashflow_offset": cashflow_offset,
    }
    req_hash = cache_key(json.dumps(cache_payload, sort_keys=True, default=str))

    # 2. Try cache load (ttl=0 → no expiry for deterministic API responses)
    cached = cache_get(req_hash, ttl=0)
    if cached is not None:
        return cached
            
    # 3. API Request (Cache Miss)
    # Format dates as YYYY-MM-DD (API requirement)
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    
    payload = {
        "start_date": start_str,
        "end_date":   end_str,
        "start_val":  start_val,
        "adj_inflation": False,
        "cashflow": cashflow,
        "cashflow_freq": cashfreq,
        "cashflow_offset": cashflow_offset,
        "rolling_window": rolling,
        "backtests": [{
            "invest_dividends": invest_div,
            "rebalance_freq":   rebalance,
            "rebalance_offset": rebalance_offset,
            "allocation":       allocation,
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Referer": "https://testfol.io/",
        "Origin": "https://testfol.io"
    }
    # Check for Bearer Token (Arg > Auto-login > Env)
    from app.services.testfol_auth import get_token as _get_auth_token
    token = kwargs.get('bearer_token') or _get_auth_token()
    if token:
        # Sanitize: Remove 'Bearer ' prefix if user pasted it
        if token.startswith("Bearer "):
            token = token.replace("Bearer ", "", 1).strip()
            
        headers["Authorization"] = f"Bearer {token}"
        
    r = None
    try:
        # Configure Retry Strategy with Exponential Backoff
        # 429: Rate Limit, 5xx: Server Errors
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=5,  # 5 retries
            backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        r = session.post(API_URL, json=payload, headers=headers, timeout=45)
        r.raise_for_status()

    except requests.exceptions.RetryError:
         raise requests.exceptions.HTTPError(f"Max retries exceeded for {API_URL}")
    except requests.exceptions.HTTPError as e:
        # Include response text in error message for debugging
        error_msg = f"HTTP Error: {e}\nResponse: {r.text if r is not None else 'No Response'}"
        raise requests.exceptions.HTTPError(error_msg, response=r if r is not None else None)
    except Exception as e:
        raise e


    def _format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    logger.debug(f"API Success {req_hash} (Msg Size: {_format_size(len(r.content))})")
    resp = r.json()
    
    if return_raw:
        result = resp
        cache_set(req_hash, result)
        return result
    
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    
    # Validation: Ensure charts exist and are well-formed
    if "charts" not in resp or "history" not in resp["charts"]:
         raise ValueError("Invalid API response: missing chart history")

    history = resp["charts"]["history"]
    if not isinstance(history, (list, tuple)) or len(history) != 2:
        raise ValueError(f"Invalid API response: history must be a 2-element list, got {type(history).__name__}")

    ts, vals = history
    if not isinstance(ts, list) or not isinstance(vals, list):
        raise ValueError("Invalid API response: history timestamps/values must be lists")
    if len(ts) != len(vals):
        raise ValueError(f"Invalid API response: timestamps ({len(ts)}) and values ({len(vals)}) length mismatch")
    if not ts:
        raise ValueError("Invalid API response: empty history data")
    dates = pd.to_datetime(ts, unit="s")
    
    extra_data = {
        "rebalancing_events": resp.get("rebalancing_events", []),
        "rebalancing_stats": resp.get("rebalancing_stats", []),
        "daily_returns": resp.get("daily_returns", [])
    }
    
    if include_raw:
        extra_data["raw_response"] = resp
    
    # Construct Result Tuple
    result = (pd.Series(vals, index=dates, name="Portfolio"), stats, extra_data)
    
    # 4. Save to Cache
    cache_set(req_hash, result)
    
    return result

def simulate_margin(
    port: pd.Series,
    starting_loan: float,
    rate_annual: float | dict,
    draw_monthly: float,
    maint_pct: float,
    tax_series: pd.Series | None = None,
    repayment_series: pd.Series | None = None,
    draw_start_date=None,
    draw_monthly_retirement: float = 0.0,
    retirement_date=None,
    dca_series: pd.Series | None = None,
    fund_dca_margin: bool = False,
    accrual_start_date=None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simulate real USD margin debt using Actual/360 calendar-day accrual.

    Accrued interest is included in account liability immediately, while
    capitalization into settled principal occurs on the third observed trading
    day of the following month. Synthetic LETF financing intentionally uses a
    separate Testfol-compatible /252 calculation in ``shadow_backtest.py``.
    """
    from app.core.margin_interest import UsdMarginInterestLedger

    _log = logging.getLogger("margin_sim")
    _log.info(
        "simulate_margin Actual/360: loan=$%.0f draw=$%.0f/mo ret_draw=$%.0f/mo maint=%.1f%%",
        starting_loan,
        draw_monthly,
        draw_monthly_retirement,
        maint_pct * 100,
    )

    if port.empty:
        empty = pd.Series(dtype=float, index=port.index)
        return (
            empty.rename("Loan"),
            empty,
            empty.rename("Equity %"),
            empty.rename("Margin usage %"),
            empty.rename("Margin Rate %"),
        )

    port = port.sort_index()
    cashflows = pd.Series(0.0, index=port.index)

    if draw_monthly > 0 or draw_monthly_retirement > 0:
        months = port.index.month.values
        month_changes = months != np.roll(months, -1)
        month_changes[-1] = False
        if draw_start_date is not None:
            month_changes &= np.array(port.index >= pd.Timestamp(draw_start_date))
        if draw_monthly_retirement > 0 and retirement_date is not None:
            after_ret = np.array(port.index >= pd.Timestamp(retirement_date))
            cashflows.values[month_changes & ~after_ret] += draw_monthly
            cashflows.values[month_changes & after_ret] += draw_monthly_retirement
        elif draw_monthly_retirement > 0 and retirement_date is None:
            cashflows.values[month_changes] += draw_monthly_retirement
        else:
            cashflows.values[month_changes] += draw_monthly

    if tax_series is not None:
        cashflows += tax_series.reindex(port.index, fill_value=0.0)
    if repayment_series is not None:
        cashflows -= repayment_series.reindex(port.index, fill_value=0.0)

    dca_values = None
    if dca_series is not None:
        dca_values = dca_series.reindex(port.index, fill_value=0.0).values

    if accrual_start_date is None:
        ledger_start = pd.Timestamp(port.index[0]).normalize()
    else:
        ledger_start = max(
            pd.Timestamp(port.index[0]).normalize(),
            pd.Timestamp(accrual_start_date).normalize(),
        )
    ledger_dates = port.index[port.index >= ledger_start]
    ledger = UsdMarginInterestLedger(ledger_dates, starting_loan, rate_annual)
    loan_values = np.zeros(len(port), dtype=float)
    rate_values = np.zeros(len(port), dtype=float)
    ledger_started = False

    for idx, date in enumerate(port.index):
        if date < ledger_start:
            loan_values[idx] = starting_loan
            rate_values[idx] = 0.0
            continue

        if not ledger_started and date > ledger_start:
            # The requested start can be a weekend/holiday absent from the
            # portfolio index. Accrue that calendar day before the next session.
            ledger.advance(ledger_start)
            ledger_started = True

        dca_loan_change = 0.0
        if dca_values is not None:
            if fund_dca_margin:
                dca_loan_change = float(dca_values[idx])
            elif ledger.total_liability < 0:
                dca_loan_change = min(float(dca_values[idx]), abs(ledger.total_liability))

        snapshot = ledger.advance(
            date,
            loan_change=float(cashflows.iloc[idx]) + dca_loan_change,
        )
        ledger_started = True
        loan_values[idx] = snapshot.total_liability
        rate_values[idx] = snapshot.effective_rate_pct

    loan_series = pd.Series(loan_values, index=port.index, name="Loan")
    effective_rate_series = pd.Series(rate_values, index=port.index, name="Margin Rate %")
    equity = port - loan_series
    safe_port = port.replace(0, np.nan)
    equity_pct = (equity / safe_port).fillna(0).rename("Equity %")
    usage_pct = (loan_series / (safe_port * (1 - maint_pct))).fillna(0).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct, effective_rate_series
