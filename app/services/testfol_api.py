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
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Simulates margin loan and calculates equity/usage metrics.
    """
    _log = logging.getLogger("margin_sim")
    _log.info("simulate_margin: loan=$%.0f draw=$%.0f/mo ret_draw=$%.0f/mo draw_start=%s ret_date=%s maint=%.1f%%",
              starting_loan, draw_monthly, draw_monthly_retirement, draw_start_date, retirement_date, maint_pct * 100)
    # 1. Create Cashflow Series (Draws + Taxes)
    # Initialize with zeros
    cashflows = pd.Series(0.0, index=port.index)
    
    # Add Monthly Draws (pre-retirement and retirement)
    if draw_monthly > 0 or draw_monthly_retirement > 0:
        months = port.index.month.values
        month_changes = months != np.roll(months, -1)
        month_changes[-1] = False  # Last day has no next day to compare
        if draw_start_date is not None:
            after_start = np.array(port.index >= pd.Timestamp(draw_start_date))
            month_changes = month_changes & after_start
        if draw_monthly_retirement > 0 and retirement_date is not None:
            after_ret = np.array(port.index >= pd.Timestamp(retirement_date))
            pre_ret = month_changes & ~after_ret
            post_ret = month_changes & after_ret
            cashflows.values[pre_ret] += draw_monthly
            cashflows.values[post_ret] += draw_monthly_retirement
        elif draw_monthly_retirement > 0 and retirement_date is None:
            # retirement_date not set — use retirement draw for all periods
            cashflows.values[month_changes] += draw_monthly_retirement
        else:
            cashflows.values[month_changes] += draw_monthly

    # Add Tax Payments
    if tax_series is not None:
        aligned_taxes = tax_series.reindex(port.index, fill_value=0.0)
        cashflows += aligned_taxes
        
    # Add Repayments (Reduce Loan)
    if repayment_series is not None:
        aligned_repayments = repayment_series.reindex(port.index, fill_value=0.0)
        cashflows -= aligned_repayments

    # DCA cash depletion is handled in the iterative loop (only while loan < 0)
    _dca_vals = None
    if dca_series is not None:
        _dca_vals = dca_series.reindex(port.index, fill_value=0.0).values

    # 2. Determine Rate Logic
    # Legacy: rate_annual is a float -> Fixed Rate
    # New: rate_annual can be a dict -> Margin Model
    
    use_vectorized = True
    rate_factor = 0.0 # Will be array or scalar
    effective_rate_series = pd.Series(0.0, index=port.index) # To store annualized % for display
    
    if isinstance(rate_annual, dict):
        model = rate_annual
        mode = model.get("type", "Fixed")
        
        if mode == "Fixed":
            r = model.get("rate_pct", 5.0)
            rate_daily = (1 + r / 100) ** (1 / 252) - 1
            rate_factor = 1 + rate_daily
            effective_rate_series[:] = r
            
        elif mode == "Variable":
            # Variable: Base + Spread
            base_series = model.get("base_series", None)
            spread = model.get("spread_pct", 1.0)
            
            if base_series is not None:
                aligned_base = base_series.reindex(port.index).ffill().fillna(0.0)
                daily_rates_pct = aligned_base + spread
                daily_rates_pct = daily_rates_pct.clip(lower=0.0) 
                
                rate_daily_series = (1 + daily_rates_pct / 100) ** (1 / 252) - 1
                rate_factor = 1 + rate_daily_series
                effective_rate_series = daily_rates_pct
            else:
                # Fallback
                rate_factor = 1 + (0.05 / 252)
                effective_rate_series[:] = 5.0

        elif mode == "Tiered":
            use_vectorized = False
            tiers = model.get("tiers", []) 
            base_series = model.get("base_series", None)
            
            if base_series is not None:
                aligned_base = base_series.reindex(port.index).ffill().fillna(0.0)
            else:
                aligned_base = pd.Series(5.0, index=port.index)
            
            # --- ITERATIVE CALCULATION FOR TIERED RATES ---
            loan_vals = np.zeros(len(port))
            eff_rate_vals = np.zeros(len(port)) # Store effective rate
            current_loan = starting_loan
            
            base_vals = aligned_base.values
            cf_vals = cashflows.values
            
            for t in range(len(port)):
                interest_accrued = 0.0
                calc_balance = max(current_loan, 0)  # No interest on cash (negative loan)

                # Calculate Blended Interest
                for i, (limit, spread) in enumerate(tiers):
                    next_limit = tiers[i+1][0] if i+1 < len(tiers) else float('inf')
                    chunk = min(max(0, calc_balance - limit), next_limit - limit)

                    if chunk > 0:
                        tier_rate_daily = (1 + (base_vals[t] + spread) / 100) ** (1 / 252) - 1
                        interest_accrued += chunk * tier_rate_daily
                    
                    if calc_balance < next_limit:
                        break
                
                # Back-calculate effective annualized rate for this day
                # Rate = (Interest / Balance) * 252 * 100
                if current_loan > 1e-9:
                     day_rate = (interest_accrued / current_loan)
                     eff_rate_vals[t] = day_rate * 252 * 100
                else:
                     # If no loan, what is the rate? Technically undefined or Base + lowest spread.
                     # Let's show Base + Tier 1 spread as 'potential' rate
                     base_s = tiers[0][1] if tiers else 0
                     eff_rate_vals[t] = base_vals[t] + base_s
                
                # Update Loan
                dca_amt = 0.0
                if _dca_vals is not None:
                    if fund_dca_margin:
                        dca_amt = _dca_vals[t]  # Always add to loan (margin-funded)
                    elif current_loan < 0:
                        dca_amt = min(_dca_vals[t], abs(current_loan))  # Only deplete remaining cash
                current_loan = current_loan + interest_accrued + cf_vals[t] + dca_amt
                loan_vals[t] = current_loan
                
            loan_series = pd.Series(loan_vals, index=port.index)
            effective_rate_series = pd.Series(eff_rate_vals, index=port.index)
            
    else:
        # Legacy Float
        r = rate_annual
        rate_daily = (1 + r / 100) ** (1 / 252) - 1
        rate_factor = 1 + rate_daily
        effective_rate_series[:] = r
    
    if use_vectorized:
        # Vectorized formula assumes constant interest rate — doesn't handle
        # zero-interest on negative balances. Use iterative fallback if starting with cash.
        if starting_loan < 0 or _dca_vals is not None or repayment_series is not None:
            # Iterative: handles zero-interest on negative balances + DCA cash depletion
            loan_vals = np.zeros(len(port))
            current_loan = starting_loan
            # rate_factor can be a scalar (Fixed/Legacy) or Series (Variable)
            if isinstance(rate_factor, float):
                _rate_daily_arr = np.full(len(port), rate_factor - 1)
            else:
                _rate_daily_arr = (rate_factor.values if hasattr(rate_factor, 'values') else np.array(rate_factor)) - 1
            cf_vals = cashflows.values
            for t in range(len(port)):
                if current_loan > 0:
                    current_loan *= (1 + _rate_daily_arr[t])
                dca_amt = 0.0
                if _dca_vals is not None:
                    if fund_dca_margin:
                        dca_amt = _dca_vals[t]  # Always add to loan (margin-funded)
                    elif current_loan < 0:
                        dca_amt = min(_dca_vals[t], abs(current_loan))  # Only deplete cash
                current_loan += cf_vals[t] + dca_amt
                loan_vals[t] = current_loan
            loan_series = pd.Series(loan_vals, index=port.index)
        else:
            cum_rate = pd.Series(rate_factor, index=port.index).cumprod()
            discounted_cashflows = cashflows / cum_rate
            cum_discounted_cashflows = discounted_cashflows.cumsum()
            loan_series = cum_rate * (starting_loan + cum_discounted_cashflows)

    loan_series.name = "Loan"
    effective_rate_series.name = "Margin Rate %"

    
    equity = port - loan_series
    safe_port = port.replace(0, np.nan)
    equity_pct = (equity / safe_port).fillna(0).rename("Equity %")
    denom = safe_port * (1 - maint_pct)
    usage_pct = (loan_series / denom).fillna(0).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct, effective_rate_series
