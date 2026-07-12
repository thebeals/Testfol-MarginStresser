from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_cagr(series: pd.Series) -> float:
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or start_val <= 0: return 0.0
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr

def calculate_max_drawdown(series: pd.Series) -> float:
    if series.empty: return 0.0
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min() * 100
    return max_dd

def calculate_sharpe_ratio(series: pd.Series, risk_free_rate: float = 0.0) -> float:
    if series.empty: return 0.0
    # Calculate daily returns
    returns = series.pct_change().dropna()
    if returns.empty: return 0.0

    # Calculate excess returns
    # risk_free_rate is annual, convert to daily
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily

    # Calculate Sharpe
    std = excess_returns.std()
    if std == 0: return 0.0

    sharpe = (excess_returns.mean() / std) * (252 ** 0.5)
    return sharpe

def calculate_tax_adjusted_equity(
    base_equity_series: pd.Series,
    tax_payment_series: pd.Series,
    port_series: pd.Series,
    loan_series: pd.Series,
    rate_annual: float,
    draw_monthly: float = 0.0,
    draw_start_date=None,
    draw_monthly_retirement: float = 0.0,
    retirement_date=None,
) -> tuple[pd.Series, pd.Series]:
    """
    Simulates the equity curve if taxes AND draws were paid from capital (reducing the base).
    Accounts for lost compounding and scales future taxes down proportionally.
    Correctly models leverage dynamics: Assets shrink, but Loan (and Interest) remains constant (or zero).

    Returns:
        (adjusted_equity_series, adjusted_tax_series)
    """
    # 1. Calculate Asset Returns
    asset_returns = port_series.pct_change().fillna(0)

    # 2. USD margin interest uses simple Actual/360 over elapsed calendar days.
    from app.core.margin_interest import usd_margin_period_rates

    daily_rate = usd_margin_period_rates(base_equity_series.index, rate_annual)
    daily_rate.index = base_equity_series.index

    # 3. Create "External Flow" Series (B_t)
    # B_t = Loan_{t-1} * (r_asset - r_loan) - Draws - Taxes

    loan_component = loan_series * (asset_returns - daily_rate)

    draws = pd.Series(0.0, index=base_equity_series.index)
    if draw_monthly > 0 or draw_monthly_retirement > 0:
        months = base_equity_series.index.month
        month_changes = months != pd.Series(months, index=base_equity_series.index).shift(1)
        month_changes[0] = False
        if draw_start_date is not None:
            after_start = base_equity_series.index >= pd.Timestamp(draw_start_date)
            month_changes = month_changes & after_start
        if draw_monthly_retirement > 0 and retirement_date is not None:
            after_ret = base_equity_series.index >= pd.Timestamp(retirement_date)
            pre_ret = month_changes & ~after_ret
            post_ret = month_changes & after_ret
            draws[pre_ret] = draw_monthly
            draws[post_ret] = draw_monthly_retirement
        else:
            draws[month_changes] = draw_monthly

    taxes = tax_payment_series.reindex(base_equity_series.index, fill_value=0.0)

    external_flows = loan_component - draws - taxes

    # Solve Recurrence: E_t = E_{t-1} * A_t + B_t
    growth_factors = 1 + asset_returns

    external_flows.iloc[0] = 0
    growth_factors.iloc[0] = 1.0
    cum_growth = growth_factors.cumprod()

    discounted_flows = external_flows / cum_growth
    cum_discounted_flows = discounted_flows.cumsum()

    e_start = base_equity_series.iloc[0]

    adj_equity_series = cum_growth * (e_start + cum_discounted_flows)

    # Reconstruct Tax Series (just the input taxes)
    adj_tax_series = taxes

    return adj_equity_series, adj_tax_series

def process_rebalancing_data(
    rebal_events: list,
    port_series: pd.Series,
    allocation: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Process rebalancing events to calculate trade amounts and realized P&L.
    """
    if not rebal_events:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    trades = []
    composition = []

    # Initialize cost basis with initial allocation
    initial_val = port_series.iloc[0]
    cost_basis = {}



    # Normalize allocation to 100% just in case
    total_alloc = sum(allocation.values())
    if total_alloc > 0:
        for ticker, weight in allocation.items():
            cost_basis[ticker] = initial_val * (weight / total_alloc)

    # Process each event group
    for group in rebal_events:
        tickers = group.get("tickers", [])
        events = group.get("events", [])

        for event in events:
            date_str = event[0]
            date = pd.to_datetime(date_str)

            # Find portfolio value at rebalance date
            try:
                port_val = port_series.asof(date)
            except (KeyError, IndexError, TypeError):
                continue

            if pd.isna(port_val):
                continue

            n_tickers = len(tickers)

            for i, ticker in enumerate(tickers):
                drift_idx = 1 + i
                weight_idx = 1 + n_tickers + i
                trade_idx = 1 + 2*n_tickers + i

                if trade_idx >= len(event):
                    break

                # drift_pct = event[drift_idx]
                trade_pct = event[trade_idx] # This is the actual trade % executed

                target_weight = allocation.get(ticker, 0)

                trade_amt = port_val * (trade_pct / 100)

                # Current Value (Post-Rebalance)
                curr_val = port_val * (target_weight / total_alloc)

                # Record composition snapshot
                composition.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Value": curr_val
                })

                # Update Cost Basis and Calculate P&L
                realized_pl = 0

                if ticker not in cost_basis:
                    cost_basis[ticker] = 0

                if trade_amt > 0: # BUY
                    cost_basis[ticker] += trade_amt
                elif trade_amt < 0: # SELL
                    sell_amt = -trade_amt
                    if curr_val > 0:
                        fraction_sold = sell_amt / (curr_val + sell_amt)
                        # Cap fraction at 1.0
                        fraction_sold = min(fraction_sold, 1.0)

                        basis_reduction = cost_basis[ticker] * fraction_sold
                        realized_pl = sell_amt - basis_reduction
                        cost_basis[ticker] -= basis_reduction

                if trade_amt != 0:
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized P&L": realized_pl,
                        "Year": date.year
                    })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["Date"] = pd.to_datetime(trades_df["Date"])
        trades_df = trades_df.sort_values("Date")

    composition_df = pd.DataFrame(composition)
    if not composition_df.empty:
        composition_df["Date"] = pd.to_datetime(composition_df["Date"])

    # Aggregate P&L by Year
    if not trades_df.empty:
        pl_by_year = trades_df.groupby("Year")["Realized P&L"].sum()
    else:
        pl_by_year = pd.Series(dtype=float)

    return trades_df, pl_by_year, composition_df

def generate_stats(series: pd.Series) -> dict:
    """
    Calculates portfolio statistics from a daily value series.
    Returns a dict matching the structure of Testfol API 'stats'.
    """
    if series.empty:
        return {}

    cagr = calculate_cagr(series)
    mdd = calculate_max_drawdown(series)
    sharpe = calculate_sharpe_ratio(series)

    # Calculate Standard Deviation (Annualized)
    returns = series.pct_change().dropna()
    std = returns.std() * (252 ** 0.5) * 100

    best_year = 0.0
    worst_year = 0.0

    # Yearly Returns
    if not series.empty:
        yearly = series.resample('YE').last().pct_change()
        if not yearly.empty:
             yearly = yearly.iloc[1:]  # Drop partial first year (NaN from pct_change)
             # Drop partial last year only if the series doesn't end near Dec 31
             if len(yearly) > 1 and not series.empty:
                 last_day = series.index[-1]
                 if last_day.month < 12 or last_day.day < 25:
                     yearly = yearly.iloc[:-1]
             if not yearly.empty:
                 best_year = yearly.max() * 100
                 worst_year = yearly.min() * 100

    # Sortino Ratio
    sortino = 0.0
    if not returns.empty:
        neg = np.minimum(returns.values, 0)
        downside_dev = np.sqrt(np.mean(neg ** 2))
        if downside_dev > 0:
            sortino = (returns.mean() / downside_dev) * (252 ** 0.5)

    # Ulcer Index: sqrt(mean(drawdown_pct^2))
    ulcer_index = 0.0
    cummax = series.cummax()
    dd_pct = (series - cummax) / cummax * 100
    ulcer_index = float(np.sqrt(np.mean(dd_pct ** 2)))

    # Calmar Ratio: CAGR / |Max DD|
    calmar = abs(cagr / mdd) if mdd != 0 else 0.0

    # Average Drawdown
    below = dd_pct[dd_pct < 0]
    avg_drawdown = float(below.mean()) if not below.empty else 0.0

    return {
        "cagr": cagr,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "best_year": best_year,
        "worst_year": worst_year,
        "ulcer_index": ulcer_index,
        "sortino": sortino,
        "calmar": calmar,
        "avg_drawdown": avg_drawdown,
    }


def find_drawdown_episodes(series: pd.Series, threshold: float = -0.05) -> list[dict]:
    """Find all drawdown episodes exceeding threshold.

    Returns list of dicts with keys: peak_date, peak_val, trough_date,
    trough_val, dd (decimal, e.g. -0.25), recovery (date or None).
    """
    if series.empty:
        return []
    dd = series / series.cummax() - 1.0
    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough_dd = 0.0
    for date, d in dd.items():
        if d == 0.0:
            if in_dd and trough_dd < threshold:
                episodes.append({
                    "peak_date": peak_date, "peak_val": series[peak_date],
                    "trough_date": trough_date, "trough_val": series[trough_date],
                    "dd": trough_dd, "recovery": date,
                })
            in_dd = False
            peak_date = date
            trough_dd = 0.0
        else:
            in_dd = True
            if d < trough_dd:
                trough_dd = d
                trough_date = date
    if in_dd and trough_dd < threshold:
        episodes.append({
            "peak_date": peak_date, "peak_val": series[peak_date],
            "trough_date": trough_date, "trough_val": series[trough_date],
            "dd": trough_dd, "recovery": None,
        })
    return episodes


def fmt_duration(days: int) -> str:
    """Format a day count as human-readable duration (e.g. '5.0yr', '3mo', '59d')."""
    if days >= 365:
        return f"{days / 365.25:.1f}yr"
    if days >= 60:
        return f"{days // 30}mo"
    return f"{days}d"


MARKET_EVENT_MAP: dict[tuple[int, int], str] = {
    (2000, 1): "Y2K Fears, Dot-com Excess",
    (2000, 3): "Dot-com Bubble Burst, Tech Wreck",
    (2000, 7): "Dot-com Continued, Tech Wreck",
    (2001, 1): "Dot-com Unwind, Recession Begins",
    (2001, 9): "9/11 Attacks, Market Shutdown",
    (2002, 3): "WorldCom/Enron Scandals, Accounting Crisis",
    (2002, 7): "Corporate Fraud, Market Capitulation",
    (2003, 1): "Iraq War Buildup, Terror Fears",
    (2003, 3): "Iraq Invasion, Oil Spike, SARS",
    (2004, 3): "Madrid Bombings, Rate Hike Fears",
    (2004, 7): "Oil >$50, Rate Hikes Begin",
    (2005, 3): "GM/Ford Downgrades, Credit Fears",
    (2005, 10): "Katrina Aftermath, Oil Spike, Inflation",
    (2006, 5): "EM Selloff, Rate Hike Fears",
    (2006, 7): "Lebanon War, Oil $78, Housing Cools",
    (2007, 2): "Subprime Early Signs, Shanghai Crash",
    (2007, 7): "Subprime Contagion, Quant Blowups",
    (2007, 11): "GFC: Lehman/AIG/Bear Stearns Collapse",
    (2010, 11): "Ireland Bailout, EU Debt Contagion",
    (2011, 2): "Arab Spring, Japan Earthquake/Fukushima",
    (2011, 4): "EU Debt (Portugal), S&P Warning",
    (2011, 7): "US Downgrade (AAA->AA+), EU Crisis",
    (2011, 11): "EU Debt Crisis, Italy/Greece Contagion, MF Global",
    (2012, 3): "EU Debt (Spain/Italy), Austerity",
    (2012, 9): "Fiscal Cliff Fears, Election",
    (2013, 5): "Taper Tantrum, Fed Signals QE Wind-Down",
    (2014, 1): "EM Currency Crisis (Turkey/Argentina)",
    (2014, 3): "Russia Annexes Crimea, Sanctions",
    (2014, 7): "Ukraine/MH17, Gaza, ISIS, Ebola",
    (2014, 9): "Ebola Fears, Oil Price Collapse, ISIS",
    (2014, 11): "Oil Crash, OPEC Refuses Cut, Ruble",
    (2015, 3): "Dollar Surge, Rate Hike Fears",
    (2015, 4): "China Slowdown, Dollar Drag",
    (2015, 7): "China Devaluation, Yuan Shock, EM Crash",
    (2015, 11): "Paris Attacks, Rate Hike, Oil <$40",
    (2015, 12): "China, Oil <$30, Recession Fears",
    (2016, 6): "Brexit Vote, EU Uncertainty",
    (2016, 10): "Election Uncertainty, Trump Shock",
    (2017, 6): "Tech/FANG Rotation, Valuation Fears",
    (2018, 1): "Volmageddon (XIV Blowup), Rate Fears",
    (2018, 2): "Inflation/Rate Fears, VIX Aftershock",
    (2018, 3): "Trade War (Tariffs), Facebook Scandal",
    (2018, 6): "Tariff Escalation, Turkey Lira Crisis",
    (2018, 7): "Trade War Intensifies, EM Stress",
    (2018, 8): "Trade War + Fed Hawkish + Housing",
    (2019, 4): "Trade War, Tariff Tweets, China",
    (2019, 7): "Yield Curve Inversion, Recession Signal",
    (2020, 2): "COVID Pandemic, Lockdowns, Depression Fears",
    (2020, 6): "COVID Second Wave, Reopening Doubts",
    (2020, 7): "COVID Resurgence, Tech Bubble Talk",
    (2020, 8): "Softbank Whale, Value Rotation",
    (2020, 9): "No Stimulus, COVID, Election Uncertainty",
    (2021, 1): "GameStop/Meme Frenzy, Rate Scare",
    (2021, 2): "Rate Spike (10Y>1.5%), Value Rotation",
    (2021, 4): "Inflation Spike (CPI 5%), Tax Plan",
    (2021, 7): "Delta Variant, China Tech Crackdown",
    (2021, 9): "Evergrande, Fed Taper, Supply Chain",
    (2021, 11): "Omicron, Fed Hawkish, Inflation 6.8%",
    (2023, 7): "Fitch Downgrade, 10Y>4.5%, Higher Longer",
    (2023, 12): "Rate Cuts Repriced, Strong Economy",
    (2024, 1): "Rate Cut Delay, Strong Jobs",
    (2024, 4): "Hot CPI, Rate Cuts Repriced, Iran/Israel",
    (2024, 7): "Yen Carry Unwind, Japan Hike, Nikkei Crash",
    (2024, 11): "Post-Election Tariff Fears, Strong $",
    (2024, 12): "DeepSeek AI Shock, Mag7 Selloff, Tariffs",
    (2025, 8): "Recession Fears, AI Capex Pullback",
    (2025, 10): "Tariff Impact, Earnings Downgrades",
    (2026, 1): "Iran War, Tariff Recession, AI Bubble Fears",
}


def get_market_event(peak_date) -> str:
    """Look up a market event by peak date, with +/- 2 month fuzzy matching."""
    key = (peak_date.year, peak_date.month)
    if key in MARKET_EVENT_MAP:
        return MARKET_EVENT_MAP[key]
    for offset in [1, -1, 2, -2]:
        m = peak_date.month + offset
        y = peak_date.year
        if m > 12: m -= 12; y += 1
        if m < 1: m += 12; y -= 1
        if (y, m) in MARKET_EVENT_MAP:
            return MARKET_EVENT_MAP[(y, m)]
    return ""


def _classify_severity(decline_pct: float) -> str:
    """Classify drawdown severity by percentage (already negative)."""
    d = abs(decline_pct)
    if d >= 25: return "Severe"
    if d >= 15: return "Moderate"
    if d >= 10: return "Mild"
    return "Minor"


def build_drawdown_table(port_series: pd.Series, spy_series: pd.Series) -> pd.DataFrame:
    """Build a DataFrame of all drawdown episodes with SPY comparison.

    Args:
        port_series: Portfolio value series (DatetimeIndex).
        spy_series: SPY/SPYSIM value series, aligned to same dates as port_series.

    Returns:
        DataFrame with display columns and hidden _metadata columns for filtering.
    """
    columns = [
        "Correction Period", "Days", "% Decline",
        "Recovery from Bottom", "Decline + Recovery Time",
        "SPY DD", "SPY Recovery from Bottom", "SPY Decline + Recovery Time",
        "Ratio", "Market Event",
        "_ongoing", "_severity", "_peak_date",
    ]

    episodes = find_drawdown_episodes(port_series, threshold=-0.05)
    if not episodes:
        return pd.DataFrame(columns=columns)

    spy_norm = spy_series / spy_series.iloc[0] * port_series.iloc[0]
    last_date = port_series.index[-1]
    rows = []

    for ep in episodes:
        peak, trough, recovery = ep["peak_date"], ep["trough_date"], ep["recovery"]
        pdd = ep["dd"] * 100
        n_days = (trough - peak).days

        # SPY drawdown during same window
        end = recovery if recovery else last_date
        sw = spy_norm.loc[peak:end]
        if sw.empty:
            sdd = 0.0
            spy_trough = peak
        else:
            sddw = sw / sw.cummax() - 1.0
            sdd = sddw.min() * 100
            spy_trough = sddw.idxmin()
        ratio = abs(pdd / sdd) if sdd != 0 else 0.0

        # Portfolio recovery (store raw days for sortability)
        if recovery:
            split_recov_days = (recovery - trough).days
            split_total_days = (recovery - peak).days
        else:
            split_recov_days = -(last_date - trough).days   # negative = ongoing
            split_total_days = -(last_date - peak).days     # negative = ongoing

        # SPY recovery (store raw days for sortability)
        if not sw.empty:
            spy_peak_val = spy_norm.loc[peak]
            spy_after = spy_norm.loc[spy_trough:]
            spy_recovered = spy_after[spy_after >= spy_peak_val]
            if len(spy_recovered) > 0:
                spy_rd = spy_recovered.index[0]
                spy_recov_days = (spy_rd - spy_trough).days
                spy_total_days = (spy_rd - peak).days
            else:
                spy_recov_days = -(last_date - spy_trough).days
                spy_total_days = -(last_date - peak).days
        else:
            spy_recov_days = None
            spy_total_days = None

        period = f"{peak.strftime('%b %d, %Y')} - {trough.strftime('%b %d, %Y')}"
        if not recovery:
            period += "*"

        rows.append({
            "Correction Period": period,
            "Days": n_days,
            "% Decline": pdd,
            "Recovery from Bottom": split_recov_days,
            "Decline + Recovery Time": split_total_days,
            "SPY DD": sdd,
            "SPY Recovery from Bottom": spy_recov_days,
            "SPY Decline + Recovery Time": spy_total_days,
            "Ratio": ratio,
            "Market Event": get_market_event(peak),
            "_ongoing": bool(recovery is None),
            "_severity": _classify_severity(pdd),
            "_peak_date": peak,
        })

    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values("_peak_date").reset_index(drop=True)
    # Ensure _ongoing holds native Python bools (pandas coerces to np.bool_ by default)
    df["_ongoing"] = df["_ongoing"].astype(object)
    return df
