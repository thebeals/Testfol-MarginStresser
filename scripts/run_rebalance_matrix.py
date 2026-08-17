"""Evaluate matched calendar, band, and EMA rebalance policies locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from screener.rebalance import rebalance_returns


PERIOD = ("2021-01-01", "2026-08-17")
CALENDAR_METHODS = ["None", "Monthly", "Quarterly", "Yearly"]
CHECKS = ["Monthly", "Quarterly", "Yearly"]


def _metrics(returns: pd.Series) -> dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0,
    }


def _period_key(date: pd.Timestamp, frequency: str) -> tuple[int, ...]:
    if frequency == "Monthly":
        return date.year, date.month
    if frequency == "Quarterly":
        return date.year, date.quarter
    return (date.year,)


def _band_returns(prices: pd.DataFrame, allocation: dict[str, float], check: str, mode: str, band: float) -> pd.Series:
    values = {ticker: 100.0 * weight for ticker, weight in allocation.items()}
    returns = prices.loc[:, list(allocation)].pct_change().fillna(0.0)
    output = []
    prior_period = None
    for date, daily in returns.iterrows():
        before = sum(values.values())
        for ticker in values:
            values[ticker] *= 1.0 + float(daily[ticker])
        period = _period_key(date, check)
        if prior_period is not None and period != prior_period:
            total = sum(values.values())
            weights = {ticker: value / total for ticker, value in values.items()}
            breached = any(
                abs(weights[ticker] - target) > (band * target if mode == "relative" else band)
                for ticker, target in allocation.items()
            )
            if breached:
                values = {ticker: total * target for ticker, target in allocation.items()}
        output.append(sum(values.values()) / before - 1.0)
        prior_period = period
    return pd.Series(output, index=returns.index)


def _ema_returns(prices: pd.DataFrame, allocation: dict[str, float], check: str, window: int, policy: str) -> pd.Series:
    tickers = list(allocation)
    signal_prices = prices.loc[:, tickers]
    ema = signal_prices.ewm(span=window, adjust=False, min_periods=window).mean()
    returns = signal_prices.pct_change().fillna(0.0)
    values = {ticker: 100.0 * weight for ticker, weight in allocation.items()}
    cash = 10.0 if policy == "cash_only" else 0.0
    if policy == "cash_only":
        values = {ticker: value * 0.9 for ticker, value in values.items()}
    output = []
    prior_period = None
    prior_signals = {ticker: True for ticker in tickers}
    for date, daily in returns.iterrows():
        before = sum(values.values()) + cash
        for ticker in values:
            values[ticker] *= 1.0 + float(daily[ticker])
        if "SHV" in prices.columns:
            cash *= 1.0 + float(prices["SHV"].pct_change().fillna(0.0).get(date, 0.0))
        period = _period_key(date, check)
        signal_date = date - pd.offsets.BDay(1)
        if prior_period is not None and period != prior_period and signal_date in ema.index:
            signals = {ticker: bool(signal_prices.loc[signal_date, ticker] >= ema.loc[signal_date, ticker]) for ticker in tickers}
            changed = [ticker for ticker in tickers if signals[ticker] != prior_signals[ticker]]
            if changed:
                for ticker in tickers:
                    if not signals[ticker]:
                        cash += values[ticker]
                        values[ticker] = 0.0
                active = [ticker for ticker in tickers if signals[ticker]]
                if active:
                    active_weights = {ticker: allocation[ticker] for ticker in active}
                    weight_total = sum(active_weights.values())
                    active_weights = {ticker: weight / weight_total for ticker, weight in active_weights.items()}
                    investable = sum(values.values()) + cash
                    desired = {ticker: investable * active_weights[ticker] for ticker in active}
                    if policy == "cash_only":
                        for ticker in active:
                            purchase = min(max(desired[ticker] - values[ticker], 0.0), cash)
                            values[ticker] += purchase
                            cash -= purchase
                    elif policy == "replacement":
                        for ticker in sorted(active, key=lambda item: values[item]):
                            needed = max(desired[ticker] - values[ticker], 0.0)
                            source = min(needed, values[ticker])
                            values[ticker] -= source
                            cash += source
                        for ticker in active:
                            purchase = min(max(desired[ticker] - values[ticker], 0.0), cash)
                            values[ticker] += purchase
                            cash -= purchase
                    else:
                        values = {ticker: desired.get(ticker, 0.0) for ticker in tickers}
                        cash = max(investable - sum(values.values()), 0.0)
            prior_signals = signals
        output.append((sum(values.values()) + cash) / before - 1.0)
        prior_period = period
    return pd.Series(output, index=returns.index)


def run(prices: pd.DataFrame, allocations: list[dict[str, float]]) -> list[dict[str, object]]:
    test_prices = prices.loc[PERIOD[0] : PERIOD[1]]
    records: list[dict[str, object]] = []
    for index, allocation in enumerate(allocations, start=1):
        returns = test_prices.loc[:, list(allocation)].pct_change().dropna()
        methods: list[tuple[str, pd.Series]] = []
        for frequency in CALENDAR_METHODS:
            methods.append((f"Calendar:{frequency}", rebalance_returns(returns, allocation, frequency)))
        for mode in ("relative", "absolute"):
            for band in (0.05, 0.10, 0.20):
                for check in CHECKS:
                    methods.append((f"{mode.title()}Band:{check}:{band:g}", _band_returns(test_prices, allocation, check, mode, band)))
        for window in (100, 200):
            for policy in ("cash_only", "pro_rata", "replacement"):
                for check in CHECKS:
                    methods.append((f"EMA{window}:{policy}:{check}", _ema_returns(test_prices, allocation, check, window, policy)))
        for method, portfolio_returns in methods:
            metrics = _metrics(portfolio_returns.dropna())
            records.append({"allocation_id": index, "allocation": allocation, "method": method, **metrics})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/screener-results.json")
    parser.add_argument("--universe", default="data/allocation-feasibility.json")
    parser.add_argument("--output", default="data/rebalance-matrix-results.jsonl")
    args = parser.parse_args()
    accepted = json.loads(Path(args.input).read_text(encoding="utf-8"))["accepted"]
    allocations = [row["local"]["allocation"] for row in accepted]
    universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-17", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for allocation in allocations for ticker in allocation})
    results = run(prices.dropna(subset=required), allocations)
    Path(args.output).write_text("\n".join(json.dumps(row, sort_keys=True) for row in results) + "\n", encoding="utf-8")
    best = {}
    for row in results:
        best.setdefault(row["allocation_id"], row)
        if row["cagr"] > best[row["allocation_id"]]["cagr"]:
            best[row["allocation_id"]] = row
    print(json.dumps({"allocations": len(allocations), "records": len(results), "best": list(best.values())}, indent=2))


if __name__ == "__main__":
    main()
