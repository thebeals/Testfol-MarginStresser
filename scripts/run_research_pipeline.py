"""Build a deduplicated allocation pool, walk-forward rank it, and export finalists."""

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
from screener.diversify import check_diversification
from screener.hybrid_data import extend_hybrid_prices
from screener.rebalance import SCREENER_REBALANCE_FREQUENCIES, rebalance_returns
from screener.search import discover_feasible_allocations


def _metrics(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"cagr": float("nan"), "max_drawdown": float("nan"), "sharpe": float("nan")}
    wealth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    drawdown = wealth / wealth.cummax() - 1.0
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0
    return {"cagr": cagr, "max_drawdown": float(drawdown.min()), "sharpe": sharpe}


def _period_metrics(returns: pd.DataFrame, allocation: dict[str, float], frequency: str) -> dict[str, object]:
    portfolio = rebalance_returns(returns, allocation, frequency)
    periods = {
        "train": ("2006-01-01", "2016-12-31"),
        "validation": ("2017-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2026-08-17"),
    }
    return {name: _metrics(portfolio.loc[start:end]) for name, (start, end) in periods.items()}


def build_pool(prices: pd.DataFrame, *, target: int, samples_per_size: int, seed: int) -> list[dict[str, object]]:
    returns = prices.pct_change().dropna()
    allocations = discover_feasible_allocations(
        returns,
        target=target,
        samples_per_size=samples_per_size,
        seed=seed,
    )
    pool: list[dict[str, object]] = []
    for allocation in allocations:
        diversification = check_diversification(returns, allocation)
        for frequency in SCREENER_REBALANCE_FREQUENCIES:
            periods = _period_metrics(returns, allocation, frequency)
            validation = periods["validation"]
            test = periods["test"]
            score = (
                0.5 * validation["cagr"]
                + 0.5 * test["cagr"]
                - 0.2 * abs(test["max_drawdown"])
            )
            local_gate_passed = (
                diversification.passed
                and validation["cagr"] > 0.0
                and test["cagr"] > 0.0
                and test["sharpe"] >= 0.4
                and test["max_drawdown"] >= -0.50
            )
            pool.append(
                {
                    "allocation": allocation,
                    "rebalance_freq": frequency,
                    "walk_forward": periods,
                    "walk_forward_score": score,
                    "diversification": {
                        "factor_breadth": diversification.factor_breadth,
                        "factor_variance_share": diversification.factor_variance_share,
                        "stress_correlations": diversification.stress_correlations,
                        "violations": list(diversification.violations),
                    },
                    "diversification_passed": diversification.passed,
                    "local_gate_passed": local_gate_passed,
                    "fitness": score,
                }
            )
    return sorted(pool, key=lambda row: float(row["walk_forward_score"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--samples-per-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--universe", default="data/allocation-feasibility.json")
    parser.add_argument("--benchmark", default="data/screener-benchmark.json")
    parser.add_argument("--output", default="data/allocation-pool.jsonl")
    parser.add_argument("--shortlist", default="data/research-shortlist.jsonl")
    args = parser.parse_args()

    universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    spy_test_cagr = float(benchmark["periods"]["test"]["cagr"])
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(
        list(tickers),
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )["Close"].reindex(columns=tickers).dropna(axis=1, how="all")
    prices, _ = extend_hybrid_prices(prices, get_fed_funds_rate())
    prices = prices.reindex(columns=[ticker for ticker in tickers if ticker in prices]).dropna(how="any")
    pool = build_pool(prices, target=args.target, samples_per_size=args.samples_per_size, seed=args.seed)
    for row in pool:
        row["local_gate_passed"] = bool(
            row["local_gate_passed"]
            and row["walk_forward"]["test"]["cagr"] > spy_test_cagr
        )
        row["spy_test_cagr"] = spy_test_cagr
    Path(args.output).write_text("\n".join(json.dumps(row, sort_keys=True) for row in pool) + "\n", encoding="utf-8")

    seen: set[tuple[tuple[str, float], ...]] = set()
    shortlist: list[dict[str, object]] = []
    for row in pool:
        if not row["local_gate_passed"]:
            continue
        key = tuple(sorted((ticker, round(float(weight), 8)) for ticker, weight in row["allocation"].items()))
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(row)
        if len(shortlist) >= 20:
            break
    Path(args.shortlist).write_text("\n".join(json.dumps(row, sort_keys=True) for row in shortlist) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_tickers": len(tickers), "observations": len(prices), "local_gate_shortlist": len(shortlist), "pool_records": len(pool)}, indent=2))


if __name__ == "__main__":
    main()
