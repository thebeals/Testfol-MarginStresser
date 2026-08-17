"""Scan whether the allocation rules produce feasible portfolios."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.diversify import check_diversification
from screener.diversify import STRESS_WINDOWS
from screener.universe import all_tickers


def _download_prices(start: str, end: str):
    prices = yf.download(
        all_tickers(),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if hasattr(prices.columns, "levels"):
        prices = prices["Close"]
    else:
        prices = prices.rename(columns={"Close": all_tickers()[0]})
    prices = prices.dropna(axis=1, how="all")
    return prices, [str(column) for column in prices.columns]


def _stress_coverage(prices) -> dict[str, int]:
    return {
        name: int(prices.loc[start:end].dropna(how="any").shape[0])
        for name, (start, end) in STRESS_WINDOWS.items()
    }


def scan(prices, *, samples_per_size: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    coverage = {
        ticker: _stress_coverage(prices[[ticker]]) for ticker in prices.columns
    }
    eligible = tuple(
        ticker for ticker, windows in coverage.items() if all(count >= 2 for count in windows.values())
    )
    tickers = tuple(str(column) for column in eligible)
    by_size: dict[str, dict[str, int]] = {}
    examples: list[dict[str, object]] = []
    for size in range(3, min(8, len(tickers)) + 1):
        feasible = 0
        for _ in range(samples_per_size):
            selected = tuple(rng.choice(tickers, size=size, replace=False).tolist())
            values = rng.dirichlet(np.ones(size))
            allocation = dict(zip(selected, values))
            candidate_returns = prices.loc[:, list(selected)].dropna(how="any").pct_change().dropna()
            report = check_diversification(candidate_returns, allocation)
            if report.passed:
                feasible += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "tickers": selected,
                            "weights": {ticker: round(float(weight), 6) for ticker, weight in allocation.items()},
                            "factor_breadth": report.factor_breadth,
                            "factor_variance_share": report.factor_variance_share,
                            "stress_correlations": report.stress_correlations,
                        }
                    )
        by_size[str(size)] = {
            "sampled": samples_per_size,
            "feasible": feasible,
        }
    return {
        "seed": seed,
        "samples_per_size": samples_per_size,
        "ticker_count": len(tickers),
        "eligible_tickers": tickers,
        "by_size": by_size,
        "examples": examples,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--samples-per-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/allocation-feasibility.json")
    args = parser.parse_args()

    prices, available = _download_prices(args.start, args.end)
    result = scan(prices, samples_per_size=args.samples_per_size, seed=args.seed)
    result["date_range"] = {"start": args.start, "end": args.end}
    result["available_tickers"] = available
    result["price_start"] = str(prices.index.min().date()) if not prices.empty else None
    result["price_end"] = str(prices.index.max().date()) if not prices.empty else None
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("date_range", "price_start", "price_end", "ticker_count", "eligible_tickers", "by_size")}, indent=2))


if __name__ == "__main__":
    main()
