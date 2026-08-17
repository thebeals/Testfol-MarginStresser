"""Build the locked SPY baseline used by the Streamlit screener results view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))


PERIODS = {
    "train": ("2006-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2026-08-17"),
}


def _metrics(prices):
    returns = prices.pct_change().dropna()
    wealth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)),
        "observations": int(len(returns)),
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
    }


def build(start: str, end: str) -> dict[str, object]:
    downloaded = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = downloaded["Close"].squeeze("columns").dropna()
    return {
        "benchmark": "SPY",
        "source": "yfinance auto_adjust=True",
        "date_range": {"start": start, "end": end},
        "periods": {
            name: _metrics(prices.loc[period_start:period_end])
            for name, (period_start, period_end) in PERIODS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--output", default="data/screener-benchmark.json")
    args = parser.parse_args()
    report = build(args.start, args.end)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
