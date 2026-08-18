"""Compare cached Testfol synthetic managed-futures histories with observed ETFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.hybrid_data import load_testfol_simulated_prices


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def compare(start: str, end: str, cache_path: str | Path) -> dict[str, object]:
    observed = yf.download(
        ["DBMF", "KMLM"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )["Close"]
    simulated = load_testfol_simulated_prices(cache_path)
    reports: dict[str, object] = {}
    for ticker in ("DBMF", "KMLM"):
        actual_returns = observed[ticker].dropna().pct_change().dropna().rename("observed")
        testfol_returns = simulated[ticker].pct_change().dropna().rename("testfol")
        aligned = pd.concat([actual_returns, testfol_returns], axis=1).dropna()
        if aligned.empty:
            reports[ticker] = {"error": "no overlapping observations"}
            continue
        difference = aligned["observed"] - aligned["testfol"]
        reports[ticker] = {
            "overlap_start": str(aligned.index.min().date()),
            "overlap_end": str(aligned.index.max().date()),
            "observations": len(aligned),
            "daily_correlation": float(aligned.corr().iloc[0, 1]),
            "annualized_tracking_error": float(difference.std(ddof=1) * np.sqrt(252)),
            "observed_cagr": float((1.0 + aligned["observed"]).prod() ** (252.0 / len(aligned)) - 1.0),
            "testfol_cagr": float((1.0 + aligned["testfol"]).prod() ** (252.0 / len(aligned)) - 1.0),
            "observed_max_drawdown": _max_drawdown(aligned["observed"]),
            "testfol_max_drawdown": _max_drawdown(aligned["testfol"]),
        }
    return {"date_range": {"start": start, "end": end}, "tickers": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--cache", default="data/testfol-simulated-returns.json")
    parser.add_argument("--output", default="data/testfol-observed-comparison.json")
    args = parser.parse_args()
    report = compare(args.start, args.end, args.cache)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
