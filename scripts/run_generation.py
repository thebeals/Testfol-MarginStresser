"""Run a bounded local screener generation for operator verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.loop import run_generation
from screener.hybrid_data import extend_hybrid_prices
from app.services.data_service import get_fed_funds_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--db", default="data/screener.db")
    parser.add_argument("--log", default="data/generation.jsonl")
    parser.add_argument("--subset-limit", type=int, default=2)
    parser.add_argument("--search-trials", type=int, default=10)
    args = parser.parse_args()

    tickers = ("SPY", "TLT", "TIP", "SGOV", "GLD", "DBC", "DBMF", "VNQ")
    prices = yf.download(
        list(tickers),
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = prices["Close"].reindex(columns=tickers)
    prices, provenance = extend_hybrid_prices(prices, get_fed_funds_rate())
    prices = prices.reindex(columns=tickers).dropna(how="any")
    returns = prices.pct_change().dropna()
    results = run_generation(
        returns,
        db_path=args.db,
        log_path=args.log,
        subset_limit=args.subset_limit,
        search_trials=args.search_trials,
        workers=2,
    )
    print(
        f"generated {len(results)} candidates from {len(returns)} daily observations "
        f"(Testfol synthetic: {provenance['testfol_simulated_tickers']})"
    )


if __name__ == "__main__":
    main()
