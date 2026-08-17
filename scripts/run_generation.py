"""Run a bounded local screener generation for operator verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.loop import run_generation
from screener.hybrid_data import extend_hybrid_prices
from screener.universe import all_tickers
from app.services.data_service import get_fed_funds_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--db", default="data/screener.db")
    parser.add_argument("--log", default="data/generation.jsonl")
    parser.add_argument("--subset-limit", type=int, default=2)
    parser.add_argument("--search-trials", type=int, default=10)
    parser.add_argument("--tickers", help="Comma-separated active ticker list; defaults to the full universe")
    parser.add_argument("--feasibility-cache", default="data/allocation-feasibility.json")
    args = parser.parse_args()

    tickers = tuple(
        ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()
    ) if args.tickers else tuple(all_tickers())
    prices = yf.download(
        list(tickers),
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = prices["Close"].reindex(columns=tickers).dropna(axis=1, how="all")
    prices, provenance = extend_hybrid_prices(prices, get_fed_funds_rate())
    available_tickers = tuple(str(ticker) for ticker in prices.columns)
    prices = prices.reindex(columns=available_tickers).dropna(how="any")
    returns = prices.pct_change().dropna()
    feasibility = json.loads(Path(args.feasibility_cache).read_text(encoding="utf-8"))
    seed_allocations = [
        {str(ticker): float(weight) for ticker, weight in example["weights"].items()}
        for example in feasibility.get("examples", [])
    ]
    results = run_generation(
        returns,
        db_path=args.db,
        log_path=args.log,
        subset_limit=args.subset_limit,
        search_trials=args.search_trials,
        workers=2,
        seed_allocations=seed_allocations,
    )
    print(
        f"generated {len(results)} candidates from {len(returns)} daily observations "
        f"(Testfol synthetic: {provenance['testfol_simulated_tickers']})"
    )


if __name__ == "__main__":
    main()
