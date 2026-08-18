"""Sequential Phase 4 testfol.io verification for persisted candidates."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import date

import requests

ENDPOINT = "https://testfol.io/api/backtest"


def _payload(allocation: dict[str, float], start: str, end: str) -> dict[str, object]:
    percentages = {ticker: round(weight * 100, 8) for ticker, weight in allocation.items()}
    last_ticker = next(reversed(percentages))
    percentages[last_ticker] = round(100.0 - sum(value for ticker, value in percentages.items() if ticker != last_ticker), 8)
    return {
        "start_date": start,
        "end_date": end,
        "start_val": 100000,
        "adj_inflation": False,
        "cashflow": 1000,
        "cashflow_freq": "Monthly",
        "cashflow_offset": 0,
        "rolling_window": 60,
        "target_currency": "USD",
        "cashflow_legs": [],
        "one_time_cashflows": [],
        "backtests": [{
            "invest_dividends": True,
            "rebalance_freq": "Yearly",
            "rebalance_offset": 0,
            "allocation": percentages,
            "drag": 0,
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/screener.db")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy-approved", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.strategy_approved:
        raise SystemExit(
            "Live verification is paused: allocation strategy requires explicit approval. "
            "Use --strategy-approved only after the strategy regroup."
        )
    with sqlite3.connect(args.db) as connection:
        try:
            rows = connection.execute(
                "SELECT candidate_hash, allocation_json, diversification_json FROM candidates "
                "ORDER BY fitness DESC"
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error):
                print("No candidates available for verification.")
                return
            raise
    rows = [row[:2] for row in rows if not json.loads(row[2]).get("violations")][: args.limit]
    if not rows:
        print("No diversified candidates available for verification.")
        return
    for index, (candidate_hash, allocation_json) in enumerate(rows):
        payload = _payload(json.loads(allocation_json), args.start, args.end)
        if args.dry_run:
            print(json.dumps({"candidate_hash": candidate_hash, "payload": payload}))
        else:
            try:
                response = requests.post(ENDPOINT, json=payload, timeout=120)
                response.raise_for_status()
                print(json.dumps({"candidate_hash": candidate_hash, "ok": True, "response": response.json()}))
            except requests.RequestException as error:
                print(json.dumps({
                    "candidate_hash": candidate_hash,
                    "ok": False,
                    "error": type(error).__name__,
                    "message": str(error),
                }))
        if index + 1 < len(rows):
            time.sleep(1.5)


if __name__ == "__main__":
    main()
