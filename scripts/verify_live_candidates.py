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
            "allocation": {ticker: round(weight * 100, 8) for ticker, weight in allocation.items()},
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
    args = parser.parse_args()
    with sqlite3.connect(args.db) as connection:
        try:
            rows = connection.execute(
                "SELECT candidate_hash, allocation_json FROM candidates ORDER BY fitness DESC LIMIT ?",
                (args.limit,),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error):
                print("No candidates available for verification.")
                return
            raise
    for index, (candidate_hash, allocation_json) in enumerate(rows):
        payload = _payload(json.loads(allocation_json), args.start, args.end)
        if args.dry_run:
            print(json.dumps({"candidate_hash": candidate_hash, "payload": payload}))
        else:
            response = requests.post(ENDPOINT, json=payload, timeout=120)
            response.raise_for_status()
            print(json.dumps({"candidate_hash": candidate_hash, "response": response.json()}))
        if index + 1 < len(rows):
            time.sleep(1.5)


if __name__ == "__main__":
    main()
