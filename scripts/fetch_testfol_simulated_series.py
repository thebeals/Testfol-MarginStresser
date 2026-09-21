"""Fetch Testfol synthetic component series sequentially and cache them locally."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.testfol_api import fetch_backtest


ALIASES = {
    "DBMF": "DBMFSIM",
    "KMLM": "KMLMSIM",
}


def fetch_series(start: str, end: str, delay_seconds: float) -> dict[str, object]:
    result: dict[str, object] = {"date_range": {"start": start, "end": end}, "series": {}, "errors": {}}
    for index, (ticker, alias) in enumerate(ALIASES.items()):
        try:
            raw = fetch_backtest(
                start,
                end,
                100000,
                0,
                "Monthly",
                60,
                True,
                "None",
                {alias: 100.0},
                return_raw=True,
            )
            daily_returns = raw.get("daily_returns", [])
            if not daily_returns:
                raise ValueError(f"Testfol returned no daily returns for {alias}")
            result["series"][ticker] = {
                "source_ticker": alias,
                "daily_returns_pct": daily_returns,
                "observations": len(daily_returns),
                "start": daily_returns[0][0],
                "end": daily_returns[-1][0],
            }
        except Exception as error:
            result["errors"][ticker] = {"source_ticker": alias, "message": str(error)}
        if index + 1 < len(ALIASES):
            time.sleep(delay_seconds)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--output", default="data/testfol-simulated-returns.json")
    args = parser.parse_args()
    result = fetch_series(args.start, args.end, args.delay)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"series": {key: value["observations"] for key, value in result["series"].items()}, "errors": result["errors"]}, indent=2))


if __name__ == "__main__":
    main()
