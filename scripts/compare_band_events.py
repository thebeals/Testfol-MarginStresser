"""Compare local band-breach dates with Testfol rebalance event dates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from app.services.testfol_api import fetch_backtest
from screener.hybrid_data import extend_hybrid_prices
from screener.live_verify import verify_sequential


TEST_START = pd.Timestamp("2021-01-01")


def _period_key(timestamp: pd.Timestamp, frequency: str) -> tuple[int, ...]:
    if frequency == "Monthly":
        return timestamp.year, timestamp.month
    if frequency == "Quarterly":
        return timestamp.year, timestamp.quarter
    return (timestamp.year,)


def _parse_method(method: str) -> tuple[str, str, float]:
    mode, frequency, band = method.split(":")
    return mode.removesuffix("Band").lower(), frequency, float(band)


def _local_event_dates(prices: pd.DataFrame, allocation: dict[str, float], mode: str, frequency: str, band: float) -> list[str]:
    values = {ticker: 100.0 * weight for ticker, weight in allocation.items()}
    returns = prices.loc[:, list(allocation)].pct_change().fillna(0.0)
    events: list[str] = []
    prior_period = None
    for timestamp, daily in returns.iterrows():
        for ticker in values:
            values[ticker] *= 1.0 + float(daily[ticker])
        total = sum(values.values())
        weights = {ticker: value / total for ticker, value in values.items()}
        breached = any(
            abs(weights[ticker] - target) > (band * target if mode == "relative" else band)
            for ticker, target in allocation.items()
        )
        period = _period_key(timestamp, frequency)
        scheduled = prior_period is not None and period != prior_period
        if breached or scheduled:
            event_timestamp = returns.index[returns.index.get_loc(timestamp) - 1] if scheduled else timestamp
            events.append(event_timestamp.date().isoformat())
            values = {ticker: total * target for ticker, target in allocation.items()}
        prior_period = period
    return events


def _testfol_event_dates(raw: dict[str, object]) -> list[str]:
    dates: list[str] = []
    for group in raw.get("rebalancing_events", []) or []:
        for event in group.get("events", []) if isinstance(group, dict) else []:
            if event and isinstance(event[0], str):
                dates.append(event[0])
    return sorted(set(dates))


def compare(prices: pd.DataFrame, candidates: list[dict[str, object]], start: str, end: str, delay: float) -> list[dict[str, object]]:
    local_prices = prices.loc[TEST_START:]

    def request(candidate: dict[str, object]) -> dict[str, object]:
        mode, frequency, band = _parse_method(candidate["method"])
        allocation = {ticker: float(weight) * 100.0 for ticker, weight in candidate["allocation"].items()}
        allocation[next(reversed(allocation))] += 100.0 - sum(allocation.values())
        local_dates = _local_event_dates(local_prices, candidate["allocation"], mode, frequency, band)
        try:
            kwargs = {"absolute_dev": band * 100.0 if mode == "absolute" else 0.0, "relative_dev": band * 100.0 if mode == "relative" else 0.0}
            raw = fetch_backtest(start, end, 100000, 0, "Monthly", 60, True, frequency, allocation, return_raw=True, **kwargs)
            api_dates = [event_date for event_date in _testfol_event_dates(raw) if event_date >= TEST_START.date().isoformat()]
            local_set, api_set = set(local_dates), set(api_dates)
            return {
                "allocation_id": candidate["allocation_id"],
                "method": candidate["method"],
                "local_event_count": len(local_dates),
                "testfol_event_count": len(api_dates),
                "exact_date_matches": len(local_set & api_set),
                "local_only_count": len(local_set - api_set),
                "testfol_only_count": len(api_set - local_set),
                "local_event_dates": local_dates,
                "testfol_event_dates": api_dates,
                "first_local_only": sorted(local_set - api_set)[:10],
                "first_testfol_only": sorted(api_set - local_set)[:10],
                "testfol_errors": raw.get("errors", []),
            }
        except Exception as exc:  # Keep the sequential run going after one API failure.
            return {"allocation_id": candidate["allocation_id"], "method": candidate["method"], "error": str(exc), "local_event_dates": local_dates}

    return verify_sequential(candidates, request, delay_seconds=delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/screener-results.json")
    parser.add_argument("--universe", default="data/allocation-feasibility.json")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--output", default="data/rebalance-band-event-parity.json")
    args = parser.parse_args()
    accepted = json.loads(Path(args.input).read_text(encoding="utf-8"))["accepted"]
    allocations = [row["local"]["allocation"] for row in accepted]
    universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end=args.end, auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for allocation in allocations for ticker in allocation})
    prices = prices.dropna(subset=required)
    matrix = [json.loads(line) for line in Path("data/rebalance-matrix-results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [row for row in matrix if row["method"].startswith(("RelativeBand", "AbsoluteBand"))]
    report = compare(prices, candidates, args.start, args.end, args.delay)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": len(report), "errors": sum(bool(row.get("error") or row.get("testfol_errors")) for row in report)}, indent=2))


if __name__ == "__main__":
    main()
