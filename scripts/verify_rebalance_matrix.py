"""Verify the best Testfol-compatible rebalance method for each allocation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.testfol_api import fetch_backtest
from screener.live_verify import verify_sequential


def _parse_method(method: str) -> dict[str, object]:
    parts = method.split(":")
    if parts[0] == "Calendar":
        return {"rebalance": parts[1], "absolute_dev": 0.0, "relative_dev": 0.0}
    mode = parts[0].lower().replace("band", "")
    band_pct = float(parts[2]) * 100.0
    return {
        "rebalance": parts[1],
        "absolute_dev": band_pct if mode == "absolute" else 0.0,
        "relative_dev": band_pct if mode == "relative" else 0.0,
    }


def _key(allocation: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((ticker, round(float(weight), 8)) for ticker, weight in allocation.items()))


def _test_window_metrics(history: list[list[float]]) -> dict[str, float | int]:
    dates, values = history
    rows = [
        (datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(), float(value))
        for timestamp, value in zip(dates, values)
    ]
    rows = [(date, value) for date, value in rows if date >= "2021-01-01"]
    series = pd.Series(
        [value for _, value in rows],
        index=pd.to_datetime([date for date, _ in rows]),
    )
    returns = series.pct_change().dropna()
    wealth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "observations": int(len(returns)),
    }


def verify(matrix_path: str | Path, start: str, end: str, delay: float) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in Path(matrix_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    best: dict[int, dict[str, object]] = {}
    for row in rows:
        if row["method"].startswith("EMA"):
            continue
        current = best.get(row["allocation_id"])
        if current is None or row["cagr"] > current["cagr"]:
            best[row["allocation_id"]] = row

    def request(candidate: dict[str, object]) -> dict[str, object]:
        config = _parse_method(candidate["method"])
        allocation = {ticker: float(weight) * 100.0 for ticker, weight in candidate["allocation"].items()}
        allocation[next(reversed(allocation))] += 100.0 - sum(allocation.values())
        raw = fetch_backtest(
            start,
            end,
            100000,
            0,
            "Monthly",
            60,
            True,
            config["rebalance"],
            allocation,
            return_raw=True,
            absolute_dev=config["absolute_dev"],
            relative_dev=config["relative_dev"],
        )
        stats = (raw.get("stats") or [{}])[0] if isinstance(raw.get("stats"), list) else raw.get("stats", {})
        history = raw.get("charts", {}).get("history", [[], []])
        return {
            "allocation_id": candidate["allocation_id"],
            "allocation": candidate["allocation"],
            "method": candidate["method"],
            "local": {key: candidate[key] for key in ("cagr", "max_drawdown", "sharpe")},
            "testfol_errors": raw.get("errors", []),
            "testfol_stats": stats,
            "testfol_rebalancing_stats": raw.get("rebalancing_stats", []),
            "testfol_test": _test_window_metrics(history) if history[0] else {},
        }

    candidates = list(best.values())
    return verify_sequential(candidates, request, delay_seconds=delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data/rebalance-matrix-results.jsonl")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--output", default="data/rebalance-testfol-verification.json")
    args = parser.parse_args()
    report = verify(args.matrix, args.start, args.end, args.delay)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": len(report), "errors": sum(bool(row["testfol_errors"]) for row in report)}, indent=2))


if __name__ == "__main__":
    main()
