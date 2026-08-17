"""Verify the best Testfol-compatible rebalance method for each allocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.testfol_api import fetch_backtest
from screener.live_verify import verify_sequential


def _parse_method(method: str) -> dict[str, object]:
    parts = method.split(":")
    if parts[0] == "Calendar":
        return {"rebalance": parts[1], "absolute_dev": 0.0, "relative_dev": 0.0}
    mode = parts[0].lower().replace("band", "")
    return {
        "rebalance": parts[1],
        "absolute_dev": float(parts[2]) if mode == "absolute" else 0.0,
        "relative_dev": float(parts[2]) if mode == "relative" else 0.0,
    }


def _key(allocation: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((ticker, round(float(weight), 8)) for ticker, weight in allocation.items()))


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
        return {
            "allocation_id": candidate["allocation_id"],
            "allocation": candidate["allocation"],
            "method": candidate["method"],
            "local": {key: candidate[key] for key in ("cagr", "max_drawdown", "sharpe")},
            "testfol_errors": raw.get("errors", []),
            "testfol_stats": stats,
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
