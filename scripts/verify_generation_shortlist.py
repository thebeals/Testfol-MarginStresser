"""Sequentially verify the highest-scoring local candidates through Testfol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.testfol_api import fetch_backtest
from screener.live_verify import verify_sequential


def _load_shortlist(path: str | Path, limit: int) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: float(row.get("fitness", float("-inf"))), reverse=True)
    shortlist: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for row in rows:
        allocation = row.get("allocation", {})
        key = tuple(sorted((str(ticker), round(float(weight), 8)) for ticker, weight in allocation.items()))
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(row)
        if len(shortlist) >= limit:
            break
    return shortlist


def verify(shortlist: list[dict[str, object]], start: str, end: str, delay: float) -> list[dict[str, object]]:
    def request(candidate: dict[str, object]) -> dict[str, object]:
        allocation = {ticker: float(weight) * 100.0 for ticker, weight in candidate["allocation"].items()}
        total = sum(allocation.values())
        allocation[next(reversed(allocation))] += 100.0 - total
        raw = fetch_backtest(
            start,
            end,
            100000,
            0,
            "Monthly",
            60,
            True,
            candidate["rebalance_freq"],
            allocation,
            return_raw=True,
        )
        history = raw.get("charts", {}).get("history", [[], []])
        return {
            "local_fitness": candidate.get("fitness"),
            "allocation": candidate["allocation"],
            "rebalance_freq": candidate["rebalance_freq"],
            "testfol_errors": raw.get("errors", []),
            "testfol_start": history[0][0] if history[0] else None,
            "testfol_end": history[0][-1] if history[0] else None,
            "testfol_observations": len(history[0]),
            "testfol_stats": raw.get("stats", []),
        }

    return verify_sequential(shortlist, request, delay_seconds=delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/full-screener-generation-0.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--output", default="data/testfol-shortlist-verification.json")
    args = parser.parse_args()
    shortlist = _load_shortlist(args.input, args.limit)
    report = verify(shortlist, args.start, args.end, args.delay)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"requested": len(shortlist), "verified": len(report), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
