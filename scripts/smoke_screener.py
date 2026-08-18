"""Offline smoke test for the screener hot path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.db import CandidateStore
from screener.loop import CandidateTask, evaluate_candidate, evaluate_candidates_parallel
from screener.universe import all_tickers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if len(all_tickers()) != 62:
        raise RuntimeError("ticker universe is incomplete")
    dates = pd.bdate_range("2020-01-01", periods=max(260, args.limit))
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (len(dates), 2)), index=dates, columns=["SPY", "TLT"])
    result = evaluate_candidate(returns, CandidateTask({"SPY": 0.55, "TLT": 0.45}, "Monthly", 0))
    store = CandidateStore(":memory:")
    assert store.save({**result.__dict__, "allocation": result.allocation})
    assert store.contains(result.allocation, result.rebalance_freq)
    store.close()
    output_path = Path(".smoke-results.jsonl")
    database_path = Path(".smoke-results.db")
    parallel = evaluate_candidates_parallel(
        returns,
        [CandidateTask({"SPY": 0.55, "TLT": 0.45}, "None", 0), CandidateTask({"SPY": 0.60, "TLT": 0.40}, "Yearly", 1)],
        output_path=output_path,
        db_path=database_path,
        workers=2,
    )
    assert len(parallel) == 2
    output_path.unlink(missing_ok=True)
    database_path.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(f".smoke-results.db{suffix}").unlink(missing_ok=True)
    print(f"smoke ok: limit={args.limit}, candidate={result.confidence_badge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
