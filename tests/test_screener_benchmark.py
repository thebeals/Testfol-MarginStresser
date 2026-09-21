import json
from pathlib import Path


def test_locked_spy_benchmark_has_all_walk_forward_periods() -> None:
    path = Path("data/screener-benchmark.json")
    benchmark = json.loads(path.read_text(encoding="utf-8"))

    assert benchmark["benchmark"] == "SPY"
    assert set(benchmark["periods"]) == {"train", "validation", "test"}
    assert all("cagr" in benchmark["periods"][period] for period in benchmark["periods"])
