import json
from pathlib import Path

from screener.results import load_screened_results


def test_canonical_screener_results_are_available() -> None:
    results = load_screened_results()

    assert all("CTA" not in result["local"]["allocation"] for result in results)
    assert all(not result["gate_reasons"] for result in results)
    benchmark = json.loads(Path("data/screener-benchmark.json").read_text(encoding="utf-8"))
    spy_test_cagr = benchmark["periods"]["test"]["cagr"]
    assert all(result["local"]["walk_forward"]["test"]["cagr"] > spy_test_cagr for result in results)
