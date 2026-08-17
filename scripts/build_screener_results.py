"""Apply final local/Testfol gates and export the canonical screener results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _key(allocation: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((ticker, round(float(weight), 8)) for ticker, weight in allocation.items()))


def build(local_path: str | Path, testfol_paths: list[str | Path]) -> dict[str, object]:
    local_rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    testfol_rows = [
        row
        for testfol_path in testfol_paths
        for row in json.loads(Path(testfol_path).read_text(encoding="utf-8"))
    ]
    testfol_by_key = {_key(row["allocation"]): row for row in testfol_rows}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for local in local_rows:
        key = _key(local["allocation"])
        if key in seen:
            continue
        seen.add(key)
        verification = testfol_by_key.get(key)
        reasons: list[str] = []
        if verification is None:
            reasons.append("missing_testfol_verification")
        else:
            stats = (verification.get("testfol_stats") or [{}])[0]
            if verification.get("testfol_errors"):
                reasons.append("testfol_api_error")
            if float(stats.get("max_drawdown", -100.0)) < -50.0:
                reasons.append("testfol_max_drawdown")
            if float(stats.get("cagr", 0.0)) < 5.0:
                reasons.append("testfol_cagr")
            if int(verification.get("testfol_observations", 0)) < 1000:
                reasons.append("testfol_history_too_short")
        if not local.get("local_gate_passed", False):
            reasons.append("local_gate")
        if "CTA" in local["allocation"]:
            reasons.append("cta_excluded")
        result = {"local": local, "testfol": verification, "gate_reasons": reasons}
        (accepted if not reasons else rejected).append(result)
    return {
        "policy": {
            "cta_excluded": True,
            "local_test_max_drawdown_min": -0.50,
            "testfol_max_drawdown_min_pct": -50.0,
            "testfol_cagr_min_pct": 5.0,
            "testfol_min_observations": 1000,
        },
        "accepted": accepted,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default="data/research-shortlist.jsonl")
    parser.add_argument("--testfol", action="append", default=["data/research-testfol-verification.json"])
    parser.add_argument("--output", default="data/screener-results.json")
    args = parser.parse_args()
    report = build(args.local, args.testfol)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(report["accepted"]), "rejected": len(report["rejected"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
