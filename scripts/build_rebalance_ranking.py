"""Merge local rebalance matrix results with sequential Testfol verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(matrix_path: str | Path, verification_path: str | Path) -> dict[str, object]:
    matrix = [json.loads(line) for line in Path(matrix_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    by_allocation: dict[int, list[dict[str, object]]] = {}
    for row in matrix:
        by_allocation.setdefault(int(row["allocation_id"]), []).append(row)
    ranked = []
    for allocation_id, rows in by_allocation.items():
        local_best = max(rows, key=lambda row: row["cagr"])
        testfol = next(row for row in verification if int(row["allocation_id"]) == allocation_id)
        ranked.append(
            {
                "allocation_id": allocation_id,
                "allocation": local_best["allocation"],
                "best_local": local_best,
                "best_testfol": testfol,
            }
        )
    ranked.sort(key=lambda row: row["best_testfol"]["testfol_stats"].get("cagr", float("-inf")), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {"allocations": ranked, "methods_tested_per_allocation": len(by_allocation[1]) if by_allocation else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data/rebalance-matrix-results.jsonl")
    parser.add_argument("--verification", default="data/rebalance-testfol-verification.json")
    parser.add_argument("--output", default="data/rebalance-ranking.json")
    args = parser.parse_args()
    report = build(args.matrix, args.verification)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"allocations": len(report["allocations"]), "methods": report["methods_tested_per_allocation"]}, indent=2))


if __name__ == "__main__":
    main()
