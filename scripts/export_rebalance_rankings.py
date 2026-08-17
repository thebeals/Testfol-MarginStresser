"""Export sortable rebalance matrix rankings to CSV and Excel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _method_fields(method: str) -> dict[str, object]:
    parts = method.split(":")
    if parts[0].startswith("EMA"):
        return {
            "method_family": "EMA",
            "ema_window": int(parts[0][3:]),
            "funding_policy": parts[1],
            "check_frequency": parts[2],
            "band_type": "",
            "band": "",
        }
    if parts[0] == "Calendar":
        return {
            "method_family": "Calendar",
            "ema_window": "",
            "funding_policy": "",
            "check_frequency": parts[1],
            "band_type": "",
            "band": "",
        }
    return {
        "method_family": parts[0].replace("Band", ""),
        "ema_window": "",
        "funding_policy": "",
        "check_frequency": parts[1],
        "band_type": parts[0].replace("Band", ""),
        "band": float(parts[2]),
    }


def export(matrix_path: str | Path, verification_path: str | Path, csv_path: str | Path, xlsx_path: str | Path) -> pd.DataFrame:
    matrix = [json.loads(line) for line in Path(matrix_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    verified = {(row["allocation_id"], row["method"]): row for row in verification}
    rows = []
    for row in matrix:
        method = _method_fields(row["method"])
        tf = verified.get((row["allocation_id"], row["method"]), {})
        tf_test = tf.get("testfol_test", {})
        rows.append(
            {
                "allocation_id": row["allocation_id"],
                "allocation": " / ".join(f"{ticker} {weight:.2%}" for ticker, weight in row["allocation"].items()),
                "method": row["method"],
                **method,
                "local_test_cagr": row["cagr"],
                "local_test_max_drawdown": row["max_drawdown"],
                "local_test_sharpe": row["sharpe"],
                "testfol_verified": bool(tf),
                "testfol_test_cagr": tf_test.get("cagr"),
                "testfol_test_max_drawdown": tf_test.get("max_drawdown"),
                "testfol_rebalancing_stats": json.dumps(tf.get("testfol_rebalancing_stats", []), sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows).sort_values("local_test_cagr", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank_by_local_test_cagr", range(1, len(frame) + 1))
    top = frame.head(100)
    top.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        top.to_excel(writer, sheet_name="Top 100", index=False)
        frame[frame["testfol_verified"]].to_excel(writer, sheet_name="Testfol Verified", index=False)
        frame.to_excel(writer, sheet_name="All 400", index=False)
    return top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data/rebalance-matrix-results.jsonl")
    parser.add_argument("--verification", default="data/rebalance-testfol-verification.json")
    parser.add_argument("--csv", default="data/rebalance-top-100.csv")
    parser.add_argument("--xlsx", default="data/rebalance-top-100.xlsx")
    args = parser.parse_args()
    top = export(args.matrix, args.verification, args.csv, args.xlsx)
    print(f"exported {len(top)} rows to {args.csv} and {args.xlsx}")


if __name__ == "__main__":
    main()
