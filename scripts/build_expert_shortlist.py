"""Select and explain an expert shortlist from the drawdown-gated research set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
REPORT_PATH = ROOT / "data" / "rebalance-research-report.json"
BENCHMARK_PATH = ROOT / "data" / "screener-benchmark.json"
OUTPUT_PATH = ROOT / "data" / "rebalance-expert-shortlist.json"
CSV_PATH = ROOT / "data" / "rebalance-expert-shortlist.csv"

LEVERAGED = {"SSO", "TQQQ", "UPRO", "TMF", "TYD"}


def _asset_class(ticker: str) -> str:
    if ticker == "DBC":
        return "commodities"
    if ticker == "DBMF":
        return "managed futures"
    if ticker == "GLD":
        return "gold"
    if ticker in {"TIP", "TLT", "IEF", "EDV"}:
        return "bonds"
    if ticker == "SHV":
        return "cash"
    if ticker == "VNQ":
        return "real estate"
    if ticker == "VEA":
        return "international equity"
    return "US equity"


def _diversification(row: dict[str, object]) -> dict[str, object]:
    allocation = row["allocation"]
    classes = [_asset_class(ticker) for ticker in allocation]
    leveraged_weight = sum(weight for ticker, weight in allocation.items() if ticker in LEVERAGED)
    largest_weight = max(allocation.values())
    overlap = float(allocation.get("QQQ", 0.0) + allocation.get("TQQQ", 0.0))
    return {
        "asset_classes": sorted(set(classes)),
        "asset_class_count": len(set(classes)),
        "largest_weight": largest_weight,
        "leveraged_weight": leveraged_weight,
        "qqq_tqqq_overlap": overlap,
    }


def _score(row: dict[str, object], spy_cagr: float) -> tuple[float, dict[str, float]]:
    test = row["local_test"]
    dd = abs(float(test["max_drawdown"]))
    cagr_excess = max(0.0, float(test["cagr"]) - spy_cagr)
    div = _diversification(row)
    resilience = row["regimes"]
    crisis_returns = [resilience[name]["portfolio"].get("total_return", 0.0) for name in ("2008_financial_crisis", "2020_covid_shock", "2022_inflation_bear")]
    crisis_score = max(0.0, min(1.0, (sum(crisis_returns) / len(crisis_returns) + 0.5)))
    parts = {
        "cagr_excess": min(cagr_excess / 0.08, 1.0),
        "drawdown_control": max(0.0, 1.0 - dd / 0.20),
        "sharpe": max(0.0, min(float(test["sharpe"]) / 1.5, 1.0)),
        "rolling_spy": float(row["rolling_3y_vs_spy"]["beat_rate"]),
        "rolling_vt": float(row["rolling_3y_vs_vt"]["beat_rate"]),
        "diversification": min(div["asset_class_count"] / 5.0, 1.0),
        "leverage_control": max(0.0, 1.0 - div["leveraged_weight"]),
        "crisis_resilience": crisis_score,
        "testfol_validation": 1.0 if row["testfol_validated"] else 0.0,
    }
    weights = {
        "cagr_excess": 0.18,
        "drawdown_control": 0.18,
        "sharpe": 0.12,
        "rolling_spy": 0.10,
        "rolling_vt": 0.08,
        "diversification": 0.10,
        "leverage_control": 0.08,
        "crisis_resilience": 0.08,
        "testfol_validation": 0.08,
    }
    return sum(parts[key] * weights[key] for key in parts), parts


def _rationale(row: dict[str, object], spy_cagr: float) -> dict[str, object]:
    div = _diversification(row)
    test = row["local_test"]
    strengths = [
        f"Test CAGR of {test['cagr']:.1%}, {test['cagr'] - spy_cagr:+.1%} versus SPY.",
        f"Observed max drawdown of {test['max_drawdown']:.1%} in the locked test window.",
        f"Spans {div['asset_class_count']} asset classes: {', '.join(div['asset_classes'])}.",
    ]
    if row["testfol_validated"]:
        strengths.append("The exact rebalance method has a zero-error Testfol verification record.")
    if div["leveraged_weight"] <= 0.15:
        strengths.append("Keeps explicit leveraged sleeves relatively contained.")
    weaknesses = []
    if div["leveraged_weight"] > 0.15:
        weaknesses.append(f"Explicit leveraged ETF exposure is {div['leveraged_weight']:.1%}; losses can compound sharply.")
    if div["qqq_tqqq_overlap"] > 0.20:
        weaknesses.append(f"QQQ/TQQQ overlap is {div['qqq_tqqq_overlap']:.1%}, so the equity sleeve is less diversified than the ticker count suggests.")
    if row["rolling_3y_vs_spy"]["beat_rate"] < 0.40:
        weaknesses.append("It did not beat SPY in most rolling three-year windows.")
    if not row["testfol_validated"]:
        weaknesses.append("The exact dynamic/local method is not directly Testfol-validated.")
    weaknesses.append(f"Worst observed rolling three-month return was {row['recovery']['worst_rolling_3m']:.1%}.")
    return {
        "why_consider": strengths,
        "what_can_go_wrong": weaknesses,
        "black_swan_note": "Diversifiers may help in the specific historical shock regimes shown, but they can correlate during a liquidity crisis. No rule here guarantees protection from a simultaneous fall in equities, credit, commodities, gold, and leveraged ETFs.",
    }


def build() -> dict[str, object]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    spy_cagr = float(benchmark["periods"]["test"]["cagr"])
    eligible = [row for row in report["candidates"] if float(row["local_test"]["cagr"]) > spy_cagr]
    scored = []
    for row in eligible:
        score, components = _score(row, spy_cagr)
        scored.append({**row, "expert_score": score, "score_components": components})
    scored.sort(key=lambda row: row["expert_score"], reverse=True)

    selected = []
    allocation_counts: dict[int, int] = {}
    family_counts: dict[str, int] = {}
    for row in scored:
        allocation_id = int(row["allocation_id"])
        family = str(row["family"])
        if allocation_counts.get(allocation_id, 0) >= 3 or family_counts.get(family, 0) >= 5:
            continue
        allocation_counts[allocation_id] = allocation_counts.get(allocation_id, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        selected.append(row)
        if len(selected) == 10:
            break

    for rank, row in enumerate(selected, start=1):
        row["expert_rank"] = rank
        row["diversification"] = _diversification(row)
        row["rationale"] = _rationale(row, spy_cagr)
    return {
        "selection_rule": "Expert shortlist: candidates must beat the locked SPY test CAGR and pass the 20% method-level drawdown gate. Score blends CAGR excess, drawdown, Sharpe, rolling SPY/VT win rates, asset-class diversification, explicit leverage, crisis-regime returns, and exact Testfol validation. No allocation appears more than three times.",
        "spy_test_cagr": spy_cagr,
        "candidate_pool": len(eligible),
        "shortlist": selected,
        "limitations": [
            "This is a judgment-assisted historical shortlist, not personalized investment advice or a probability forecast.",
            "The historical sample contains limited true black-swan observations and may not represent future correlations.",
            "EMA methods are local research methods unless the exact method is separately Testfol-validated.",
            "No margin, taxes, slippage, or investor behavior are modeled here.",
        ],
    }


def main() -> None:
    output = build()
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for row in output["shortlist"]:
        rows.append({
            "expert_rank": row["expert_rank"], "allocation_id": row["allocation_id"], "allocation": row["allocation_label"],
            "method": row["method"], "expert_score": row["expert_score"], "test_cagr": row["local_test"]["cagr"],
            "max_drawdown": row["local_test"]["max_drawdown"], "sharpe": row["local_test"]["sharpe"],
            "rolling_3y_spy_beat_rate": row["rolling_3y_vs_spy"]["beat_rate"], "rolling_3y_vt_beat_rate": row["rolling_3y_vs_vt"]["beat_rate"],
            "testfol_validated": row["testfol_validated"], "asset_classes": "; ".join(row["diversification"]["asset_classes"]),
        })
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(json.dumps({"pool": output["candidate_pool"], "shortlist": len(output["shortlist"])}, indent=2))


if __name__ == "__main__":
    main()
