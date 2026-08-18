"""Build chart-ready original/protected wealth paths for the expert top 10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.build_rebalance_research import _metrics, _method_returns
from scripts.search_expert_protection_grid import _overlay


ROOT = Path(__file__).parents[1]
SHORTLIST = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE = ROOT / "data" / "allocation-feasibility.json"
GRID = ROOT / "data" / "expert-protection-grid-results.json"
OUTPUT = ROOT / "data" / "expert-overlay-paths.json"
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-08-17")


def main() -> None:
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))["shortlist"]
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    preferred = grid["consensus"][1]["config"] if len(grid["consensus"]) > 1 else grid["consensus"][0]["config"]
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required)
    cash = prices["SHV"].pct_change().fillna(0.0) if "SHV" in prices else pd.Series(0.0, index=prices.index)
    spy = yf.download(["SPY"], start="2021-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]["SPY"].pct_change().fillna(0.0)
    paths = []
    for row in shortlist:
        base = _method_returns(prices.loc[START:END], row["allocation"], row["method"]).dropna()
        assets = prices.loc[base.index, list(row["allocation"])].pct_change().fillna(0.0)
        protected = _overlay(base, assets, cash, preferred)
        joined = pd.concat([base.rename("original"), protected.rename("protected"), spy.rename("SPY")], axis=1).dropna()
        wealth = (1.0 + joined).cumprod() * 100.0
        paths.append({"expert_rank": row["expert_rank"], "allocation_id": row["allocation_id"], "method": row["method"], "allocation_label": row["allocation_label"], "baseline_metrics": _metrics(base), "protected_metrics": _metrics(protected), "dates": [date.date().isoformat() for date in wealth.index], "original": wealth["original"].round(8).tolist(), "protected": wealth["protected"].round(8).tolist(), "SPY": wealth["SPY"].round(8).tolist()})
    OUTPUT.write_text(json.dumps({"period": {"start": START.date().isoformat(), "end": END.date().isoformat()}, "preferred_config": preferred, "paths": paths}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"paths": len(paths), "observations": len(paths[0]["dates"]), "preferred_config": preferred}, indent=2))


if __name__ == "__main__":
    main()
