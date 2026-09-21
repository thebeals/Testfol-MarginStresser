"""Build chart paths for the best balanced downside-idea configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.search_20_downside_ideas import END, IDEAS, START, _simulate

ROOT = Path(__file__).parents[1]
SHORTLIST = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE = ROOT / "data" / "allocation-feasibility.json"
RESULTS = ROOT / "data" / "expert-20-downside-ideas-results.json"
OUTPUT = ROOT / "data" / "expert-downside-idea-paths.json"


def main() -> None:
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))["shortlist"]
    study = json.loads(RESULTS.read_text(encoding="utf-8"))
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    raw = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)
    prices, _ = extend_hybrid_prices(raw["Close"].reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required).loc[START:END]
    spy = yf.download("SPY", start=START.strftime("%Y-%m-%d"), end=(END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True, progress=False, threads=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.reindex(prices.index).ffill().dropna()
    selected = study["robust"][:20]
    paths = []
    for row in selected:
        candidate = next(item for item in shortlist if item["expert_rank"] == row["expert_rank"])
        idea = next(item for item in IDEAS if item[0] == row["idea"])
        protected = _simulate(prices, candidate["allocation"], idea[2], *row["parameters"])
        wealth = 100.0 * (1.0 + protected).cumprod()
        paths.append({"path_id": len(paths) + 1, "idea": row["idea"], "variant": row["variant"], "expert_rank": row["expert_rank"], "parameters": row["parameters"], "cagr_delta": row["cagr_delta"], "dd_delta": row["dd_delta"], "dates": [date.strftime("%Y-%m-%d") for date in wealth.index], "protected": wealth.tolist()})
    spy = spy.loc[pd.to_datetime(paths[0]["dates"])]
    OUTPUT.write_text(json.dumps({"paths": paths, "SPY": (100.0 * spy / spy.iloc[0]).tolist(), "dates": paths[0]["dates"], "path_count": len(paths), "selection": "Top 20 balanced results from the 2,000-configuration study."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paths": len(paths), "observations": len(paths[0]["dates"])}, indent=2))


if __name__ == "__main__":
    main()
