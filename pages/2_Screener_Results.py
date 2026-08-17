"""Streamlit results page for persisted screener candidates."""

from pathlib import Path
import json
import sqlite3

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parents[1]
DATABASES = {
    "Production generation with CAGR": ROOT / "data/screener-production-cagr.db",
    "Production generation": ROOT / "data/screener-production.db",
    "Bounded verification": ROOT / "data/screener.db",
}

st.set_page_config(page_title="Screener Results", layout="wide")
st.title("LETF Screener Results")
st.warning("These are preliminary strategy outputs. Allocation rules are under review; do not treat them as approved portfolios.")

available = {name: path for name, path in DATABASES.items() if path.exists()}
if not available:
    st.info("No screener database found. Run a generation first.")
    st.stop()

selection = st.sidebar.selectbox("Result set", list(available))
db_path = available[selection]
query = "SELECT * FROM candidates ORDER BY fitness DESC, created_at DESC"
with sqlite3.connect(db_path) as connection:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
    if "cagr" not in columns:
        connection.execute("ALTER TABLE candidates ADD COLUMN cagr REAL")
        connection.commit()
    candidates = pd.read_sql_query(query, connection)

if candidates.empty:
    st.info("The database is empty.")
    st.stop()

def _violations(value: str) -> list[str]:
    try:
        return json.loads(value).get("violations", [])
    except (TypeError, json.JSONDecodeError):
        return ["invalid diversification record"]


candidates["diversified"] = candidates["diversification_json"].map(lambda value: not _violations(value))
candidates["allocation"] = candidates["allocation_json"].map(
    lambda value: ", ".join(f"{ticker} {weight:.1%}" for ticker, weight in json.loads(value).items())
)

metric_columns = st.columns(4)
metric_columns[0].metric("Candidates", len(candidates))
metric_columns[1].metric("Diversified", int(candidates["diversified"].sum()))
metric_columns[2].metric("Robust", int((candidates["confidence_badge"] == "Robust").sum()))
metric_columns[3].metric("Best CAGR", f"{candidates['cagr'].max():.2%}" if candidates["cagr"].notna().any() else "Unavailable")

badges = st.multiselect("Confidence badge", sorted(candidates["confidence_badge"].dropna().unique()))
only_diversified = st.checkbox("Show only diversification-passing candidates", value=True)
if badges:
    candidates = candidates[candidates["confidence_badge"].isin(badges)]
if only_diversified:
    candidates = candidates[candidates["diversified"]]
st.caption(f"{len(candidates)} candidates")
display_columns = [
    "candidate_hash", "allocation", "rebalance_freq", "rotation_variant",
    "generation", "fitness", "mwrr", "cagr", "dsr", "confidence_badge", "diversified", "created_at",
]
st.dataframe(candidates[display_columns], use_container_width=True, hide_index=True)
