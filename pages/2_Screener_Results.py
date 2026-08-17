"""Streamlit results page for persisted screener candidates."""

from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("data/screener.db")

st.set_page_config(page_title="Screener Results", layout="wide")
st.title("LETF Screener Results")

if not DB_PATH.exists():
    st.info(f"No screener database found at `{DB_PATH}`. Run a generation first.")
    st.stop()

query = "SELECT * FROM candidates ORDER BY fitness DESC, created_at DESC"
with __import__("sqlite3").connect(DB_PATH) as connection:
    candidates = pd.read_sql_query(query, connection)

if candidates.empty:
    st.info("The database is empty.")
    st.stop()

badges = st.multiselect("Confidence badge", sorted(candidates["confidence_badge"].dropna().unique()))
if badges:
    candidates = candidates[candidates["confidence_badge"].isin(badges)]
st.caption(f"{len(candidates)} candidates")
st.dataframe(candidates, use_container_width=True, hide_index=True)
