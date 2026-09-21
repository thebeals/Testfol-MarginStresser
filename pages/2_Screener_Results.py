"""Streamlit page for the canonical, baseline-qualified screener results."""

from pathlib import Path

import streamlit as st

from app.ui.screener_results import render_screened_results


st.set_page_config(page_title="Screener Results", layout="wide")
render_screened_results(Path(__file__).parents[1] / "data" / "screener-results.json")
