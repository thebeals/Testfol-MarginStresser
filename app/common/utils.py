import pandas as pd
import streamlit as st
import json
import os

APP_VERSION = "3.9.4"

CHANGELOG_MARKDOWN = """
## Changelog

### v3.9.4 - Restored Total-Return NDXMEGASPLIT
- Restored the ER-aware NDXMEGASPLIT variant using `NDXMEGASIM?L=2&E=0.95` alongside the newer `QQUPSIM` version.
- Kept identical hedge sleeves, maintenance requirements, and annual rebalance settings so the two return conventions can be compared directly.

### v3.9.3 - Organized Presets
- Grouped all presets into Core NDXMEGA, Single Asset & Proxies, Classic & Diversified, Leveraged Strategies, and Research & Experimental.
- Added category labels and stable category-aware ordering to the preset selector without changing stored preset names or strategy settings.

### v3.9.2 - QQUP Realized-Tracker Simulation
- Added `QQUPSIM`: a pre-inception 2x backfill built from reconstructed NDXMEGA price returns, Testfol-compatible DFF financing, and one 0.95% expense deduction.
- Spliced the backfill first to the official `NASDAQNDXMEGA` price index and then to actual adjusted QQUP returns, preserving every return across both boundaries.
- Updated the ER-aware NDXMEGASPLIT and standalone QQUP proxy presets to use `QQUPSIM`; theoretical `NDXMEGASIM?L=2&E=0.95` remains available.

### v3.9.1 - NDX Mega & Financing Accuracy
- Fixed the local shadow engine dropping the first investable return interval.
- Matched Testfol synthetic LETF financing with daily FRED `DFF`, simple `/252` costs, `SW=1.10`, signed `SP=sign(L)×0.40%`, and exact inverse-expense defaults.
- Separated real USD margin debt into an Actual/360 calendar-day ledger with monthly interest posting.
- Replaced historical QBIG splicing with official `NASDAQNDXMEGAT` and `NASDAQNDXMEGA2T` total-return series while retaining QBIG/QQUP for live quotes.
- Added regression coverage against the prior engine, official FRED data, IBKR interest examples, and Testfol's live API.

### v3.9.0 - Dynamic Nasdaq LETFs & Rate-Aware Financing
- Added dynamic Nasdaq rotation tickers such as `NDX_TOP8_ANN` and `NDXMEGA_TOP8_ANN`, with annual component schedules, company-level share-class collapsing, official NDX membership filtering, and `E=AUTO` individual LETF expense ratios.
- Updated the local shadow engine so synthetic `?L` leverage uses historical FRED Fed Funds financing plus a default 0.50% implementation spread instead of a flat 4% assumption.
- Added dynamic allocation schedule support to local backtests, including annual forced rebalances and inactive ticker handling so future baskets are not backfilled.
- Cleaned up NDXMEGASPLIT presets, removed rejected hedge duplicates, restored the non-legacy NDXMEGASPLIT names, and added the research cherry-picked enhanced preset with ERs.
- Documented dynamic Nasdaq tickers, LETF financing cost methodology, and the backtesting bias notes around static Mag 7 / cherry-picked baskets.
- Added regression coverage for dynamic Nasdaq expansion, official membership corrections, rate-sensitive leverage financing, and default implementation spread behavior.

### v3.8.0 - Risk Parity, SIM Tickers & DCA Cache Fixes
- Added a Risk Parity Split preset with DBMF, gold, bonds, international equity, NDX Mega leverage, and long Treasury sleeves.
- Centralized Testfol special ticker metadata so SIM/TR aliases share one provider fallback map across data fetching and shadow backtests.
- Expanded local component data support for Testfol preset tickers, including provider/API fallback handling and local ZEROX, cash, Treasury bill, fed funds, and inflation series fallbacks.
- Fixed stale local portfolio curves where DCA-funded portfolios could be displayed as rebased TWR instead of deposit-inclusive balances.
- Versioned backtest result cache keys and added stale-result detection so old TWR-shaped local results are recomputed before charting.
- Improved Global Capital inputs so whole-dollar Start Value and Cashflow values hide trailing cents, while cent values still display two decimals.
- Added regression coverage for special ticker support, local DCA value curves, stale result detection, cache key versioning, and optional cents formatting.

### v3.7.0 - Changelog Navigation & Sidebar Cleanup
- Moved the changelog out of the simulator control stack and into its own top-level navigation destination.
- Added a compact `Changelog` menu item beside `Simulator` and `Docs`, keeping release notes accessible without taking over the sidebar.
- Removed the bulky changelog expander from the main simulator sidebar so global settings, run controls, and API settings stay closer to the top.
- Shortened the documentation nav label to `Docs` to avoid wrapping in Streamlit's horizontal sidebar navigation.
- Centralized the visible app version as `APP_VERSION`, making future release bumps less scattered.
- Consolidated release notes into a dedicated changelog renderer instead of embedding a long markdown block in the simulator sidebar.

### v3.6.0 - Fresh Start Returns & Rebalance Timing Fix
- Fresh Start yearly column: per-year backtests for drift-free annual returns
- Fresh Start toggle: switch entire Returns Analysis to use fresh-start data
- Stitched fresh-start series for quarterly, monthly, daily, and drawdown breakdowns
- Rebalance timing fix: Custom mode now correctly triggers on target date instead of end-of-period
- Leveraged presets switched from Standard to Custom (Jan 1) rebalancing
- Single-ticker presets (QLD, QQUP) set to no rebalancing
- None rebalance mode added to UI

### v3.5.0 - Drawdowns Tab & Corrections Analysis
- New Drawdowns tab in Returns Analysis
- Corrections greater than 5% with SPY comparison and severity filter
- 70+ market event labels from 2000-2026
- Sortable duration columns

### v3.4.0 - NDX Simulation Accuracy
- Official Nasdaq membership auditing
- Survivorship bias dampening
- Price cache improvements

### v3.3.0 - Multi-Provider Price Data
- Polygon.io to yfinance automatic failover
- Component performance chart
- ER-aware presets for NDXMEGASPLIT with ERs

### v3.2.0 - Margin & Tax Overhaul
- Historical smart tax rates from 2013-2023
- Variable Fed Funds margin interest
- Draw start date and retirement income
- 164 regression tests

### v3.1.0 - Portfolio Margin & State Taxes
- Dynamic PM comparison and buy restrictions
- State tax library for all 50 states
- Rolling metrics and risk charts

### v3.0.0 - Architecture Refactor
- Split into `app/` package structure
- FastAPI backend with REST endpoints
- Shadow backtest engine with FIFO tax lots
- Disk-based HMAC cache

### v2.5.0 - Technical Analysis
- 200DMA, 150MA, and Munger 200WMA
- Weinstein Stage Analysis
- NDX-100 MA scanner
- Trader's Cheat Sheet

### v2.0.0 - Returns & Monte Carlo
- Seasonal summary, heatmaps, and distributions
- Annual, quarterly, monthly, and daily returns
- Monte Carlo simulation
- Benchmark comparisons

### v1.5.0 - Backtesting Engine
- Shadow yfinance backtester
- Custom rebalancing logic
- Tax calculations for federal and GLD handling
- NDXMEGA simulated tickers

### v1.0.0 - Initial Release
- Margin stress testing
- Log scaling and chart metrics
- Testfol API integration
"""


def color_return(val):
    if pd.isna(val): return ""
    color = '#00CC96' if val >= 0 else '#EF553B'
    return f'color: {color}'

def optional_cents_format(value) -> str:
    """Use cents only when the numeric value has non-zero cents."""
    try:
        cents = int(round(abs(float(value)) * 100)) % 100
    except (TypeError, ValueError, OverflowError):
        cents = 0
    return "%.2f" if cents else "%.0f"


def num_input(label, key, default, step, **kwargs):
    return st.number_input(
        label,
        value=st.session_state.get(key, default),
        step=step,
        key=key,
        **kwargs
    )

def sync_equity():
    sv = st.session_state.get("g_start", 10000.0)
    loan = st.session_state.get("starting_loan", 0.0)
    if sv > 0:
        st.session_state.equity_init = 100 * max(0, 1 - loan / sv)

def sync_loan():
    sv = st.session_state.get("g_start", 10000.0)
    eq = st.session_state.get("equity_init", 100.0)
    st.session_state.starting_loan = sv * max(0, 1 - eq / 100)

def get_presets_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "../../data/presets.json")

def load_presets():
    path = get_presets_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return []
    return []

def save_preset(preset_data):
    """
    Saves a preset dict to presets.json.
    Updates existing if name matches, else appends.
    """
    path = get_presets_path()
    current_presets = load_presets()
    
    # Check if exists
    existing_idx = next((i for i, p in enumerate(current_presets) if p["name"] == preset_data["name"]), -1)
    
    if existing_idx >= 0:
        current_presets[existing_idx] = preset_data
    else:
        current_presets.append(preset_data)
        
    with open(path, "w") as f:
        json.dump(current_presets, f, indent=4)

def delete_preset(name):
    path = get_presets_path()
    current_presets = load_presets()
    current_presets = [p for p in current_presets if p["name"] != name]
    with open(path, "w") as f:
        json.dump(current_presets, f, indent=4)

def resample_data(series: pd.Series, timeframe: str, method="ohlc") -> pd.DataFrame:
    if timeframe == "1D":
        if method == "ohlc":
            df = series.to_frame(name="Close")
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            return df
        else:
            return series

    rule_map = {
        "1W": "W-FRI",
        "1M": "ME",
        "3M": "QE",
        "1Y": "YE"
    }
    rule = rule_map.get(timeframe, "ME")

    if method == "ohlc":
        ohlc = series.resample(rule).ohlc()
        ohlc.columns = ["Open", "High", "Low", "Close"]
        return ohlc.dropna()
    elif method == "max":
        return series.resample(rule).max().dropna()
    else:
        return series.resample(rule).last().dropna()

def render_documentation():
    st.sidebar.title("📚 Documentation")
    st.sidebar.markdown("---")
    
    docs_dir = "docs"
    available_docs = {
        "User Guide": "user_guide.md",
        "Methodology": "methodology.md",
        "FAQ & Troubleshooting": "faq.md"
    }
    
    doc_selection = st.sidebar.radio("Select Topic", list(available_docs.keys()))
    
    doc_file = available_docs[doc_selection]
    doc_path = os.path.join(docs_dir, doc_file)
    
    if os.path.exists(doc_path):
        with open(doc_path, "r") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error(f"Documentation file not found: {doc_path}")


def render_changelog():
    st.sidebar.title("Changelog")
    st.sidebar.caption(f"Current version: v{APP_VERSION}")
    st.sidebar.markdown("---")

    st.title(f"Testfol Charting v{APP_VERSION}")
    st.markdown(CHANGELOG_MARKDOWN)
