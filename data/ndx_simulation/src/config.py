import os

# --- General ---
# --- General ---
# Start from 'src' directory
# Base directory: data/ndx_simulation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ASSETS_DIR = os.path.join(DATA_DIR, "assets")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
NDX_CACHE_DIR = os.path.join(CACHE_DIR, "ndx_filings")
NPORT_CACHE_DIR = os.path.join(CACHE_DIR, "ndx_nport")
INDEX_FUND_CACHE_DIR = os.path.join(CACHE_DIR, "ndx_index_funds")
FREE_HISTORY_CACHE_DIR = os.path.join(CACHE_DIR, "free_history")
# Legacy alias
DOWNLOAD_DIR = NDX_CACHE_DIR

DEBUG_DIR = os.path.join(BASE_DIR, "debug") # Debug can stay at root of module

COMPONENTS_FILE = os.path.join(ASSETS_DIR, "nasdaq_components.csv")
NPORT_COMPONENTS_FILE = os.path.join(ASSETS_DIR, "nasdaq_nport_components.csv")
HOLDINGS_SNAPSHOTS_FILE = os.path.join(ASSETS_DIR, "ndx_holdings_snapshots.csv")
HOLDINGS_MANIFEST_FILE = os.path.join(ASSETS_DIR, "ndx_holdings_manifest.csv")
INDEX_FUND_POSITIONS_FILE = os.path.join(
    ASSETS_DIR, "ndx_index_fund_positions.csv"
)
INDEX_FUND_MANIFEST_FILE = os.path.join(
    ASSETS_DIR, "ndx_index_fund_manifest.csv"
)
WEIGHTS_FILE = os.path.join(RESULTS_DIR, "nasdaq_quarterly_weights.csv")
CHANGES_FILE = os.path.join(ASSETS_DIR, "nasdaq_changes.csv")
PRICE_CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.pkl")
BENCHMARK_TICKER = "QQQ"
# Preserve the published synthetic strategy history.  Earlier reconstructed
# events remain in the parent-index audit, but the March/June 2000 schedules
# are lower-confidence pre-bootstrap observations.
STRATEGY_START_DATE = "2000-06-30"

# --- Common Methodology Constants ---
# (Can be overridden by specific strategies)

# --- NDX Mega 1.0 Settings ---
MEGA1_TARGET_THRESHOLD = 0.47  # Select top 47% cumulative weight
MEGA1_BUFFER_THRESHOLD = 0.50  # Buffer to 50%
MEGA1_SINGLE_STOCK_CAP = 0.35  # Cap single stock at 35%

# --- NDX Mega 2.0 Settings ---
MEGA2_TARGET_THRESHOLD = 0.47  # Select top 47% cumulative weight
MEGA2_BUFFER_THRESHOLD = 0.50  # Buffer to 50%
MEGA2_SINGLE_STOCK_CAP = 0.30  # Cap single stock at 30%
MEGA2_MIN_CONSTITUENTS = 9     # Minimum 9 stocks

# --- NDX30 (Nasdaq-100 Top 30) Settings ---
NDX30_NUM_CONSTITUENTS = 30
NDX30_HARD_CAP = 0.225          # 22.5% individual cap
NDX30_SOFT_CAP = 0.045          # 4.5% threshold for aggregate constraint
NDX30_AGG_LIMIT = 0.48          # Sum of weights > 4.5% cannot exceed 48%

# --- Dual-Class Share Groups ---
# Nasdaq methodology: "the market capitalization of each company is the
# combined market capitalization of all eligible share classes."
# For SELECTION purposes (47% threshold), these tickers are treated as
# one company. For WEIGHTING, individual securities are kept.
# Maps secondary ticker → primary ticker (canonical).
DUAL_CLASS_GROUPS = {
    'GOOG':  'GOOGL',   # Alphabet Class C → Class A
    'FOXA':  'FOX',     # Fox Corp Class A → Class B
    'LBTYA': 'LBTYK',   # Liberty Global Class A → Series C
    'LBTYK': 'LBTYK',   # Identity (canonical)
    'DISCA': 'DISCK',   # Discovery Class A → Class C
    'DISCK': 'DISCK',   # Identity (canonical)
    'NWSA':  'NWS',     # News Corp Class A → Class B
    'NWS':   'NWS',     # Identity (canonical)
}

# --- Validation Settings ---
WEIGHT_TOLERANCE = 0.01        # Allow 1% deviation (handling fillers etc)
MAX_CAP_ITERATIONS = 20        # Increase from 10 to ensure convergence
