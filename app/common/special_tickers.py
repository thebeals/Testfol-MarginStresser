"""Testfol preset ticker metadata and provider fallback aliases."""

from __future__ import annotations

import pandas as pd


# Provider fallbacks are only approximations for when Testfol's preset data is
# unavailable. The Testfol API remains the preferred source for these presets.
TESTFOL_PRESET_PROVIDER_FALLBACKS: dict[str, str] = {
    # Cash/Bills
    "TBILL": "BIL",
    "CASHX": "BIL",
    "EFFRX": "BIL",

    # US broad market and style boxes
    "SPYSIM": "SPY",
    "SPYTR": "SPY",
    "OEFSIM": "OEF",
    "MDYSIM": "MDY",
    "IJRSIM": "IJR",
    "IWMSIM": "IWM",
    "USMVSIM": "USMV",
    "VTISIM": "VTI",
    "VTITR": "VTI",
    "VTSIM": "VT",
    "VVSIM": "VOO",
    "VOOSIM": "VOO",
    "VTVSIM": "VTV",
    "VUGSIM": "VUG",
    "VOSIM": "VO",
    "VOESIM": "VOE",
    "VOTSIM": "VOT",
    "VBSIM": "VB",
    "VBRSIM": "VBR",
    "VBKSIM": "VBK",
    "IWCSIM": "IWC",
    "MTUMSIM": "MTUM",
    "REITSIM": "VNQ",

    # International
    "VXUSSIM": "VXUS",
    "VXUSX": "VXUS",
    "EFASIM": "EFA",
    "VEASIM": "VEA",
    "VWOSIM": "VWO",
    "VSSSIM": "VSS",
    "EFVSIM": "EFV",

    # Bonds
    "TLTSIM": "TLT",
    "TLTTR": "TLT",
    "ZROZSIM": "ZROZ",
    "ZROZX": "ZROZ",
    "IEFSIM": "IEF",
    "IEFTR": "IEF",
    "IEISIM": "IEI",
    "IEITR": "IEI",
    "SHYSIM": "SHY",
    "SHYTR": "SHY",
    "TIPSIM": "TIP",
    "BNDSIM": "BND",

    # Metals and commodities
    "GLDSIM": "GLD",
    "GOLDX": "GLD",
    "SLVSIM": "SLV",
    "GSGSIM": "GSG",
    "GSGTR": "GSG",
    "UUPSIM": "UUP",

    # Managed futures / return stacking
    "KMLMSIM": "KMLM",
    "KMLMX": "KMLM",
    "DBMFSIM": "DBMF",
    "DBMFX": "DBMF",
    "CAOSSIM": "CAOS",
    "GDESIM": "GDE",
    "RSSBSIM": "RSSB",
    "NTSDSIM": "NTSD",

    # Volatility
    "VIXSIM": "^VIX",
    "VOLIX": "^VIX",
    "SVIXSIM": "SVIX",
    "SVIXX": "SVIX",
    "UVIXSIM": "UVIX",
    "ZVOLSIM": "ZVOL",
    "ZIVBX": "ZVOL",

    # Crypto
    "BTCSIM": "BTC-USD",
    "BTCTR": "BTC-USD",
    "ETHSIM": "ETH-USD",
    "ETHTR": "ETH-USD",

    # Sector ETFs
    "XLBSIM": "XLB",
    "XLBTR": "XLB",
    "XLCSIM": "XLC",
    "XLCTR": "XLC",
    "XLESIM": "XLE",
    "XLETR": "XLE",
    "XLFSIM": "XLF",
    "XLFTR": "XLF",
    "XLISIM": "XLI",
    "XLITR": "XLI",
    "XLKSIM": "XLK",
    "XLKTR": "XLK",
    "XLPSIM": "XLP",
    "XLPTR": "XLP",
    "XLUSIM": "XLU",
    "XLUTR": "XLU",
    "XLVSIM": "XLV",
    "XLVTR": "XLV",
    "XLYSIM": "XLY",
    "XLYTR": "XLY",
    "QQQSIM": "QQQ",
    "QQQTR": "QQQ",

    # Leveraged/special products
    "FNGUSIM": "FNGU",
    "MCISIM": "MCI",

    # Legacy local alias
    "DIA_SIM": "DIA",
}


TESTFOL_PRESET_TICKERS: frozenset[str] = frozenset(
    set(TESTFOL_PRESET_PROVIDER_FALLBACKS)
    | {
        "ZEROX",
        "INFLATION",
    }
)


LIVE_PRICE_PROVIDER_FALLBACKS: dict[str, str] = {
    **TESTFOL_PRESET_PROVIDER_FALLBACKS,
    # Local synthetic Nasdaq sleeves are spliced to live ETFs for current quotes.
    "NDXMEGASIM": "QBIG",
    "NDXMEGA2SIM": "QBIG",
    "QQUPSIM": "QQUP",
    "NDX30SIM": "QTOP",
}

LIVE_LEVERAGED_PROVIDER_FALLBACKS: dict[tuple[str, float], str] = {
    ("AAPL", 2.0): "AAPU",
    ("AMZN", 2.0): "AMZU",
    ("AVGO", 2.0): "AVL",
    ("GOOG", 2.0): "GGLL",
    ("GOOGL", 2.0): "GGLL",
    ("META", 2.0): "METU",
    ("MSFT", 2.0): "MSFU",
    ("NVDA", 2.0): "NVDU",
    ("QQQ", 2.0): "QLD",
    ("QQQ", 3.0): "TQQQ",
    ("QQQSIM", 2.0): "QLD",
    ("QQQSIM", 3.0): "TQQQ",
    ("QQQTR", 2.0): "QLD",
    ("QQQTR", 3.0): "TQQQ",
    ("TSLA", 2.0): "TSLL",
    ("NDXMEGASIM", 2.0): "QQUP",
}


def clean_ticker_symbol(ticker: str) -> str:
    """Return the uppercase base ticker without query params or share overrides."""
    return str(ticker).split("?")[0].split("@")[0].strip().upper()


def _leverage_modifier(ticker: str) -> float | None:
    if "?" not in str(ticker):
        return None

    query = str(ticker).split("?", 1)[1]
    for pair in query.split("&"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key.upper() != "L":
            continue
        try:
            return float(value)
        except ValueError:
            return None

    return None


def is_fama_french_factor_ticker(ticker: str) -> bool:
    """Return True for Testfol's FF3*/FF5* factor pseudo-tickers."""
    base = clean_ticker_symbol(ticker)
    return base.startswith("FF3") or base.startswith("FF5")


def is_testfol_preset_ticker(ticker: str) -> bool:
    """Return True when ticker is one of Testfol's special preset tickers."""
    base = clean_ticker_symbol(ticker)
    return base in TESTFOL_PRESET_TICKERS or is_fama_french_factor_ticker(base)


def provider_fallback_ticker(ticker: str) -> str:
    """Return a provider-compatible fallback ticker when one is known."""
    base = clean_ticker_symbol(ticker)
    return TESTFOL_PRESET_PROVIDER_FALLBACKS.get(base, base)


def live_price_fallback_ticker(ticker: str) -> str:
    """Return a live quote symbol for Testfol/SIM tickers when one is known."""
    base = clean_ticker_symbol(ticker)
    leverage = _leverage_modifier(ticker)
    if leverage is not None:
        live_leveraged = LIVE_LEVERAGED_PROVIDER_FALLBACKS.get((base, leverage))
        if live_leveraged:
            return live_leveraged
    return LIVE_PRICE_PROVIDER_FALLBACKS.get(base, base)


def live_price_uses_native_leverage(ticker: str) -> bool:
    """Return True when live prices use an already-levered traded product."""
    base = clean_ticker_symbol(ticker)
    leverage = _leverage_modifier(ticker)
    return leverage is not None and (base, leverage) in LIVE_LEVERAGED_PROVIDER_FALLBACKS


def zero_return_series(start_date, end_date, name: str = "ZEROX") -> pd.Series:
    """Build a constant-price series for Testfol's 0% nominal return ticker."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        return pd.Series(dtype=float, name=name)
    index = pd.bdate_range(start_ts, end_ts)
    return pd.Series(100.0, index=index, name=name)
