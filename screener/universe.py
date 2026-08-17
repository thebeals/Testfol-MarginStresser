"""Ticker universe and risk-factor mapping.

Source of truth: SCREENER-SPEC.md "Ticker universe" section.

62 tickers total, grouped by actual risk factor (not strategy-packaging label):
50 tickers mapped directly to one of the 8 factors, plus 12 return-stacked
wrapper products whose underlying exposure stacks multiple factors.

The 8 factors:
  - Equity (includes leveraged UPRO/SSO/TQQQ and sector XLE/XLI/XLP/XLU/XLV/VPU)
  - Nominal duration (includes leveraged TMF/TYD/UBT)
  - Real/inflation-linked duration
  - Cash
  - Gold
  - Broad commodities
  - Managed futures / trend
  - Real estate

This is static data only. No original judgment is required to add a ticker; the
list and factor mapping are taken verbatim from SCREENER-SPEC.md.
"""

from __future__ import annotations

from enum import Enum


class Factor(str, Enum):
    """The 8 real risk factors in the universe."""

    EQUITY = "equity"
    NOMINAL_DURATION = "nominal_duration"
    REAL_DURATION = "real_duration"
    CASH = "cash"
    GOLD = "gold"
    COMMODITIES = "commodities"
    MANAGED_FUTURES = "managed_futures"
    REAL_ESTATE = "real_estate"


# Ticker -> Factor. Leveraged variants of equity and duration share their base
# factor; sector ETFs are a subset of the equity factor.
FACTOR_MAP: dict[str, Factor] = {
    # Equity (includes leveraged: UPRO, SSO, TQQQ; sector: XLE, XLI, XLP, XLU, XLV, VPU)
    "SPY": Factor.EQUITY,
    "VTI": Factor.EQUITY,
    "VOO": Factor.EQUITY,
    "VT": Factor.EQUITY,
    "VXUS": Factor.EQUITY,
    "VEA": Factor.EQUITY,
    "VWO": Factor.EQUITY,
    "EZU": Factor.EQUITY,
    "AVUV": Factor.EQUITY,
    "VBR": Factor.EQUITY,
    "VTV": Factor.EQUITY,
    "MTUM": Factor.EQUITY,
    "USMV": Factor.EQUITY,
    "QQQ": Factor.EQUITY,
    "PTLC": Factor.EQUITY,
    "UPRO": Factor.EQUITY,
    "SSO": Factor.EQUITY,
    "TQQQ": Factor.EQUITY,
    "XLE": Factor.EQUITY,
    "XLI": Factor.EQUITY,
    "XLP": Factor.EQUITY,
    "XLU": Factor.EQUITY,
    "XLV": Factor.EQUITY,
    "VPU": Factor.EQUITY,
    # Nominal duration (includes leveraged: TMF, TYD, UBT)
    "BND": Factor.NOMINAL_DURATION,
    "BNDX": Factor.NOMINAL_DURATION,
    "IEF": Factor.NOMINAL_DURATION,
    "TLT": Factor.NOMINAL_DURATION,
    "VGLT": Factor.NOMINAL_DURATION,
    "VGIT": Factor.NOMINAL_DURATION,
    "GOVT": Factor.NOMINAL_DURATION,
    "ZROZ": Factor.NOMINAL_DURATION,
    "EDV": Factor.NOMINAL_DURATION,
    "SCHP": Factor.NOMINAL_DURATION,
    "TMF": Factor.NOMINAL_DURATION,
    "TYD": Factor.NOMINAL_DURATION,
    "UBT": Factor.NOMINAL_DURATION,
    # Real/inflation-linked duration
    "TIP": Factor.REAL_DURATION,
    "VTIP": Factor.REAL_DURATION,
    # Cash
    "SGOV": Factor.CASH,
    "SHV": Factor.CASH,
    "BIL": Factor.CASH,
    # Gold
    "GLD": Factor.GOLD,
    "GLDM": Factor.GOLD,
    # Broad commodities
    "DBC": Factor.COMMODITIES,
    "PDBC": Factor.COMMODITIES,
    # Managed futures / trend
    "DBMF": Factor.MANAGED_FUTURES,
    "KMLM": Factor.MANAGED_FUTURES,
    "CTA": Factor.MANAGED_FUTURES,
    # Real estate
    "VNQ": Factor.REAL_ESTATE,
}


# Return-stacked wrappers map to the factor(s) they stack, not their own
# category (their underlying exposure varies by product). A wrapper may stack
# more than one factor.
WRAPPER_FACTORS: dict[str, tuple[Factor, ...]] = {
    "RSSB": (Factor.EQUITY, Factor.NOMINAL_DURATION),
    "RSST": (Factor.EQUITY, Factor.MANAGED_FUTURES),
    "RSBT": (Factor.NOMINAL_DURATION, Factor.MANAGED_FUTURES),
    "RSBY": (Factor.NOMINAL_DURATION, Factor.GOLD),
    "RSSY": (Factor.NOMINAL_DURATION, Factor.GOLD),
    # RSSX carries bitcoin exposure via the return-stacked wrapper. Flagged,
    # unresolved: in/out is Jonathan's call. Kept in the universe for now.
    "RSSX": (Factor.EQUITY,),
    "RSMV": (Factor.EQUITY, Factor.REAL_ESTATE),
    "NTSX": (Factor.EQUITY, Factor.NOMINAL_DURATION),
    "RPAR": (Factor.EQUITY, Factor.NOMINAL_DURATION, Factor.REAL_DURATION, Factor.COMMODITIES),
    "UPAR": (Factor.EQUITY, Factor.NOMINAL_DURATION, Factor.REAL_DURATION, Factor.COMMODITIES),
    "ALLW": (Factor.EQUITY, Factor.NOMINAL_DURATION),
    "CAOS": (Factor.EQUITY, Factor.NOMINAL_DURATION, Factor.REAL_DURATION, Factor.GOLD),
}


def all_tickers() -> list[str]:
    """Return every ticker in the universe (factor-mapped + wrappers)."""
    return list(FACTOR_MAP.keys()) + list(WRAPPER_FACTORS.keys())


def factor_tickers() -> list[str]:
    """Return the 50 factor-mapped tickers (excludes return-stacked wrappers)."""
    return list(FACTOR_MAP.keys())


def wrapper_tickers() -> list[str]:
    """Return the return-stacked wrapper tickers."""
    return list(WRAPPER_FACTORS.keys())


def factors_for(ticker: str) -> tuple[Factor, ...]:
    """Return the risk factor(s) for a ticker.

    Factor-mapped tickers return a single factor; wrappers return the tuple of
    factors they stack. Raises KeyError for an unknown ticker.
    """
    if ticker in FACTOR_MAP:
        return (FACTOR_MAP[ticker],)
    if ticker in WRAPPER_FACTORS:
        return WRAPPER_FACTORS[ticker]
    raise KeyError(f"Unknown ticker in universe: {ticker}")
