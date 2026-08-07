"""Current Nasdaq-100 component sourcing with resilient fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import StringIO
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests


logger = logging.getLogger(__name__)

NASDAQ_COMPONENTS_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
WIKIPEDIA_COMPONENTS_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
REQUEST_TIMEOUT_SECONDS = 20
MIN_EXPECTED_SECURITIES = 95
MAX_EXPECTED_SECURITIES = 110

NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
WIKIPEDIA_HEADERS = {
    "User-Agent": "Testfol-MarginStresser/1.0 (Nasdaq-100 component refresh)",
}


class ComponentSourceError(RuntimeError):
    """Raised when a component source is unavailable or malformed."""


@dataclass
class ComponentSnapshot:
    """Normalized component membership plus display metadata."""

    components: pd.DataFrame
    as_of: pd.Timestamp
    source: str
    weight_basis: str
    warning: str | None = None


def _normalize_ticker(value: Any) -> str:
    """Normalize vendor symbols to the dash form accepted by yfinance."""
    return str(value).strip().upper().replace(".", "-")


def _validate_components(components: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"Ticker", "Name", "Weight"}
    missing = required.difference(components.columns)
    if missing:
        raise ComponentSourceError(
            f"{source} component data is missing columns: {', '.join(sorted(missing))}"
        )

    result = components.copy()
    result["Ticker"] = result["Ticker"].map(_normalize_ticker)
    result["Name"] = result["Name"].astype(str).str.strip()
    result = result[(result["Ticker"] != "") & (result["Name"] != "")]
    result = result.drop_duplicates("Ticker", keep="first")

    count = len(result)
    if not MIN_EXPECTED_SECURITIES <= count <= MAX_EXPECTED_SECURITIES:
        raise ComponentSourceError(
            f"{source} returned {count} unique securities; expected "
            f"{MIN_EXPECTED_SECURITIES}-{MAX_EXPECTED_SECURITIES}"
        )

    return result.reset_index(drop=True)


def _client_get(client, url: str, **kwargs):
    getter = client.get if client is not None else requests.get
    return getter(url, **kwargs)


def fetch_nasdaq_components(client=None) -> ComponentSnapshot:
    """Fetch current membership and market-cap proxy weights from Nasdaq."""
    try:
        response = _client_get(
            client,
            NASDAQ_COMPONENTS_URL,
            headers=NASDAQ_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        rows = (data.get("data") or {}).get("rows") or []
        as_of = pd.to_datetime(data.get("date"), errors="coerce")
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise ComponentSourceError(f"Nasdaq API request failed: {exc}") from exc

    if not rows:
        raise ComponentSourceError("Nasdaq API returned no component rows")
    if pd.isna(as_of):
        raise ComponentSourceError("Nasdaq API returned an invalid component date")

    frame = pd.DataFrame(rows)
    required = {"symbol", "companyName", "marketCap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ComponentSourceError(
            f"Nasdaq API rows are missing fields: {', '.join(sorted(missing))}"
        )

    market_cap = pd.to_numeric(
        frame["marketCap"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    total_market_cap = market_cap.sum(min_count=1)
    if pd.isna(total_market_cap) or total_market_cap <= 0:
        raise ComponentSourceError("Nasdaq API returned no usable market-cap values")

    components = pd.DataFrame(
        {
            "Ticker": frame["symbol"],
            "Name": frame["companyName"],
            # Displayed as percent in Streamlit; this is a ranking proxy, not an
            # official modified Nasdaq-100 index weight.
            "Weight": market_cap / total_market_cap * 100.0,
        }
    )
    components = _validate_components(components, "Nasdaq API")
    components = components.sort_values(
        ["Weight", "Ticker"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    return ComponentSnapshot(
        components=components,
        as_of=pd.Timestamp(as_of).normalize(),
        source="Nasdaq API",
        weight_basis="market-cap proxy (not official index weight)",
    )


def fetch_wikipedia_components(client=None, *, retrieved_at=None) -> ComponentSnapshot:
    """Fetch the current Wikipedia component table as a fallback/check."""
    try:
        response = _client_get(
            client,
            WIKIPEDIA_COMPONENTS_URL,
            headers=WIKIPEDIA_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise ComponentSourceError(f"Wikipedia component request failed: {exc}") from exc

    table = next(
        (
            candidate
            for candidate in tables
            if {"Ticker", "Company"}.issubset({str(col) for col in candidate.columns})
        ),
        None,
    )
    if table is None:
        raise ComponentSourceError("Wikipedia component table was not found")

    # Wikipedia does not publish weights in this table. Equal values keep the
    # existing UI schema usable while clearly marking the fallback basis.
    components = pd.DataFrame(
        {
            "Ticker": table["Ticker"],
            "Name": table["Company"],
            "Weight": 100.0 / len(table),
        }
    )
    components = _validate_components(components, "Wikipedia")
    components = components.sort_values("Ticker").reset_index(drop=True)

    as_of = pd.Timestamp(retrieved_at or pd.Timestamp.now()).normalize()
    return ComponentSnapshot(
        components=components,
        as_of=as_of,
        source="Wikipedia",
        weight_basis="equal-weight fallback (Wikipedia publishes membership only)",
    )


def load_static_components(
    components_file: str | Path,
    name_mapping_file: str | Path,
) -> ComponentSnapshot:
    """Load the existing SEC-derived component snapshot as an offline fallback."""
    try:
        frame = pd.read_csv(components_file)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame = frame[
            ~frame["Company"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
        ]
        frame = frame[(frame["Value"] > 0) & (frame["Date"] <= pd.Timestamp.now().normalize())]
        latest_date = frame["Date"].max()
        current = frame[frame["Date"] == latest_date].copy()
        with open(name_mapping_file, "r", encoding="utf-8") as handle:
            name_mapping = json.load(handle)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ComponentSourceError(f"Local component snapshot failed: {exc}") from exc

    if current.empty or pd.isna(latest_date):
        raise ComponentSourceError("Local component snapshot contains no usable rows")

    current["Ticker"] = current["Company"].map(name_mapping)
    current = current[current["Ticker"].notna()].copy()
    total_value = current["Value"].sum()
    current["Weight"] = current["Value"] / total_value * 100.0
    current = current.rename(columns={"Company": "Name"})
    components = _validate_components(
        current[["Ticker", "Name", "Weight"]], "Local SEC snapshot"
    )
    components = components.sort_values("Weight", ascending=False).reset_index(drop=True)

    return ComponentSnapshot(
        components=components,
        as_of=pd.Timestamp(latest_date).normalize(),
        source="Local SEC-derived snapshot",
        weight_basis="QQQ holding-value snapshot",
    )


def load_current_ndx_components(
    components_file: str | Path,
    name_mapping_file: str | Path,
    *,
    nasdaq_client=None,
    wikipedia_client=None,
) -> ComponentSnapshot:
    """Load Nasdaq membership, verify it with Wikipedia, then fall back safely."""
    errors: list[str] = []

    try:
        nasdaq = fetch_nasdaq_components(nasdaq_client)
    except ComponentSourceError as exc:
        errors.append(str(exc))
        nasdaq = None

    try:
        wikipedia = fetch_wikipedia_components(wikipedia_client)
    except ComponentSourceError as exc:
        errors.append(str(exc))
        wikipedia = None

    if nasdaq is not None:
        if wikipedia is None:
            return replace(
                nasdaq,
                warning="Wikipedia cross-check unavailable; using Nasdaq membership.",
            )

        nasdaq_tickers = set(nasdaq.components["Ticker"])
        wikipedia_tickers = set(wikipedia.components["Ticker"])
        if nasdaq_tickers == wikipedia_tickers:
            return replace(nasdaq, source="Nasdaq API (verified against Wikipedia)")

        only_nasdaq = sorted(nasdaq_tickers - wikipedia_tickers)
        only_wikipedia = sorted(wikipedia_tickers - nasdaq_tickers)
        warning = (
            "Nasdaq and Wikipedia component lists differ; Nasdaq remains authoritative. "
            f"Nasdaq only: {', '.join(only_nasdaq) or 'none'}. "
            f"Wikipedia only: {', '.join(only_wikipedia) or 'none'}."
        )
        return replace(nasdaq, warning=warning)

    if wikipedia is not None:
        return replace(
            wikipedia,
            warning="Nasdaq API unavailable; using Wikipedia membership. " + "; ".join(errors),
        )

    static = load_static_components(components_file, name_mapping_file)
    return replace(
        static,
        warning="Live component sources unavailable; using the local snapshot. "
        + "; ".join(errors),
    )
