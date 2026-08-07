"""Build and merge free archival prices for historical NDX constituents.

Sources, in priority order:

* Nasdaq Data Link's public-domain WIKI end-of-day database.  Raw close plus
  split ratios reconstructs price return; adjusted close supplies total return.
* Carnegie Mellon Database Group's historical US-stock archive.  It extends
  coverage for delisted dot-com constituents through 2006.  The raw archive is
  cached locally and split discontinuities are normalized conservatively.
* Stooq's CC0 U.S. stock archive, used only after adjusted-series identity
  checks against SEC holdings.
* Public monthly CompaniesMarketCap histories for four otherwise orphaned
  dot-com constituents, interpolated only inside their observed date range.

Ticker identity is checked against SEC fund holdings (value / shares) before a
known reused symbol is admitted.  This prevents a modern company that inherited
an old ticker from silently entering the historical index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tarfile
import zipfile

import numpy as np
import pandas as pd
import requests

import config
from official_index_data import LOCAL_MEMBERSHIP_FILES


WIKI_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites"
)
WIKI_LICENSE_URL = "https://docs.data.nasdaq.com/v1.0/docs/in-depth-usage"
CMU_URL = "https://db.cs.cmu.edu/files/data/stocks/history.tar.gz"
CMU_INFO_URL = "https://www.cs.cmu.edu/~pavlo/datasets/stocks/"
STOOQ_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    "?datasetVersionNumber=3"
)
STOOQ_INFO_URL = (
    "https://www.kaggle.com/datasets/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs"
)
CMC_MONTHLY = {
    "NXTL": "nextel-communications",
    "WCOM": "worldcom",
    "GLBC": "global-crossing",
    "LCOS": "lycos",
}

PRICE_RETURN_FILE = os.path.join(
    config.FREE_HISTORY_CACHE_DIR, "free_history_price_return.parquet"
)
TOTAL_RETURN_FILE = os.path.join(
    config.FREE_HISTORY_CACHE_DIR, "free_history_total_return.parquet"
)
MANIFEST_FILE = os.path.join(
    config.FREE_HISTORY_CACHE_DIR, "free_history_manifest.csv"
)
WIKI_ARCHIVE = os.path.join(config.FREE_HISTORY_CACHE_DIR, "wiki_prices.zip")
CMU_ARCHIVE = os.path.join(config.FREE_HISTORY_CACHE_DIR, "cmu_history.tar.gz")
STOOQ_ARCHIVE = os.path.join(config.FREE_HISTORY_CACHE_DIR, "stooq_us.zip")

# Symbols known to have represented unrelated issuers in modern databases.
# They are admitted only after matching SEC-observed per-share values.
REUSED_SYMBOLS = {"GOLD", "GIB", "MNST", "SUNW", "VRTS"}
SPLIT_FACTORS = np.array(
    [10.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.25, 0.8, 2 / 3, 0.5, 1 / 3, 0.25, 0.2, 0.125, 0.1]
)


def required_tickers():
    # QQQ is the reference path for the SEC relative-weight fallback even
    # though it is an ETF rather than an NDX constituent.
    tickers = {"QQQ"}
    membership = pd.read_csv(LOCAL_MEMBERSHIP_FILES["NDX"])
    for value in membership.get("Tickers", []):
        tickers.update(str(value).split("|"))
    if os.path.exists(config.HOLDINGS_SNAPSHOTS_FILE):
        holdings = pd.read_csv(config.HOLDINGS_SNAPSHOTS_FILE, usecols=["Ticker"])
        tickers.update(holdings["Ticker"].dropna().astype(str))
    return sorted(ticker for ticker in tickers if ticker and ticker != "nan")


def _download(url, path, refresh=False):
    if os.path.exists(path) and os.path.getsize(path) > 1024 and not refresh:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.part"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(temporary, path)
    return path


def _as_series(frame, date_column, value_column):
    result = frame[[date_column, value_column]].copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.dropna().drop_duplicates(date_column, keep="last")
    return result.set_index(date_column)[value_column].sort_index()


def _price_return_series(close, split_ratio=None):
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if close.empty:
        return close, 0
    ratio = close / close.shift(1)
    corrections = 0
    if split_ratio is not None:
        split_ratio = pd.to_numeric(split_ratio, errors="coerce").reindex(close.index)
        split_ratio = split_ratio.fillna(1.0).replace(0, 1.0)
        corrections = int((split_ratio != 1.0).sum())
        ratio = ratio * split_ratio
    ratio.iloc[0] = 1.0
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return ratio.cumprod(), corrections


def _conservative_split_adjust(close):
    """Remove only near-rational discontinuities from raw CMU closes."""
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if close.empty:
        return close, 0
    ratio = close / close.shift(1)
    adjusted = ratio.copy()
    corrections = 0
    for date, value in ratio.dropna().items():
        if value <= 0 or abs(math.log(value)) < math.log(1.35):
            continue
        residuals = np.abs(np.log(value * SPLIT_FACTORS))
        best = int(np.argmin(residuals))
        # Seven percent permits normal split-day movement but avoids treating
        # most genuine dot-com crashes as corporate actions.
        if residuals[best] <= math.log(1.07):
            adjusted.loc[date] = value * SPLIT_FACTORS[best]
            corrections += 1
    adjusted.iloc[0] = 1.0
    adjusted = adjusted.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return adjusted.cumprod(), corrections


def _holding_anchors():
    if not os.path.exists(config.HOLDINGS_SNAPSHOTS_FILE):
        return pd.DataFrame(columns=["Date", "Ticker", "ImpliedPrice"])
    frame = pd.read_csv(
        config.HOLDINGS_SNAPSHOTS_FILE,
        usecols=["Date", "Ticker", "Shares", "Value", "MappingScore"],
        parse_dates=["Date"],
    )
    frame["Shares"] = pd.to_numeric(frame["Shares"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame["MappingScore"] = pd.to_numeric(frame["MappingScore"], errors="coerce")
    frame = frame[
        (frame["Shares"] > 0)
        & (frame["Value"] > 0)
        & (frame["MappingScore"].fillna(0) >= 0.85)
    ].copy()
    frame["ImpliedPrice"] = frame["Value"] / frame["Shares"]
    return frame


def _identity_errors(ticker, raw_close, anchors):
    matches = anchors[anchors["Ticker"] == ticker]
    errors = []
    for row in matches.itertuples():
        window = raw_close.loc[
            (raw_close.index >= row.Date - pd.Timedelta(days=7))
            & (raw_close.index <= row.Date + pd.Timedelta(days=7))
        ]
        if window.empty:
            continue
        nearest = int(np.argmin(np.abs(window.index - row.Date)))
        archive_price = float(window.iloc[nearest])
        if archive_price <= 0:
            continue
        ratio = float(row.ImpliedPrice) / archive_price
        # Some SEC schedules report value in thousands.  Unit scale is not an
        # identity mismatch, so compare after the closest standard scale.
        unit = min((0.001, 1.0, 1000.0), key=lambda value: abs(math.log(ratio / value)))
        errors.append(abs(ratio / unit - 1.0))
    return errors


def _identity_status(ticker, raw_close, anchors):
    errors = _identity_errors(ticker, raw_close, anchors)
    if errors:
        median = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        accepted = median <= 0.15 and p90 <= 0.40
    else:
        median = p90 = np.nan
        accepted = ticker not in REUSED_SYMBOLS
    return accepted, len(errors), median, p90


def _scaled_identity_status(ticker, prices, anchors):
    """Validate an adjusted series whose absolute level is arbitrary."""
    matches = anchors[anchors["Ticker"] == ticker]
    ratios = []
    for row in matches.itertuples():
        window = prices.loc[
            (prices.index >= row.Date - pd.Timedelta(days=7))
            & (prices.index <= row.Date + pd.Timedelta(days=7))
        ]
        if window.empty:
            continue
        nearest = int(np.argmin(np.abs(window.index - row.Date)))
        price = float(window.iloc[nearest])
        if price > 0:
            ratios.append(float(row.ImpliedPrice) / price)
    if len(ratios) >= 2:
        scale = float(np.median(ratios))
        # Adjusted vendors may restate the level after a split while older SEC
        # schedules retain the then-current raw price.  Treat exact rational
        # split multiples as the same identity, but not arbitrary level jumps.
        identity_factors = np.append(SPLIT_FACTORS, 1.0)
        errors = [
            float(np.min(np.abs((ratio / scale) * identity_factors - 1.0)))
            for ratio in ratios
        ]
        median = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        return median <= 0.15 and p90 <= 0.40, len(errors), median, p90
    accepted = ticker not in REUSED_SYMBOLS
    return accepted, len(ratios), np.nan, np.nan


def _wiki_history(tickers, anchors, refresh=False):
    _download(WIKI_URL, WIKI_ARCHIVE, refresh=refresh)
    wanted = set(tickers)
    pieces = []
    with zipfile.ZipFile(WIKI_ARCHIVE) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            for chunk in pd.read_csv(
                handle,
                usecols=["ticker", "date", "close", "split_ratio", "adj_close"],
                chunksize=250000,
            ):
                selected = chunk[chunk["ticker"].astype(str).str.upper().isin(wanted)]
                if not selected.empty:
                    pieces.append(selected)
    if not pieces:
        return {}, {}, []
    frame = pd.concat(pieces, ignore_index=True)
    price_return, total_return, manifest = {}, {}, []
    for ticker, group in frame.groupby(frame["ticker"].astype(str).str.upper()):
        group = group.sort_values("date")
        raw = _as_series(group, "date", "close")
        splits = group.assign(date=pd.to_datetime(group["date"])).set_index("date")[
            "split_ratio"
        ]
        price, corrections = _price_return_series(raw, splits)
        total = _as_series(group, "date", "adj_close")
        accepted, anchor_count, median, p90 = _identity_status(ticker, raw, anchors)
        if accepted:
            price_return[ticker] = price
            total_return[ticker] = total
        manifest.append(
            {
                "Ticker": ticker,
                "Source": "Nasdaq-Data-Link-WIKI",
                "Rows": len(raw),
                "StartDate": raw.index.min().date(),
                "EndDate": raw.index.max().date(),
                "SplitCorrections": corrections,
                "AnchorCount": anchor_count,
                "MedianAnchorError": median,
                "P90AnchorError": p90,
                "Accepted": accepted,
                "SourceURL": WIKI_URL,
                "LicenseURL": WIKI_LICENSE_URL,
            }
        )
    return price_return, total_return, manifest


def _cmu_ticker(member_name):
    base = os.path.basename(member_name)
    if not base.lower().endswith(".csv"):
        return ""
    return os.path.splitext(base)[0].upper()


def _cmu_history(tickers, anchors, refresh=False):
    _download(CMU_URL, CMU_ARCHIVE, refresh=refresh)
    wanted = set(tickers)
    price_return, total_return, manifest = {}, {}, []
    with tarfile.open(CMU_ARCHIVE, "r:gz") as archive:
        for member in archive:
            ticker = _cmu_ticker(member.name)
            if ticker not in wanted or not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                frame = pd.read_csv(
                    handle,
                    comment="#",
                    header=None,
                    names=["DATE", "PRICE", "VOLUME", "OPEN", "LOW", "HIGH"],
                )
            except Exception:
                continue
            columns = {str(column).upper(): column for column in frame.columns}
            if "DATE" not in columns or "PRICE" not in columns:
                continue
            raw = _as_series(frame, columns["DATE"], columns["PRICE"])
            price, corrections = _conservative_split_adjust(raw)
            accepted, anchor_count, median, p90 = _identity_status(
                ticker, raw, anchors
            )
            if accepted:
                price_return[ticker] = price
                # Dividends are unavailable.  For dot-com-era NDX names this
                # price-return series is still far better than a QQQ proxy.
                total_return[ticker] = price
            manifest.append(
                {
                    "Ticker": ticker,
                    "Source": "CMU-DB-Historical-Stocks",
                    "Rows": len(raw),
                    "StartDate": raw.index.min().date(),
                    "EndDate": raw.index.max().date(),
                    "SplitCorrections": corrections,
                    "AnchorCount": anchor_count,
                    "MedianAnchorError": median,
                    "P90AnchorError": p90,
                    "Accepted": accepted,
                    "SourceURL": CMU_URL,
                    "LicenseURL": CMU_INFO_URL,
                }
            )
    return price_return, total_return, manifest


def _stooq_ticker(member_name):
    base = os.path.basename(member_name).lower()
    match = re.match(r"([a-z0-9.-]+)\.us\.txt$", base)
    if not match:
        return ""
    return match.group(1).replace("-", ".").upper()


def _stooq_history(tickers, anchors, refresh=False):
    _download(STOOQ_URL, STOOQ_ARCHIVE, refresh=refresh)
    wanted = set(tickers)
    price_return, total_return, manifest = {}, {}, []
    with zipfile.ZipFile(STOOQ_ARCHIVE) as archive:
        for member_name in archive.namelist():
            ticker = _stooq_ticker(member_name)
            if ticker not in wanted:
                continue
            try:
                with archive.open(member_name) as handle:
                    frame = pd.read_csv(handle)
            except Exception:
                continue
            columns = {str(column).upper(): column for column in frame.columns}
            if "DATE" not in columns or "CLOSE" not in columns:
                continue
            series = _as_series(frame, columns["DATE"], columns["CLOSE"])
            accepted, anchor_count, median, p90 = _scaled_identity_status(
                ticker, series, anchors
            )
            if accepted:
                price_return[ticker] = series
                total_return[ticker] = series
            manifest.append(
                {
                    "Ticker": ticker,
                    "Source": "Stooq-Kaggle-US",
                    "Rows": len(series),
                    "StartDate": series.index.min().date(),
                    "EndDate": series.index.max().date(),
                    "SplitCorrections": 0,
                    "AnchorCount": anchor_count,
                    "MedianAnchorError": median,
                    "P90AnchorError": p90,
                    "Accepted": accepted,
                    "SourceURL": STOOQ_URL,
                    "LicenseURL": STOOQ_INFO_URL,
                }
            )
    return price_return, total_return, manifest


def _cmc_monthly_history(tickers, anchors):
    """Load public monthly histories for four otherwise orphaned dot-com names."""
    price_return, total_return, manifest = {}, {}, []
    for ticker, slug in CMC_MONTHLY.items():
        if ticker not in tickers:
            continue
        url = f"https://companiesmarketcap.com/{slug}/stock-price-history/"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        arrays = re.findall(r"data\s*=\s*(\[[^;]+\])", response.text)
        rows = []
        for payload in arrays:
            try:
                candidate = json.loads(payload)
            except ValueError:
                continue
            if candidate and isinstance(candidate[0], dict) and {"d", "v"} <= set(
                candidate[0]
            ):
                rows = candidate
        if not rows:
            continue
        monthly = pd.Series(
            [pd.to_numeric(row["v"], errors="coerce") for row in rows],
            index=pd.to_datetime([row["d"] for row in rows], unit="s"),
            dtype=float,
        ).dropna()
        monthly = monthly[monthly > 0].sort_index()
        # The published chart series is already split-consistent (its absolute
        # level can differ from contemporaneous SEC raw prices by an exact
        # split multiple), so interpolate it directly.  Running a daily split
        # detector on monthly dot-com moves would mistake real crashes for
        # corporate actions.
        normalized = monthly
        corrections = 0
        daily_index = pd.date_range(
            normalized.index.min(), normalized.index.max(), freq="D"
        )
        daily = np.exp(
            np.log(normalized).reindex(daily_index).interpolate(method="time")
        )
        accepted, anchor_count, median, p90 = _scaled_identity_status(
            ticker, daily, anchors
        )
        if accepted:
            price_return[ticker] = daily
            total_return[ticker] = daily
        manifest.append(
            {
                "Ticker": ticker,
                "Source": "CompaniesMarketCap-monthly",
                "Rows": len(monthly),
                "StartDate": monthly.index.min().date(),
                "EndDate": monthly.index.max().date(),
                "SplitCorrections": corrections,
                "AnchorCount": anchor_count,
                "MedianAnchorError": median,
                "P90AnchorError": p90,
                "Accepted": accepted,
                "SourceURL": url,
                "LicenseURL": url,
            }
        )
    return price_return, total_return, manifest


def _wide(series_by_ticker):
    if not series_by_ticker:
        return pd.DataFrame()
    frame = pd.concat(series_by_ticker, axis=1).sort_index()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame[~frame.index.duplicated(keep="last")]


def build_free_history(refresh=False):
    tickers = required_tickers()
    anchors = _holding_anchors()
    wiki_price, wiki_total, wiki_manifest = _wiki_history(
        tickers, anchors, refresh=refresh
    )
    cmu_price, cmu_total, cmu_manifest = _cmu_history(
        tickers, anchors, refresh=refresh
    )
    stooq_price, stooq_total, stooq_manifest = _stooq_history(
        tickers, anchors, refresh=refresh
    )
    cmc_price, cmc_total, cmc_manifest = _cmc_monthly_history(tickers, anchors)

    # WIKI has explicit split/dividend fields and therefore wins overlaps.
    # Stooq is already split/dividend adjusted and wins CMU's heuristic split
    # normalization where both exist.
    combined_price = dict(cmu_price)
    combined_price.update(cmc_price)
    combined_price.update(stooq_price)
    combined_price.update(wiki_price)
    combined_total = dict(cmu_total)
    combined_total.update(cmc_total)
    combined_total.update(stooq_total)
    combined_total.update(wiki_total)
    price_frame = _wide(combined_price)
    total_frame = _wide(combined_total)
    os.makedirs(config.FREE_HISTORY_CACHE_DIR, exist_ok=True)
    price_frame.to_parquet(PRICE_RETURN_FILE)
    total_frame.to_parquet(TOTAL_RETURN_FILE)
    pd.DataFrame(wiki_manifest + stooq_manifest + cmu_manifest + cmc_manifest).to_csv(
        MANIFEST_FILE, index=False
    )
    with open(
        os.path.join(config.FREE_HISTORY_CACHE_DIR, "free_history_sources.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "priority": [
                    "Nasdaq-Data-Link-WIKI",
                    "Stooq-Kaggle-US",
                    "CompaniesMarketCap-monthly",
                    "CMU-DB-Historical-Stocks",
                ],
                "price_return": PRICE_RETURN_FILE,
                "total_return": TOTAL_RETURN_FILE,
                "ticker_count": len(price_frame.columns),
            },
            handle,
            indent=2,
        )
    print(
        f"Saved free archival history for {len(price_frame.columns)} tickers "
        f"to {config.FREE_HISTORY_CACHE_DIR}"
    )
    return price_frame, total_frame


def load_free_history(adjust_prices=True):
    path = TOTAL_RETURN_FILE if adjust_prices else PRICE_RETURN_FILE
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def merge_free_history_prices(df, adjust_prices=True):
    history = load_free_history(adjust_prices=adjust_prices)
    if history.empty:
        return df
    if df is None or df.empty:
        return history.copy()
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    union = df.index.union(history.index)
    df = df.reindex(union)
    merged = 0
    new_columns = {}
    for ticker in history.columns:
        archive = history[ticker].dropna()
        if archive.empty:
            continue
        if ticker not in df:
            new_columns[ticker] = archive.reindex(df.index)
            merged += 1
            continue
        existing = df[ticker].dropna()
        overlap = existing.index.intersection(archive.index)
        scale = 1.0
        if len(overlap) >= 5:
            ratios = existing.reindex(overlap) / archive.reindex(overlap)
            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratios.empty:
                scale = float(ratios.median())
        archive = archive * scale
        missing = df[ticker].isna()
        fill_index = archive.index.intersection(df.index[missing])
        if len(fill_index):
            df.loc[fill_index, ticker] = archive.reindex(fill_index)
            merged += 1
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    if merged:
        print(f"  [Free archival history] Added or gap-filled {merged} tickers")
    return df.sort_index()


def main():
    parser = argparse.ArgumentParser(
        description="Build identity-validated free historical NDX price caches."
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build_free_history(refresh=args.refresh)


if __name__ == "__main__":
    main()
