import pandas as pd
import json
import numpy as np
import datetime
import os
import re
import sys
import concurrent.futures as cf
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import calendar
import price_manager
from ndx_methodology import (
    NEW_METHODOLOGY_EFFECTIVE_DATE,
    apply_2026_ndx_capping,
    interpolate_new_entry_values,
    modified_market_cap_factor,
    uses_2026_methodology,
)
from official_index_data import get_official_constituents, LOCAL_MEMBERSHIP_FILES

try:
    import yfinance as yf
except Exception:
    yf = None

# Configuration
INPUT_CSV = config.COMPONENTS_FILE
MAPPING_FILE = os.path.join(config.ASSETS_DIR, "name_mapping.json")
OUTPUT_FILE = config.WEIGHTS_FILE
INDEX_EVENT_WEIGHTS_FILE = os.path.join(
    config.RESULTS_DIR,
    "nasdaq_index_event_weights.csv",
)
OFFICIAL_ALIGNMENT_FILE = os.path.join(config.RESULTS_DIR, "ndx_official_membership_alignment.csv")
FLOAT_FACTORS_FILE = os.path.join(config.ASSETS_DIR, "ndx_float_factors.csv")
PROXY_TICKER = "QQQ"
APPLY_OFFICIAL_NDX_FILTER = False
MAPPING_OVERRIDES = {
    "Xcel Energy, Inc.": "XEL",
}
# These are ticker-only renames of the same listed security.  Keep the dated
# constituent symbol for membership/audit purposes, but use the modern Yahoo
# symbol for its continuous price history.  Acquirers are deliberately absent:
# using MSFT for ATVI or AMD for XLNX before the acquisition would be a
# different security and would reintroduce survivorship bias.
PRICE_HISTORY_ALIASES = {
    "BBRY": "BB",
    "CTRP": "TCOM",
    "ERTS": "EA",
    "FB": "META",
    "HANS": "MNST",
    "IACI": "IAC",
    "PCLN": "BKNG",
    "RIMM": "BB",
    "SYMC": "GEN",
    "WLTW": "WTW",
}
OFFICIAL_OVERLAY_START_OFFSET_DAYS = 0
_SHARES_OUTSTANDING_CACHE = {}
FLOAT_FACTOR_WORKERS = 8


def apply_mapping_overrides(mapping):
    """Patch known stale mappings in the legacy name map."""
    mapping = dict(mapping)
    mapping.update(MAPPING_OVERRIDES)
    return mapping

def load_data(): # Load Components
    input_file = getattr(
        config,
        "HOLDINGS_SNAPSHOTS_FILE",
        config.COMPONENTS_FILE,
    )
    if not os.path.exists(input_file):
        input_file = config.COMPONENTS_FILE
    print(f"Loading {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Missing parsed holdings file: {input_file}. "
            "Run rebuild_all.py without --skip-download."
        )
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # FILTERING: Remove corrupt rows (e.g. where Company name is a date)
    # Valid company names don't start with digits (usually)
    # This matches the fix in ndx_scanner.py
    df = df[~df['Company'].str.match(r'^\d{4}-\d{2}-\d{2}', na=False)]
    
    # Ensure Value is positive
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Value'] > 0]
    if 'Shares' in df:
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    if 'Ticker' not in df:
        df['Ticker'] = ""
    df['Ticker'] = df['Ticker'].fillna("").astype(str)

    # Filter future dates (to avoid speculative/forward-looking data)
    today = pd.Timestamp.now().normalize()
    df = df[df['Date'] <= today]

    # Load Name Mapping
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    mapping = apply_mapping_overrides(mapping)
    return df, mapping

def _is_supported_price_ticker(ticker):
    return bool(isinstance(ticker, str) and re.match(r"^[A-Z][A-Z0-9.-]*$", ticker))


def _split_pipe_field(value):
    if not isinstance(value, str) or not value:
        return []
    return [item for item in value.split("|") if item]


def get_official_membership_tickers():
    tickers = set()
    for path in LOCAL_MEMBERSHIP_FILES.values():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["Tickers"])
        except Exception:
            continue
        for value in df["Tickers"].dropna():
            tickers.update(t for t in _split_pipe_field(value) if _is_supported_price_ticker(t))
    return tickers


def get_unique_tickers(mapping, holdings=None):
    tickers = set(mapping.values())
    tickers.update(PRICE_HISTORY_ALIASES.values())
    if holdings is not None and 'Ticker' in holdings:
        tickers.update(
            ticker
            for ticker in holdings['Ticker'].dropna().astype(str)
            if _is_supported_price_ticker(ticker)
        )
    tickers.update(get_official_membership_tickers())
    if PROXY_TICKER not in tickers:
        tickers.add(PROXY_TICKER)
    return sorted(tickers)


def _price_history_ticker(ticker, prices, start_date, end_date):
    """Return an identity-safe ticker with prices at both projection dates."""
    candidates = []
    alias = PRICE_HISTORY_ALIASES.get(ticker)
    if alias:
        candidates.append(alias)
    candidates.append(ticker)

    for candidate in candidates:
        if candidate not in prices.columns:
            continue
        if (
            pd.notna(prices.at[start_date, candidate])
            and pd.notna(prices.at[end_date, candidate])
        ):
            return candidate
    return PROXY_TICKER


def _price_source_label(price_ticker):
    if price_ticker == PROXY_TICKER:
        return "QQQ-proxy"
    return "daily-price"


def validate_reconstruction_quality(weights, alignment=None):
    """Fail closed when a rebuilt NDX history violates core accuracy invariants."""
    if weights is None or weights.empty:
        raise RuntimeError("Reconstruction quality gate failed: no weights generated")

    required = {"Date", "Ticker", "Weight", "PriceTicker", "AnchorDistanceDays"}
    missing_columns = sorted(required - set(weights.columns))
    if missing_columns:
        raise RuntimeError(
            "Reconstruction quality gate failed: missing columns "
            + ", ".join(missing_columns)
        )

    audit = weights.copy()
    audit["Date"] = pd.to_datetime(audit["Date"])
    issues = []
    if audit.duplicated(["Date", "Ticker"]).any():
        issues.append("duplicate Date/Ticker rows")
    if not np.isfinite(audit["Weight"]).all() or (audit["Weight"] <= 0).any():
        issues.append("weights must be finite and positive")

    event_sums = audit.groupby("Date")["Weight"].sum()
    max_sum_error = float((event_sums - 1.0).abs().max())
    if max_sum_error > 1e-8:
        issues.append(f"event weight-sum error {max_sum_error:.3g}")

    event_sizes = audit.groupby("Date")["Ticker"].nunique()
    min_event_size = int(event_sizes.min())
    if min_event_size < 90:
        issues.append(f"event contains only {min_event_size} securities")

    max_anchor_distance = int(
        pd.to_numeric(audit["AnchorDistanceDays"], errors="coerce").max()
    )
    if max_anchor_distance > 210:
        issues.append(f"SEC anchor is {max_anchor_distance} days from an event")

    proxy_weights = (
        audit.loc[
            audit["PriceTicker"].fillna("").astype(str).str.upper().eq(PROXY_TICKER)
        ]
        .groupby("Date")["Weight"]
        .sum()
    )
    modern_proxy = proxy_weights[proxy_weights.index >= pd.Timestamp("2019-01-01")]
    max_modern_proxy = float(modern_proxy.max()) if not modern_proxy.empty else 0.0
    if max_modern_proxy > 0.10:
        issues.append(f"post-2018 proxy weight is {max_modern_proxy:.1%}")

    if alignment is not None and not alignment.empty:
        official_rows = alignment[alignment["OfficialCount"] > 0]
        missing_official = int(official_rows["MissingOfficialCount"].sum())
        extra_official = int(official_rows["RemovedExtraCount"].sum())
        if missing_official:
            issues.append(f"{missing_official} official memberships are missing")
        if extra_official:
            issues.append(f"{extra_official} non-members remain")

    if issues:
        raise RuntimeError("Reconstruction quality gate failed: " + "; ".join(issues))

    return {
        "events": int(audit["Date"].nunique()),
        "min_event_size": min_event_size,
        "max_weight_sum_error": max_sum_error,
        "max_anchor_distance_days": max_anchor_distance,
        "max_post_2018_proxy_weight": max_modern_proxy,
    }


def get_official_name_map(index_symbol, trade_date):
    path = LOCAL_MEMBERSHIP_FILES.get(index_symbol)
    if not path or not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return {}

    if df.empty:
        return {}

    trade_ts = pd.Timestamp(trade_date).normalize()
    df = df.sort_values("Date")
    pos = df["Date"].searchsorted(trade_ts, side="right") - 1
    if pos < 0:
        return {}

    row = df.iloc[pos]
    tickers = _split_pipe_field(row.get("Tickers", ""))
    names = _split_pipe_field(row.get("Names", ""))
    return {
        ticker: names[i] if i < len(names) and names[i] else ticker
        for i, ticker in enumerate(tickers)
    }


def get_shares_outstanding(ticker):
    if ticker in _SHARES_OUTSTANDING_CACHE:
        return _SHARES_OUTSTANDING_CACHE[ticker]
    if yf is None:
        _SHARES_OUTSTANDING_CACHE[ticker] = None
        return None

    shares = None
    try:
        shares = yf.Ticker(ticker).fast_info.get("shares")
    except Exception:
        shares = None

    try:
        shares = float(shares) if shares and shares > 0 else None
    except Exception:
        shares = None

    _SHARES_OUTSTANDING_CACHE[ticker] = shares
    return shares


def _fetch_float_factor(ticker, reference_date):
    """Fetch a reproducible free-float factor for one reference date.

    Yahoo exposes current reported shares rather than a point-in-time float
    history. The first successful observation for each reference date is pinned
    in ``ndx_float_factors.csv`` so later rebuilds remain deterministic.
    """
    total_shares = None
    float_shares = None
    error = ""
    if yf is not None:
        try:
            info = yf.Ticker(ticker).info
            total_shares = info.get("sharesOutstanding")
            float_shares = info.get("floatShares")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    factor = modified_market_cap_factor(
        total_shares,
        float_shares,
        reference_date,
    )
    source = "yfinance_current_snapshot"
    if total_shares is None or float_shares is None:
        source = "missing_fallback_1.0"
    return {
        "ReferenceDate": pd.Timestamp(reference_date).date().isoformat(),
        "Ticker": ticker,
        "TotalShares": total_shares,
        "FloatShares": float_shares,
        "FloatFactor": factor,
        "Source": source,
        "RetrievedAtUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "Error": error,
    }


def get_float_factors(tickers, reference_date):
    """Load or fetch the 2026-methodology free-float factors for a date."""
    tickers = sorted(set(tickers))
    reference_date = pd.Timestamp(reference_date).normalize()
    columns = [
        "ReferenceDate",
        "Ticker",
        "TotalShares",
        "FloatShares",
        "FloatFactor",
        "Source",
        "RetrievedAtUTC",
        "Error",
    ]
    if os.path.exists(FLOAT_FACTORS_FILE):
        cache = pd.read_csv(FLOAT_FACTORS_FILE)
    else:
        cache = pd.DataFrame(columns=columns)

    refresh = os.environ.get("NDX_REFRESH_FLOAT_FACTORS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    reference_key = reference_date.date().isoformat()
    exact = cache[cache["ReferenceDate"].astype(str) == reference_key]
    available = set() if refresh else set(exact["Ticker"].astype(str))
    missing = [ticker for ticker in tickers if ticker not in available]

    # These public fields are current snapshots, not point-in-time history.
    # Reuse an already pinned observation across event reference dates instead
    # of making hundreds of calls that would return the same current values.
    if missing and not refresh and not cache.empty:
        reusable = cache[cache["Ticker"].astype(str).isin(missing)].copy()
        reusable = reusable.sort_values("ReferenceDate").drop_duplicates(
            "Ticker",
            keep="last",
        )
        if not reusable.empty:
            reusable["ReferenceDate"] = reference_key
            has_share_data = reusable["TotalShares"].notna() & reusable[
                "FloatShares"
            ].notna()
            reusable["Source"] = "reused_current_snapshot_proxy"
            reusable.loc[~has_share_data, "Source"] = "reused_missing_fallback_1.0"
            cache = pd.concat([cache, reusable], ignore_index=True)
            cache = cache.drop_duplicates(
                ["ReferenceDate", "Ticker"],
                keep="last",
            ).sort_values(["ReferenceDate", "Ticker"])
            cache.to_csv(FLOAT_FACTORS_FILE, index=False)
            exact = cache[cache["ReferenceDate"].astype(str) == reference_key]
            available = set(exact["Ticker"].astype(str))
            missing = [ticker for ticker in tickers if ticker not in available]

    if missing:
        print(
            f"  [2026 methodology] fetching free-float data for "
            f"{len(missing)} securities at {reference_key}"
        )
        fetched = []
        with cf.ThreadPoolExecutor(max_workers=FLOAT_FACTOR_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_float_factor, ticker, reference_date): ticker
                for ticker in missing
            }
            for future in cf.as_completed(futures):
                fetched.append(future.result())

        if refresh and not cache.empty:
            cache = cache[
                ~(
                    (cache["ReferenceDate"].astype(str) == reference_key)
                    & cache["Ticker"].astype(str).isin(missing)
                )
            ]
        cache = pd.concat([cache, pd.DataFrame(fetched)], ignore_index=True)
        cache = cache.drop_duplicates(
            ["ReferenceDate", "Ticker"],
            keep="last",
        ).sort_values(["ReferenceDate", "Ticker"])
        cache.to_csv(FLOAT_FACTORS_FILE, index=False)
        exact = cache[cache["ReferenceDate"].astype(str) == reference_key]

    factors = exact.set_index("Ticker")["FloatFactor"].astype(float).to_dict()
    low_float = {
        ticker: factor
        for ticker, factor in factors.items()
        if factor < 1.0 - 1e-9
    }
    if low_float:
        detail = ", ".join(
            f"{ticker}={factor:.4f}" for ticker, factor in sorted(low_float.items())
        )
        print(f"  [2026 methodology] low-float factors: {detail}")
    return {ticker: float(factors.get(ticker, 1.0)) for ticker in tickers}


def get_new_official_entries(effective_date):
    """Return constituents newly present at an index effective date."""
    effective_date = pd.Timestamp(effective_date).normalize()
    current = set(get_official_constituents("NDX", effective_date))
    previous = set(
        get_official_constituents("NDX", effective_date - pd.offsets.BDay(1))
    )
    return current - previous


def get_official_membership_change_dates(
    start_date=NEW_METHODOLOGY_EFFECTIVE_DATE,
):
    """Read official daily membership and return post-cutover change dates."""
    path = LOCAL_MEMBERSHIP_FILES.get("NDX")
    if not path or not os.path.exists(path):
        return []
    try:
        membership = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    except (OSError, ValueError):
        return []

    start_date = pd.Timestamp(start_date).normalize()
    previous = None
    event_dates = []
    for _, row in membership.iterrows():
        tickers = row.get("Tickers", "")
        current = set(tickers.split("|")) if isinstance(tickers, str) else set()
        event_date = pd.Timestamp(row["Date"]).normalize()
        if previous is not None and current != previous and event_date >= start_date:
            event_dates.append(event_date)
        previous = current
    return event_dates


def estimate_market_values(tickers, prices, date_q):
    values = {}
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        try:
            price = prices.at[date_q, ticker]
        except Exception:
            continue
        if pd.isna(price) or price <= 0:
            continue
        shares = get_shares_outstanding(ticker)
        if not shares:
            continue
        values[ticker] = float(price) * shares
    return values


def apply_recent_official_overlay(
    rows,
    official_tickers,
    official_names,
    prices,
    valuation_date,
    effective_date=None,
):
    """Use official membership to repair recent quarters after SEC filings go stale."""
    if not rows or not official_tickers:
        return rows

    effective_date = effective_date or valuation_date

    official_tickers = [t for t in official_tickers if _is_supported_price_ticker(t)]
    official_set = set(official_tickers)
    kept_rows = [
        row for row in rows
        if row.get("IsMapped") and row.get("Ticker") in official_set
    ]

    if not kept_rows:
        return rows

    present = {row["Ticker"] for row in kept_rows}
    missing = [ticker for ticker in official_tickers if ticker not in present]
    if not missing:
        return kept_rows

    scale_candidates = []
    existing_caps = estimate_market_values(
        [row["Ticker"] for row in sorted(kept_rows, key=lambda r: r["Value"], reverse=True)[:60]],
        prices,
        valuation_date,
    )
    for row in kept_rows:
        cap = existing_caps.get(row["Ticker"])
        if cap and cap > 0 and row["Value"] > 0:
            scale_candidates.append(row["Value"] / cap)

    scale = float(np.median(scale_candidates)) if scale_candidates else None
    fallback_value = float(np.median([row["Value"] for row in kept_rows if row["Value"] > 0]))

    missing_caps = estimate_market_values(missing, prices, valuation_date)
    added_rows = []
    for ticker in missing:
        value = None
        cap = missing_caps.get(ticker)
        if scale and cap:
            value = cap * scale
        if value is None or value <= 0:
            value = fallback_value
        added_rows.append({
            "Date": pd.Timestamp(effective_date).date(),
            "Ticker": ticker,
            "Name": official_names.get(ticker, ticker),
            "Value": value,
            "IsMapped": True,
        })

    print(
        f"  [Official overlay {pd.Timestamp(effective_date).date()}] "
        f"added {len(added_rows)} missing official tickers: {', '.join(missing)}"
    )
    return kept_rows + added_rows


def _nearest_price_date(prices, target_date):
    target_date = pd.Timestamp(target_date).normalize()
    position = prices.index.get_indexer([target_date], method="pad")[0]
    return None if position < 0 else prices.index[position]


def _project_snapshot(
    snapshot,
    target_date,
    prices,
    proxy_prices,
    mapping,
    *,
    value_source="anchor",
):
    """Project one observed holdings schedule with price-only returns."""
    target_price_date = _nearest_price_date(prices, target_date)
    if target_price_date is None:
        return []
    rows = []
    for _, holding in snapshot.iterrows():
        name = holding.get("Company", "")
        ticker = str(holding.get("Ticker") or "").strip()
        if not _is_supported_price_ticker(ticker):
            ticker = mapping.get(name, "")
        is_mapped = _is_supported_price_ticker(ticker)
        identity = ticker if is_mapped else f"UNMAPPED:{name}"
        try:
            observed_value = float(holding["Value"])
        except (TypeError, ValueError):
            continue
        report_date = pd.Timestamp(holding["Date"]).normalize()
        report_price_date = _nearest_price_date(prices, report_date)
        if report_price_date is None:
            continue

        price_ticker = _price_history_ticker(
            ticker,
            prices,
            report_price_date,
            target_price_date,
        )
        series = prices[price_ticker] if price_ticker in prices else proxy_prices
        projected_value = observed_value
        try:
            price_from = series.at[report_price_date]
            price_to = series.at[target_price_date]
            if pd.notna(price_from) and pd.notna(price_to) and price_from > 0:
                projected_value *= float(price_to / price_from)
        except (KeyError, TypeError, ValueError):
            price_ticker = PROXY_TICKER
            try:
                price_from = proxy_prices.at[report_price_date]
                price_to = proxy_prices.at[target_price_date]
                if pd.notna(price_from) and pd.notna(price_to) and price_from > 0:
                    projected_value = observed_value * float(price_to / price_from)
            except (KeyError, TypeError, ValueError):
                pass

        rows.append(
            {
                "Ticker": identity,
                "PriceTicker": price_ticker,
                "PriceSource": _price_source_label(price_ticker),
                "Name": name,
                "Value": projected_value,
                "IsMapped": bool(is_mapped),
                "ValueSource": value_source,
                "SourceReportDate": report_date,
                "SourceForm": holding.get("Form", "485BPOS"),
                "SourceConfidence": holding.get("Confidence", ""),
            }
        )
    return rows


def _choose_anchor_date(holdings, valuation_date, official_tickers):
    """Choose the closest high-coverage observed schedule, past or future."""
    valuation_date = pd.Timestamp(valuation_date).normalize()
    candidates = []
    official_set = set(official_tickers or [])
    for report_date, group in holdings.groupby("Date"):
        distance = abs((pd.Timestamp(report_date) - valuation_date).days)
        if distance > 550:
            continue
        mapped = set(
            ticker
            for ticker in group.get("Ticker", pd.Series(dtype=str)).astype(str)
            if _is_supported_price_ticker(ticker)
        )
        coverage = (
            len(mapped & official_set) / len(official_set) if official_set else 1.0
        )
        form = "NPORT-P" if (group.get("Form") == "NPORT-P").any() else "485BPOS"
        # Coverage is more important than a small date advantage. Quarterly
        # NPORT schedules within 45 days are effectively observed rebalances.
        score = distance + (1.0 - coverage) * 365.0
        if form == "NPORT-P" and distance <= 45:
            score -= 20.0
        candidates.append((score, distance, pd.Timestamp(report_date), coverage, form))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))


def _infer_pre_official_roster(holdings, target_date, anchor_snapshot):
    """Infer the missing 1999-2002 roster from adjacent observed schedules.

    Nasdaq's archived constituent roster begins in 2003.  Before then, use the
    mapped names that recur in nearby QQQ/Rydex/Morgan schedules.  Candidates
    observed on both sides of the target outrank one-sided names, which avoids
    treating every nearby addition or deletion as continuously eligible.
    """
    target_date = pd.Timestamp(target_date).normalize()
    window = holdings[
        (holdings["Date"] >= target_date - pd.Timedelta(days=370))
        & (holdings["Date"] <= target_date + pd.Timedelta(days=370))
        & holdings["IsMapped"].astype(bool)
    ].copy()
    window = window[
        window["Ticker"].astype(str).map(_is_supported_price_ticker)
    ]
    if window.empty:
        return [], {}

    observed_counts = window.groupby("Date")["Ticker"].nunique()
    target_count = min(100, max(90, int(observed_counts.max())))
    anchor_tickers = list(
        dict.fromkeys(
            anchor_snapshot.loc[
                anchor_snapshot["IsMapped"].astype(bool), "Ticker"
            ].astype(str)
        )
    )
    selected = list(anchor_tickers)
    selected_set = set(selected)
    candidates = []
    names = {}
    for ticker, group in window.groupby("Ticker"):
        dates = pd.to_datetime(group["Date"]).drop_duplicates()
        before = bool((dates <= target_date).any())
        after = bool((dates >= target_date).any())
        distance = int(min(abs((date - target_date).days) for date in dates))
        nearest_index = (pd.to_datetime(group["Date"]) - target_date).abs().argmin()
        nearest = group.iloc[int(nearest_index)]
        names[str(ticker)] = nearest.get("Company", str(ticker))
        candidates.append(
            (
                -(int(before) + int(after)),
                -len(dates),
                distance,
                str(ticker),
            )
        )
    for _, _, _, ticker in sorted(candidates):
        if ticker in selected_set:
            continue
        selected.append(ticker)
        selected_set.add(ticker)
        if len(selected) >= target_count:
            break
    return selected, names


def _fill_from_adjacent_snapshots(
    rows,
    holdings,
    official_tickers,
    official_names,
    target_date,
    prices,
    proxy_prices,
    mapping,
):
    """Back/forward-cast missing members from their nearest observed filing.

    QQQ fund size changes between filings.  The median value ratio of common
    constituents rescales the adjacent schedule before adding a missing member.
    This is substantially more defensible than applying today's share count to
    a historical event.
    """
    official_set = set(official_tickers or [])
    if not official_set:
        return rows, []
    rows = [row for row in rows if row.get("IsMapped") and row["Ticker"] in official_set]
    present = {row["Ticker"] for row in rows}
    missing = sorted(official_set - present)
    if not missing:
        return rows, []

    target_date = pd.Timestamp(target_date).normalize()
    anchor_values = {row["Ticker"]: row["Value"] for row in rows}
    projection_cache = {}
    added = []
    for ticker in missing:
        ticker_rows = holdings[
            (holdings["Ticker"].astype(str) == ticker)
            & ((holdings["Date"] - target_date).abs() <= pd.Timedelta(days=550))
        ]
        if ticker_rows.empty:
            continue
        source_date = min(
            ticker_rows["Date"].drop_duplicates(),
            key=lambda date: abs((pd.Timestamp(date) - target_date).days),
        )
        source_key = pd.Timestamp(source_date).normalize()
        if source_key not in projection_cache:
            source_snapshot = holdings[holdings["Date"] == source_key]
            projection_cache[source_key] = _project_snapshot(
                source_snapshot,
                target_date,
                prices,
                proxy_prices,
                mapping,
                value_source="adjacent-observation",
            )
        source_rows = projection_cache[source_key]
        source_values = {
            row["Ticker"]: row["Value"]
            for row in source_rows
            if row.get("IsMapped") and row["Ticker"] in official_set
        }
        source_value = source_values.get(ticker)
        if not source_value or source_value <= 0:
            continue
        common = set(anchor_values) & set(source_values)
        ratios = [
            anchor_values[common_ticker] / source_values[common_ticker]
            for common_ticker in common
            if source_values[common_ticker] > 0 and anchor_values[common_ticker] > 0
        ]
        scale = float(np.median(ratios)) if len(ratios) >= 5 else 1.0
        source_row = next(row for row in source_rows if row["Ticker"] == ticker)
        source_row = dict(source_row)
        source_row.update(
            {
                "Name": official_names.get(ticker, source_row.get("Name", ticker)),
                "Value": float(source_value) * scale,
                "ValueSource": "adjacent-observation-scaled",
            }
        )
        rows.append(source_row)
        anchor_values[ticker] = source_row["Value"]
        added.append(ticker)

    return rows, added



def _third_friday(year, month):
    """Return the 3rd Friday of the given month/year."""
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    fridays = [day for week in monthcal for day in week
               if day.weekday() == calendar.FRIDAY and day.month == month]
    return fridays[2] if len(fridays) >= 3 else fridays[-1]


def get_rebalance_dates(
    start_year=2000,
    end_year=None,
    include_membership_events=True,
):
    """
    Return effective/reference pairs for rebalances and official maintenance.

    Per Nasdaq methodology:
      - Reference date: Last trading day of Feb/May/Aug/Nov (weights determined here)
      - Effective date: First trading day following 3rd Friday of Mar/Jun/Sep/Dec
    """
    if end_year is None:
        end_year = pd.Timestamp.now().year

    pairs = []
    # Reference months map to effective months: Feb→Mar, May→Jun, Aug→Sep, Nov→Dec
    ref_eff_months = [(2, 3), (5, 6), (8, 9), (11, 12)]

    current_year = start_year
    while current_year <= end_year + 1:
        for ref_month, eff_month in ref_eff_months:
            if current_year > end_year and eff_month > 3:
                break

            # Effective date: first business day after 3rd Friday of eff_month
            date_3rd_fri = _third_friday(current_year, eff_month)
            effective_date = pd.Timestamp(date_3rd_fri) + pd.offsets.BDay(1)

            # Reference date: last business day of ref_month
            ref_year = current_year
            # BMonthEnd(0) from day 1 gives the last business day of that month
            reference_date = pd.Timestamp(year=ref_year, month=ref_month, day=1) + pd.offsets.BMonthEnd(0)

            pairs.append((effective_date, reference_date))
        current_year += 1

    # Filter to completed rebalance dates only. Backtests carry the latest
    # completed weights forward to the newest price date.
    limit = pd.Timestamp.now().normalize()
    pairs = [(e, r) for e, r in pairs if e <= limit]

    if include_membership_events:
        scheduled = {effective for effective, _ in pairs}
        for event_date in get_official_membership_change_dates():
            if event_date <= limit and event_date not in scheduled:
                pairs.append((event_date, event_date - pd.offsets.BDay(1)))
        pairs.sort(key=lambda pair: pair[0])

    # Edge case: data starts 2000-06-30, inject bootstrap date
    start_override_eff = pd.Timestamp("2000-06-30")
    start_override_ref = pd.Timestamp("2000-05-31") - pd.offsets.BDay(0)
    if not any(e == start_override_eff for e, _ in pairs):
        pairs.append((start_override_eff, start_override_ref))
        pairs.sort(key=lambda x: x[0])

    return pairs

def reconstruct():
    df, mapping = load_data()
    tickers = get_unique_tickers(mapping, df)
    
    # Use Centralized Price Manager (handles caching)
    # Weight construction is a price-return operation. Dividend-adjusted
    # prices incorrectly make dividend payers gain index weight between SEC
    # report dates and Nasdaq events.
    prices = price_manager.get_price_data(
        tickers,
        start_date="1999-01-01",
        adjust_prices=False,
    )
    
    if prices is None or prices.empty:
        print("No price data. Aborting.")
        return

    # Ensure index is Datetime
    prices.index = pd.to_datetime(prices.index)
    
    # Resample prices to daily if needed (fill fwd) to handle weekend quarters
    prices = prices.ffill()
    
    rebalance_pairs = get_rebalance_dates()
    scheduled_effective_dates = {
        pd.Timestamp(effective).normalize()
        for effective, _ in get_rebalance_dates(include_membership_events=False)
    }
    reference_date_by_effective = {
        pd.Timestamp(effective).normalize(): pd.Timestamp(reference).normalize()
        for effective, reference in rebalance_pairs
    }
    final_rows = []

    # Proxy Returns
    if PROXY_TICKER in prices.columns:
        proxy_prices = prices[PROXY_TICKER]
    else:
        proxy_prices = pd.Series(1.0, index=prices.index)

    event_count = sum(
        pd.Timestamp(effective).normalize() not in scheduled_effective_dates
        for effective, _ in rebalance_pairs
    )
    print(
        f"Reconstructing weights for {len(rebalance_pairs)} index events "
        f"({event_count} intra-quarter membership changes)..."
    )
    for effective_date, reference_date in rebalance_pairs:
        q_date = effective_date
        # Preserve the historical reconstruction, but apply the official
        # prior-month reference date to rebalances governed by the May 2026
        # methodology. Membership still becomes effective on q_date.
        valuation_date = (
            reference_date if uses_2026_methodology(effective_date) else effective_date
        )

        official_tickers = get_official_constituents("NDX", q_date)
        official_names = get_official_name_map("NDX", q_date)
        anchor = _choose_anchor_date(df, valuation_date, official_tickers)
        if anchor is None:
            print(f"  [Data gap {q_date.date()}] no SEC snapshot within 550 days")
            continue
        _, anchor_distance, anchor_date, anchor_coverage, anchor_form = anchor
        anchor_snapshot = df[df["Date"] == anchor_date]
        inferred_roster = False
        if not official_tickers:
            official_tickers, official_names = _infer_pre_official_roster(
                df, valuation_date, anchor_snapshot
            )
            inferred_roster = bool(official_tickers)
        quarter_rows = _project_snapshot(
            anchor_snapshot,
            valuation_date,
            prices,
            proxy_prices,
            mapping,
        )
        adjacent_added = []
        overlay_added = 0
        if official_tickers:
            quarter_rows, adjacent_added = _fill_from_adjacent_snapshots(
                quarter_rows,
                df,
                official_tickers,
                official_names,
                valuation_date,
                prices,
                proxy_prices,
                mapping,
            )
            present = {
                row["Ticker"]
                for row in quarter_rows
                if row.get("IsMapped") and row["Ticker"] in set(official_tickers)
            }
            remaining = set(official_tickers) - present
            if remaining and not inferred_roster:
                date_q = _nearest_price_date(prices, valuation_date)
                before = len(quarter_rows)
                quarter_rows = apply_recent_official_overlay(
                    quarter_rows,
                    official_tickers,
                    official_names,
                    prices,
                    date_q,
                    q_date,
                )
                overlay_added = max(0, len(quarter_rows) - before)

        present = {
            row["Ticker"]
            for row in quarter_rows
            if row.get("IsMapped")
        }
        observed_near_event = (
            anchor_distance <= 45
            and not adjacent_added
            and overlay_added == 0
            and (not official_tickers or set(official_tickers) <= present)
        )
        for row in quarter_rows:
            row.update(
                {
                    "Date": q_date.date(),
                    "AnchorDate": anchor_date,
                    "AnchorDistanceDays": anchor_distance,
                    "AnchorCoverage": anchor_coverage,
                    "AnchorForm": anchor_form,
                    "ObservedNearEvent": observed_near_event,
                }
            )
            row.setdefault(
                "PriceTicker",
                row["Ticker"]
                if row.get("IsMapped") and row["Ticker"] in prices.columns
                else PROXY_TICKER,
            )
            row.setdefault("PriceSource", _price_source_label(row["PriceTicker"]))
            row.setdefault("ValueSource", "current-market-cap-fallback")
            row.setdefault("SourceReportDate", anchor_date)
            row.setdefault("SourceForm", anchor_form)
            row.setdefault("SourceConfidence", "")

        if adjacent_added:
            print(
                f"  [Adjacent observations {q_date.date()}] filled "
                f"{len(adjacent_added)} members: {', '.join(adjacent_added)}"
            )
        final_rows.extend(quarter_rows)

    # Convert to DataFrame
    res_df = pd.DataFrame(final_rows)
    
    # Aggregate duplicate tickers (e.g. merger variations or same ticker mappings)
    # Sum 'Value' for same ('Date', 'Ticker')
    # Keep metadata from first occurrence
    if not res_df.empty:
        res_df['Date'] = pd.to_datetime(res_df['Date'])
        res_df = res_df.groupby(['Date', 'Ticker'], as_index=False).agg({
            'Value': 'sum',
            'Name': 'first',
            'IsMapped': 'first',
            'PriceTicker': 'first',
            'PriceSource': 'first',
            'ValueSource': 'first',
            'SourceReportDate': 'first',
            'SourceForm': 'first',
            'SourceConfidence': 'first',
            'AnchorDate': 'first',
            'AnchorDistanceDays': 'first',
            'AnchorCoverage': 'first',
            'AnchorForm': 'first',
            'ObservedNearEvent': 'all',
        })

        print("Generating official NDX membership alignment audit...")
        alignment_rows = []

        for dt, grp in res_df.groupby('Date', sort=True):
            grp = grp.copy()
            mapped_tickers = set(grp.loc[grp['IsMapped'].astype(bool), 'Ticker'])
            official_tickers = get_official_constituents("NDX", dt)

            if not official_tickers:
                alignment_rows.append({
                    "Date": dt.date(),
                    "OfficialCount": 0,
                    "KeptMappedCount": len(mapped_tickers),
                    "RemovedExtraCount": 0,
                    "MissingOfficialCount": 0,
                    "RemovedExtras": "",
                    "MissingOfficial": "",
                })
                continue

            official_set = set(official_tickers)
            kept_grp = grp[
                grp['IsMapped'].astype(bool) & grp['Ticker'].isin(official_set)
            ].copy()

            removed_extras = sorted(mapped_tickers - official_set)
            missing_official = sorted(official_set - set(kept_grp['Ticker']))

            alignment_rows.append({
                "Date": dt.date(),
                "OfficialCount": len(official_set),
                "KeptMappedCount": len(kept_grp),
                "RemovedExtraCount": len(removed_extras),
                "MissingOfficialCount": len(missing_official),
                "RemovedExtras": "|".join(removed_extras),
                "MissingOfficial": "|".join(missing_official),
            })

        alignment_df = pd.DataFrame(alignment_rows)
        alignment_df.to_csv(OFFICIAL_ALIGNMENT_FILE, index=False)
        print(f"Saved official membership alignment audit to {OFFICIAL_ALIGNMENT_FILE}")

        if APPLY_OFFICIAL_NDX_FILTER:
            print("Official NDX membership filter is enabled.")
            filtered_groups = []

            for dt, grp in res_df.groupby('Date', sort=True):
                official_tickers = get_official_constituents("NDX", dt)
                if not official_tickers:
                    filtered_groups.append(grp)
                    continue

                official_set = set(official_tickers)
                kept_grp = grp[
                    grp['IsMapped'].astype(bool) & grp['Ticker'].isin(official_set)
                ].copy()

                if kept_grp.empty:
                    print(f"Warning: {dt.date()} official NDX filter yielded no mapped members; keeping reconstructed universe.")
                    filtered_groups.append(grp)
                else:
                    filtered_groups.append(kept_grp)

            res_df = pd.concat(filtered_groups, ignore_index=True)

        res_df["ReferenceDate"] = res_df["Date"].map(reference_date_by_effective)
        res_df["EventType"] = res_df["Date"].map(
            lambda date: (
                "ScheduledRebalance"
                if pd.Timestamp(date).normalize() in scheduled_effective_dates
                else "MembershipChange"
            )
        )
        res_df["FloatFactor"] = 1.0
        res_df["IsInterpolatedEntry"] = False
        res_df["MethodologyVersion"] = res_df["Date"].map(
            lambda date: (
                "ndx-2026-observed"
                if uses_2026_methodology(date)
                else "observed-sec-holdings"
            )
        )

        for dt, indices in res_df.groupby("Date", sort=True).groups.items():
            if not uses_2026_methodology(dt):
                continue
            event_group = res_df.loc[indices]
            if (
                event_group["EventType"].iloc[0] != "ScheduledRebalance"
                or bool(event_group["ObservedNearEvent"].all())
            ):
                continue

            reference_date = reference_date_by_effective.get(
                pd.Timestamp(dt).normalize(),
                pd.Timestamp(dt).normalize(),
            )
            group = res_df.loc[indices]
            tickers = group.loc[group["IsMapped"].astype(bool), "Ticker"].tolist()
            float_factors = get_float_factors(tickers, reference_date)
            factors = group["Ticker"].map(float_factors).fillna(1.0).astype(float)

            res_df.loc[indices, "ReferenceDate"] = reference_date
            res_df.loc[indices, "FloatFactor"] = factors.values
            res_df.loc[indices, "Value"] = (
                group["Value"].astype(float).values * factors.values
            )
            res_df.loc[indices, "MethodologyVersion"] = "ndx-2026-05-01"

            new_entries = get_new_official_entries(dt)
            group_after_float = res_df.loc[indices].set_index("Ticker")
            interpolated_values, interpolated = interpolate_new_entry_values(
                group_after_float["Value"],
                new_entries,
            )
            for ticker, value in interpolated_values.items():
                ticker_rows = indices[group_after_float.index == ticker]
                res_df.loc[ticker_rows, "Value"] = float(value)
            if interpolated:
                mask = res_df.index.isin(indices) & res_df["Ticker"].isin(interpolated)
                res_df.loc[mask, "IsInterpolatedEntry"] = True
                print(
                    f"  [2026 methodology {pd.Timestamp(dt).date()}] "
                    f"interpolated new entries: {', '.join(sorted(interpolated))}"
                )
    
    # Calculate Weights per Date
    # GroupBy Date sum
    print("Calculating final weights...")
    sums = res_df.groupby('Date')['Value'].transform('sum')
    res_df['Weight'] = res_df['Value'] / sums
    
    # ---------------------------------------------------------
    # APPLY NDX CAPPING RULES (Methodology_NDX.pdf)
    # To ensure the "Reconstructed" weights represent the Index at Rebalance.
    #
    # Quarterly: Trigger at 24%/48%, cap at 20%/40% (iterative)
    # Annual (Dec): Trigger at 15%/top5>40%, cap at 14%/38.5%
    #               + outside-top-5 capped at min(4.4%, 5th largest)
    # ---------------------------------------------------------
    
    def apply_legacy_ndx_capping(group, is_annual=False):
        """Preserve the repo's pre-May-2026 capping behavior.

        Quarterly constraints (checked FIRST — if both satisfied, no adjustment):
          - No company weight may exceed 24%
          - Aggregate weight of companies > 4.5% may not exceed 48%

        Quarterly Stage 1: If any weight > 24%, cap all at 20%
        Quarterly Stage 2: If aggregate(>4.5%) > 48%, set aggregate to 40%

        Annual Stage 1: If any weight > 15%, cap all at 14%
        Annual Stage 2: If top-5 aggregate > 40%, set to 38.5%; cap outside-top-5
                        at min(4.4%, weight of 5th largest)
        """
        w = group['Weight'].values.copy()
        tickers = group['Ticker'].values.copy()
        w_series = pd.Series(w, index=tickers)

        # Normalize
        if w_series.sum() > 0:
            w_series = w_series / w_series.sum()

        if not is_annual:
            # --- QUARTERLY RULES ---
            # "If neither constraint is violated, no further adjustments are made"
            if w_series.max() <= 0.24 and w_series[w_series > 0.045].sum() <= 0.48:
                return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

            for _ in range(20):
                w_series = w_series / w_series.sum()

                # Stage 1: Trigger at 24%, cap at 20%
                if w_series.max() > 0.24:
                    over = w_series[w_series > 0.20]
                    if not over.empty:
                        surplus = (over - 0.20).sum()
                        w_series[w_series > 0.20] = 0.20
                        others = w_series[w_series < 0.20]
                        if not others.empty:
                            w_series[others.index] = others + (surplus * others / others.sum())

                # Stage 2: Aggregate(>4.5%) must not exceed 48%, target 40%
                above = w_series[w_series > 0.045]
                if above.sum() > 0.48:
                    target_agg = 0.40
                    scale = target_agg / above.sum()
                    scaled_above = above * scale
                    surplus = above.sum() - target_agg

                    # Some stocks may have dropped below 4.5% after scaling
                    w_series[above.index] = scaled_above
                    below = w_series[w_series <= 0.045]
                    if not below.empty:
                        w_series[below.index] = below + (surplus * below / below.sum())

                # Check convergence
                if w_series.max() <= 0.2401 and w_series[w_series > 0.045].sum() <= 0.4801:
                    break

        else:
            # --- ANNUAL RULES (December) ---
            # Check if adjustments needed at all
            w_sorted = w_series.sort_values(ascending=False)
            needs_stage1 = w_series.max() > 0.15
            needs_stage2 = w_sorted.iloc[:5].sum() > 0.40 if len(w_sorted) >= 5 else False

            if not needs_stage1 and not needs_stage2:
                return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

            for _ in range(20):
                w_series = w_series / w_series.sum()

                # Stage 1: Trigger at 15%, cap at 14%
                if w_series.max() > 0.15:
                    over = w_series[w_series > 0.14]
                    if not over.empty:
                        surplus = (over - 0.14).sum()
                        w_series[w_series > 0.14] = 0.14
                        others = w_series[w_series <= 0.14]
                        if not others.empty:
                            w_series[others.index] = others + (surplus * others / others.sum())

                # Stage 2: Top-5 aggregate > 40% → set to 38.5%
                w_sorted = w_series.sort_values(ascending=False)
                if len(w_sorted) >= 5:
                    top5_tickers = w_sorted.iloc[:5].index
                    top5_sum = w_series[top5_tickers].sum()

                    if top5_sum > 0.40:
                        scale = 0.385 / top5_sum
                        w_series[top5_tickers] = w_series[top5_tickers] * scale
                        surplus = top5_sum - 0.385

                        # Distribute surplus to non-top-5
                        others_idx = w_series.index.difference(top5_tickers)
                        others = w_series[others_idx]
                        if not others.empty:
                            w_series[others_idx] = others + (surplus * others / others.sum())

                        # Cap outside-top-5 at min(4.4%, weight of 5th largest)
                        # Use the ORIGINAL top5_tickers (pre-redistribution sort)
                        # to determine "outside" — redistribution can push others
                        # above original top-5 members, but that doesn't reclassify them.
                        fifth_val = w_series[top5_tickers].min()  # 5th largest of original top-5
                        cap_val = min(0.044, fifth_val)

                        outside_idx = w_series.index.difference(top5_tickers)
                        outside_over = w_series[outside_idx][w_series[outside_idx] > cap_val]

                        if not outside_over.empty:
                            surplus2 = (outside_over - cap_val).sum()
                            w_series[outside_over.index] = cap_val
                            # Redistribute to uncapped outside-top-5 stocks
                            uncapped = w_series[outside_idx][w_series[outside_idx] < cap_val]
                            if not uncapped.empty:
                                w_series[uncapped.index] = uncapped + (surplus2 * uncapped / uncapped.sum())

                # Check convergence
                w_check = w_series.sort_values(ascending=False)
                stage1_ok = w_check.iloc[0] <= 0.1501
                stage2_ok = (len(w_check) < 5) or (w_check.iloc[:5].sum() <= 0.4001)
                # Outside-top-5 cap: use actual min(4.4%, 5th-largest) + tolerance
                if len(w_check) >= 5:
                    fifth = w_check.iloc[4]
                    outside_cap = min(0.044, fifth) + 0.001
                    outside_ok = (len(w_check) < 6) or (w_check.iloc[5:] <= outside_cap).all()
                else:
                    outside_ok = True
                if stage1_ok and stage2_ok and outside_ok:
                    break

        return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

    print("Applying methodology caps only to synthetic scheduled rebalances...")
    
    # Apply per Date
    # Create temp df to merge back
    capped_list = []
    
    for dt, grp in res_df.groupby('Date'):
        # Determine if Annual Reconstitution (December)
        # dt is Timestamp
        is_annual = (dt.month == 12)
        
        event_type = str(grp["EventType"].iloc[0])
        observed_near_event = bool(grp["ObservedNearEvent"].all())
        needs_synthetic_cap = (
            uses_2026_methodology(dt)
            and event_type == "ScheduledRebalance"
            and not observed_near_event
        )

        if needs_synthetic_cap:
            capped_grp = apply_2026_ndx_capping(grp, is_annual=is_annual)
        else:
            # SEC fund holdings already embody the index caps. Reapplying a
            # second cap was measurably worsening tracking error. Historical
            # synthetic events retain the closest observed/back-cast weights;
            # the explicitly implemented 2026 rules remain active when there
            # is no nearby observed post-rebalance schedule.
            capped_grp = pd.DataFrame(
                {
                    "Ticker": grp["Ticker"].values,
                    "CappedWeight": grp["Weight"].values,
                }
            )
        capped_grp["CappingApplied"] = needs_synthetic_cap
        capped_grp['Date'] = dt
        capped_list.append(capped_grp)
        
    capped_df = pd.concat(capped_list)
    
    # Merge back to update weights
    # Note: 'res_df' has multiple rows, 'capped_df' has adjusted weights
    res_df = res_df.merge(capped_df, on=['Date', 'Ticker'], how='left')
    
    # Overwrite Weight
    res_df['Weight'] = res_df['CappedWeight']
    res_df.drop(columns=['CappedWeight'], inplace=True)
    
    # Keep the full parent-index event history for NDX validation, while the
    # strategy engines retain their established scheduled-rebalance cadence.
    res_df = res_df.sort_values(["Date", "Weight"], ascending=[True, False])
    quality = validate_reconstruction_quality(res_df, alignment_df)
    print(
        "Reconstruction quality gate passed: "
        f"{quality['events']} events, min {quality['min_event_size']} members, "
        f"max anchor {quality['max_anchor_distance_days']} days, "
        f"post-2018 proxy <= {quality['max_post_2018_proxy_weight']:.1%}"
    )
    res_df.to_csv(INDEX_EVENT_WEIGHTS_FILE, index=False)
    strategy_start = pd.Timestamp(getattr(config, "STRATEGY_START_DATE", "2000-06-30"))
    scheduled = res_df[
        (res_df["EventType"] == "ScheduledRebalance")
        & (res_df["Date"] >= strategy_start)
    ].copy()
    scheduled.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved full NDX event weights to {INDEX_EVENT_WEIGHTS_FILE}")
    print(f"Saved scheduled strategy weights to {OUTPUT_FILE}")

if __name__ == "__main__":
    reconstruct()
