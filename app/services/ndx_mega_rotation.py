"""Dynamic Nasdaq constituent rotation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


DYNAMIC_NDXMEGA_PREFIX = "NDXMEGA_TOP"
DYNAMIC_NDXMEGA_ANN_RE = re.compile(r"^NDXMEGA_TOP(?P<top_n>\d+)_ANN$", re.IGNORECASE)
DYNAMIC_NDX_PREFIX = "NDX_TOP"
DYNAMIC_NDX_ANN_RE = re.compile(r"^NDX_TOP(?P<top_n>\d+)_ANN$", re.IGNORECASE)
WEIGHTING_QUERY_KEYS = ("W", "WT", "WEIGHT", "WEIGHTING")
WEIGHTING_EQUAL = "EQUAL"
WEIGHTING_CAP = "CAP"
WEIGHTING_RANK = "RANK"
WEIGHTING_INVERSE_RANK = "INV_RANK"
WEIGHTING_ALIASES = {
    "": WEIGHTING_EQUAL,
    "EQUAL": WEIGHTING_EQUAL,
    "EQ": WEIGHTING_EQUAL,
    "EQUAL_WEIGHT": WEIGHTING_EQUAL,
    "EQUAL_WEIGHTED": WEIGHTING_EQUAL,
    "CAP": WEIGHTING_CAP,
    "CAP_WEIGHT": WEIGHTING_CAP,
    "CAP_WEIGHTED": WEIGHTING_CAP,
    "CAPWEIGHT": WEIGHTING_CAP,
    "CAPWEIGHTED": WEIGHTING_CAP,
    "WEIGHT": WEIGHTING_CAP,
    "WEIGHTED": WEIGHTING_CAP,
    "RANK": WEIGHTING_RANK,
    "RANK_WEIGHT": WEIGHTING_RANK,
    "RANK_WEIGHTED": WEIGHTING_RANK,
    "INV_RANK": WEIGHTING_INVERSE_RANK,
    "INVERSE_RANK": WEIGHTING_INVERSE_RANK,
    "INVERSE_RANK_WEIGHT": WEIGHTING_INVERSE_RANK,
    "INVERSE_RANK_WEIGHTED": WEIGHTING_INVERSE_RANK,
    "INVRANK": WEIGHTING_INVERSE_RANK,
}

INDIVIDUAL_2X_EXPENSE_RATIOS = {
    "AAPL": 0.96,
    "MSFT": 0.98,
    "AVGO": 1.00,
    "AMZN": 0.99,
    "META": 1.02,
    "NVDA": 0.92,
    "GOOG": 0.96,
    "GOOGL": 0.96,
    "TSLA": 0.83,
}
DEFAULT_INDIVIDUAL_2X_EXPENSE_RATIO = 0.92

# Keep this aligned with data/ndx_simulation/src/config.py without importing that
# script-style module into the app package.
DUAL_CLASS_GROUPS = {
    "GOOG": "GOOGL",
    "FOXA": "FOX",
    "LBTYA": "LBTYK",
    "LBTYK": "LBTYK",
    "DISCA": "DISCK",
    "DISCK": "DISCK",
    "NWSA": "NWS",
    "NWS": "NWS",
}
SYMBOL_HISTORY_ALIASES = {
    "BBRY": "RIMM",
    "FB": "META",
    "GEN": "SYMC",
}


@dataclass
class DynamicAllocationPlan:
    engine_allocation: dict[str, float]
    dynamic_schedule: dict[pd.Timestamp, dict[str, float]] | None
    maint_pcts: dict[str, float]
    pm_maint_pcts: dict[str, float] | None
    universe_tickers: list[str]
    has_dynamic: bool = False
    notes: list[str] | None = None


@dataclass
class _DynamicSpec:
    ticker: str
    base: str
    query_suffix: str
    source: str
    top_n: int
    weighting: str
    weight: float
    maint_pct: float | None
    pm_maint_pct: float | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _constituents_path() -> Path:
    return _repo_root() / "data" / "ndx_simulation" / "data" / "results" / "ndx_mega_constituents.csv"


def _ndx_weights_path() -> Path:
    return _repo_root() / "data" / "ndx_simulation" / "data" / "results" / "nasdaq_quarterly_weights.csv"


def _ndx_official_membership_path() -> Path:
    return (
        _repo_root()
        / "data"
        / "ndx_simulation"
        / "data"
        / "assets"
        / "official_membership"
        / "ndx_official_membership_daily.csv"
    )


def split_ticker_query(ticker: str) -> tuple[str, str]:
    raw = str(ticker).strip()
    if "?" not in raw:
        return raw.upper(), ""
    base, query = raw.split("?", 1)
    return base.strip().upper(), f"?{query}"


def _query_params(query_suffix: str) -> dict[str, str]:
    query = query_suffix[1:] if query_suffix.startswith("?") else query_suffix
    params: dict[str, str] = {}
    for pair in query.split("&"):
        if not pair:
            continue
        if "=" not in pair:
            params[pair] = ""
            continue
        key, value = pair.split("=", 1)
        params[key] = value
    return params


def _format_query(params: dict[str, str]) -> str:
    if not params:
        return ""
    return "?" + "&".join(f"{key}={value}" for key, value in params.items())


def _query_for_underlying(ticker: str, query_suffix: str) -> str:
    if not query_suffix:
        return ""

    params = _query_params(query_suffix)
    for key in WEIGHTING_QUERY_KEYS:
        params.pop(key, None)
    expense_value = params.get("E")
    if expense_value and expense_value.upper() in {"AUTO", "INDIV", "INDIVIDUAL"}:
        expense_ratio = INDIVIDUAL_2X_EXPENSE_RATIOS.get(
            ticker.upper(),
            DEFAULT_INDIVIDUAL_2X_EXPENSE_RATIO,
        )
        params["E"] = f"{expense_ratio:.2f}"
    return _format_query(params)


def _weighting_from_query(query_suffix: str) -> str:
    params = _query_params(query_suffix)
    raw_value = ""
    for key in WEIGHTING_QUERY_KEYS:
        if key in params:
            raw_value = str(params[key]).strip().upper()
            break
    normalized = raw_value.replace("-", "_").replace(" ", "_")
    weighting = WEIGHTING_ALIASES.get(normalized)
    if weighting is None:
        valid = ", ".join(sorted({value for value in WEIGHTING_ALIASES.values()}))
        raise ValueError(f"Unsupported dynamic Nasdaq weighting '{raw_value}'. Valid values: {valid}")
    return weighting


def parse_dynamic_rotation_ticker(ticker: str) -> tuple[str, int] | None:
    base, _ = split_ticker_query(ticker)
    for source, pattern in (
        ("NDXMEGA", DYNAMIC_NDXMEGA_ANN_RE),
        ("NDX", DYNAMIC_NDX_ANN_RE),
    ):
        match = pattern.match(base)
        if not match:
            continue
        top_n = int(match.group("top_n"))
        return (source, top_n) if top_n > 0 else None
    return None


def parse_dynamic_ndxmega_ticker(ticker: str) -> int | None:
    parsed = parse_dynamic_rotation_ticker(ticker)
    if parsed is None or parsed[0] != "NDXMEGA":
        return None
    return parsed[1]


def is_dynamic_ndxmega_ticker(ticker: str) -> bool:
    return parse_dynamic_ndxmega_ticker(ticker) is not None


def is_dynamic_rotation_ticker(ticker: str) -> bool:
    return parse_dynamic_rotation_ticker(ticker) is not None


@lru_cache(maxsize=1)
def load_ndxmega_constituents() -> pd.DataFrame:
    path = _constituents_path()
    if not path.exists():
        raise FileNotFoundError(f"NDXMEGA constituents file not found: {path}")

    df = pd.read_csv(path)
    required = {"Date", "Type", "Tickers", "Weights"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NDXMEGA constituents file missing columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Type"] = df["Type"].astype(str)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_ndx_quarterly_weights() -> pd.DataFrame:
    path = _ndx_weights_path()
    if not path.exists():
        raise FileNotFoundError(f"NDX quarterly weights file not found: {path}")

    df = pd.read_csv(path)
    required = {"Date", "Ticker", "IsMapped", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NDX quarterly weights file missing columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    if "PriceTicker" not in df:
        df["PriceTicker"] = df["Ticker"]
    else:
        df["PriceTicker"] = (
            df["PriceTicker"]
            .where(df["PriceTicker"].notna(), df["Ticker"])
            .astype(str)
            .str.strip()
            .str.upper()
        )
    df = df.sort_values(["Date", "Weight"], ascending=[True, False]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_ndx_official_membership() -> pd.DataFrame | None:
    path = _ndx_official_membership_path()
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"Date", "Tickers"}
    missing = required - set(df.columns)
    if missing:
        return None

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _official_ndx_company_set_asof(trade_date) -> set[str] | None:
    df = load_ndx_official_membership()
    if df is None or df.empty:
        return None

    trade_ts = pd.Timestamp(trade_date).normalize()
    pos = df["Date"].searchsorted(trade_ts, side="right") - 1
    if pos < 0:
        return None

    row = df.iloc[pos]
    tickers = _parse_pipe_list(row.get("Tickers"))
    if not tickers:
        return None
    return {_canonical_company(ticker) for ticker in tickers}


def _annual_rows_for_range(start_date, end_date) -> pd.DataFrame:
    df = load_ndxmega_constituents()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    annual = df[df["Type"].str.upper().eq("RECON")].copy()
    selected_indices: list[int] = []

    prior_annual = annual[annual["Date"] <= start_ts]
    if not prior_annual.empty:
        selected_indices.append(int(prior_annual.index[-1]))
    else:
        prior_any = df[df["Date"] <= start_ts]
        if not prior_any.empty:
            selected_indices.append(int(prior_any.index[-1]))
        elif not df.empty:
            selected_indices.append(int(df.index[0]))

    in_range = annual[(annual["Date"] > start_ts) & (annual["Date"] <= end_ts)]
    selected_indices.extend(int(idx) for idx in in_range.index)

    selected_indices = list(dict.fromkeys(selected_indices))
    return df.loc[selected_indices].sort_values("Date") if selected_indices else pd.DataFrame()


def _ndx_top_rows_for_range(start_date, end_date) -> pd.DataFrame:
    df = load_ndx_quarterly_weights()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    annual_dates = sorted(df.loc[df["Date"].dt.month.eq(12), "Date"].drop_duplicates())
    selected_dates: list[pd.Timestamp] = []

    prior_dates = [dt for dt in annual_dates if pd.Timestamp(f"{dt.year + 1}-01-01") <= start_ts]
    if prior_dates:
        selected_dates.append(prior_dates[-1])
    elif annual_dates:
        selected_dates.append(annual_dates[0])

    selected_dates.extend(
        dt for dt in annual_dates
        if start_ts < pd.Timestamp(f"{dt.year + 1}-01-01") <= end_ts
    )
    selected_dates = list(dict.fromkeys(selected_dates))

    rows = []
    for dt in selected_dates:
        q_weights = df[df["Date"].eq(dt)].copy()
        official_companies = _official_ndx_company_set_asof(dt)
        if official_companies:
            q_weights = q_weights[
                q_weights["Ticker"].map(_canonical_company).isin(official_companies)
            ].copy()
        selected_tickers = _top_ndx_company_tickers(q_weights, top_n=1000)
        weights = []
        if selected_tickers:
            company_weights = _ndx_company_weights(q_weights)
            for ticker in selected_tickers:
                weights.append(company_weights.get(_canonical_company(ticker), 0.0))
            price_map = (
                q_weights.drop_duplicates("Ticker", keep="first")
                .set_index("Ticker")["PriceTicker"]
            )
            selected_tickers = [
                (
                    price_map.get(ticker, ticker)
                    if price_map.get(ticker, ticker) != "QQQ"
                    else ticker
                )
                for ticker in selected_tickers
            ]
        rows.append(
            {
                "Date": pd.Timestamp(f"{dt.year + 1}-01-01"),
                "SelectionDate": dt,
                "Type": "Recon",
                "Tickers": "|".join(selected_tickers),
                "Weights": "|".join(f"{weight:.12f}" for weight in weights),
            }
        )
    return pd.DataFrame(rows)


def _canonical_company(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    ticker = SYMBOL_HISTORY_ALIASES.get(ticker, ticker)
    return DUAL_CLASS_GROUPS.get(ticker, ticker)


def _parse_pipe_list(value) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip().upper() for item in str(value).split("|") if item.strip()]


def _parse_weight_list(value) -> list[float]:
    if pd.isna(value):
        return []
    weights: list[float] = []
    for item in str(value).split("|"):
        try:
            weights.append(float(item))
        except ValueError:
            weights.append(0.0)
    return weights


def _top_unique_company_records(row: pd.Series, top_n: int) -> list[dict[str, float | int | str]]:
    tickers = _parse_pipe_list(row.get("Tickers"))
    weights = _parse_weight_list(row.get("Weights"))
    if not tickers:
        return []

    records = []
    for rank, ticker in enumerate(tickers):
        weight = weights[rank] if rank < len(weights) else 0.0
        records.append(
            {
                "Ticker": ticker,
                "Company": _canonical_company(ticker),
                "Weight": float(weight),
                "Rank": rank,
            }
        )

    securities = pd.DataFrame(records)
    companies = (
        securities.groupby("Company", as_index=False)
        .agg(CompanyWeight=("Weight", "sum"), Rank=("Rank", "min"))
        .sort_values(["CompanyWeight", "Rank", "Company"], ascending=[False, True, True])
    )
    representatives = (
        securities.sort_values(["Company", "Weight", "Rank", "Ticker"], ascending=[True, False, True, True])
        .drop_duplicates("Company", keep="first")
        .set_index("Company")["Ticker"]
    )

    selected: list[dict[str, float | int | str]] = []
    for company in companies["Company"].head(top_n):
        ticker = representatives.get(company)
        if isinstance(ticker, str) and ticker:
            company_row = companies[companies["Company"].eq(company)].iloc[0]
            selected.append(
                {
                    "Ticker": ticker,
                    "Company": company,
                    "CompanyWeight": float(company_row["CompanyWeight"]),
                    "Rank": int(company_row["Rank"]),
                }
            )
    return selected


def _top_unique_company_tickers(row: pd.Series, top_n: int) -> list[str]:
    return [str(record["Ticker"]) for record in _top_unique_company_records(row, top_n)]


def _selection_weight_fractions(
    selected_records: list[dict[str, float | int | str]],
    weighting: str,
) -> list[float]:
    if not selected_records:
        return []

    if weighting == WEIGHTING_CAP:
        raw_weights = [float(record.get("CompanyWeight", 0.0)) for record in selected_records]
    elif weighting == WEIGHTING_RANK:
        raw_weights = list(range(len(selected_records), 0, -1))
    elif weighting == WEIGHTING_INVERSE_RANK:
        raw_weights = [1.0 / rank for rank in range(1, len(selected_records) + 1)]
    else:
        raw_weights = [1.0] * len(selected_records)

    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(selected_records)] * len(selected_records)
    return [weight / total for weight in raw_weights]


def _ndx_company_weights(q_weights: pd.DataFrame) -> pd.Series:
    mapped = q_weights[q_weights["IsMapped"].eq(True)].copy()
    mapped["Company"] = mapped["Ticker"].map(_canonical_company)
    return mapped.groupby("Company")["Weight"].sum().sort_values(ascending=False)


def _top_ndx_company_tickers(q_weights: pd.DataFrame, top_n: int) -> list[str]:
    mapped = q_weights[q_weights["IsMapped"].eq(True)].copy()
    if mapped.empty:
        return []

    mapped["Company"] = mapped["Ticker"].map(_canonical_company)
    company_weights = mapped.groupby("Company")["Weight"].sum().sort_values(ascending=False)
    representatives = (
        mapped.sort_values(["Company", "Weight", "Ticker"], ascending=[True, False, True])
        .drop_duplicates("Company", keep="first")
        .set_index("Company")["Ticker"]
    )
    return [
        representatives[company]
        for company in company_weights.head(top_n).index
        if company in representatives
    ]


def _dynamic_rows_for_range(source: str, start_date, end_date) -> pd.DataFrame:
    if source == "NDX":
        return _ndx_top_rows_for_range(start_date, end_date)
    return _annual_rows_for_range(start_date, end_date)


def _append_query(ticker: str, query_suffix: str) -> str:
    return f"{ticker}{_query_for_underlying(ticker, query_suffix)}" if query_suffix else ticker


def expand_dynamic_ticker_universe(tickers, start_date, end_date) -> list[str]:
    """Expand dynamic Nasdaq pseudo tickers into their underlying ticker universe."""
    expanded: list[str] = []
    rows_cache: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        parsed = parse_dynamic_rotation_ticker(ticker)
        if parsed is None:
            expanded.append(str(ticker))
            continue

        source, top_n = parsed
        _, query_suffix = split_ticker_query(ticker)
        if source not in rows_cache:
            rows_cache[source] = _dynamic_rows_for_range(source, start_date, end_date)
        rows = rows_cache[source]
        for _, row in rows.iterrows():
            for selected in _top_unique_company_tickers(row, top_n):
                expanded.append(_append_query(selected, query_suffix))

    return list(dict.fromkeys(expanded))


def build_dynamic_allocation_plan(
    allocation: dict,
    maint_pcts: dict | None,
    pm_maint_pcts: dict | None,
    start_date,
    end_date,
) -> DynamicAllocationPlan:
    """Return an engine-ready allocation and schedule for dynamic NDXMEGA sleeves."""
    maint_pcts = maint_pcts or {}
    pm_maint_pcts = pm_maint_pcts or {}

    static_allocation: dict[str, float] = {}
    dynamic_specs: list[_DynamicSpec] = []

    for ticker, weight in allocation.items():
        ticker_str = str(ticker).strip()
        base, query_suffix = split_ticker_query(ticker_str)
        parsed = parse_dynamic_rotation_ticker(ticker_str)
        if parsed is None:
            static_allocation[ticker_str] = float(weight)
            continue
        source, top_n = parsed

        dynamic_specs.append(
            _DynamicSpec(
                ticker=ticker_str,
                base=base,
                query_suffix=query_suffix,
                source=source,
                top_n=top_n,
                weighting=_weighting_from_query(query_suffix),
                weight=float(weight),
                maint_pct=maint_pcts.get(base),
                pm_maint_pct=pm_maint_pcts.get(base),
            )
        )

    if not dynamic_specs:
        return DynamicAllocationPlan(
            engine_allocation={str(ticker): float(weight) for ticker, weight in allocation.items()},
            dynamic_schedule=None,
            maint_pcts=dict(maint_pcts),
            pm_maint_pcts=dict(pm_maint_pcts) if pm_maint_pcts else None,
            universe_tickers=list(dict.fromkeys(str(ticker) for ticker in allocation)),
            has_dynamic=False,
            notes=[],
        )

    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    universe: list[str] = list(static_allocation)
    expanded_maint = dict(maint_pcts)
    expanded_pm_maint = dict(pm_maint_pcts) if pm_maint_pcts else {}
    notes: list[str] = []
    rows_by_source = {
        source: _dynamic_rows_for_range(source, start_date, end_date)
        for source in sorted({spec.source for spec in dynamic_specs})
    }
    if any(rows.empty for rows in rows_by_source.values()):
        missing_sources = [source for source, rows in rows_by_source.items() if rows.empty]
        raise ValueError(
            "No annual constituent rows are available for: "
            + ", ".join(missing_sources)
        )

    effective_dates = sorted(
        set().union(*(set(rows["Date"]) for rows in rows_by_source.values()))
    )

    for effective_date in effective_dates:
        target = dict(static_allocation)

        for spec in dynamic_specs:
            rows = rows_by_source[spec.source]
            row_candidates = rows[rows["Date"].le(effective_date)]
            if row_candidates.empty:
                continue
            row = row_candidates.iloc[-1]
            selected = _top_unique_company_records(row, spec.top_n)
            if not selected:
                notes.append(f"{effective_date.date()}: no {spec.source} constituents found for {spec.ticker}")
                continue

            weight_fractions = _selection_weight_fractions(selected, spec.weighting)
            for selected_record, weight_fraction in zip(selected, weight_fractions):
                selected_ticker = str(selected_record["Ticker"])
                actual_ticker = _append_query(selected_ticker, spec.query_suffix)
                target[actual_ticker] = target.get(actual_ticker, 0.0) + (spec.weight * weight_fraction)
                universe.append(actual_ticker)

                actual_base, _ = split_ticker_query(actual_ticker)
                if spec.maint_pct is not None:
                    expanded_maint[actual_base] = float(spec.maint_pct)
                if spec.pm_maint_pct is not None:
                    expanded_pm_maint[actual_base] = float(spec.pm_maint_pct)

        schedule[effective_date] = target

    universe = list(dict.fromkeys(universe))
    engine_allocation = {ticker: static_allocation.get(ticker, 0.0) for ticker in universe}

    notes.append(
        "Dynamic Nasdaq annual rotation: "
        f"{len(dynamic_specs)} sleeve(s), {len(schedule)} annual selection point(s), "
        f"{len(universe) - len(static_allocation)} historical component ticker(s)."
    )

    return DynamicAllocationPlan(
        engine_allocation=engine_allocation,
        dynamic_schedule=schedule,
        maint_pcts=expanded_maint,
        pm_maint_pcts=expanded_pm_maint if expanded_pm_maint else None,
        universe_tickers=universe,
        has_dynamic=True,
        notes=notes,
    )
