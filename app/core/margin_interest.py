"""USD margin-loan interest accrual helpers.

Synthetic leveraged tickers use Testfol's trading-day /252 convention in
``shadow_backtest.py``.  This module is intentionally limited to real cash
margin debt, which follows the USD money-market Actual/360 convention.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


USD_MARGIN_DAY_COUNT = 360.0


def _normalize_index(index) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    if result.tz is not None:
        result = result.tz_localize(None)
    return result


def usd_margin_period_rates(index, annual_rate_pct: float) -> pd.Series:
    """Return simple Actual/360 rates for each observed portfolio interval."""
    dates = _normalize_index(index)
    if dates.empty:
        return pd.Series(dtype=float, index=dates)

    elapsed_days = np.ones(len(dates), dtype=float)
    if len(dates) > 1:
        elapsed_days[1:] = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    return pd.Series(
        elapsed_days * max(float(annual_rate_pct), 0.0) / 100.0 / USD_MARGIN_DAY_COUNT,
        index=dates,
        name="USD margin period rate",
    )


@dataclass(frozen=True)
class MarginAccrualSnapshot:
    total_liability: float
    settled_principal: float
    accrued_interest: float
    effective_rate_pct: float


class UsdMarginInterestLedger:
    """Accrue USD margin interest daily and capitalize it monthly.

    Interest accrues on positive settled principal using Actual/360.  Accrued
    interest remains part of total account liability immediately, but is not
    added to settled principal until the third observed trading day of the
    following month.
    """

    def __init__(self, dates, starting_loan: float, rate_config: float | dict):
        self.dates = _normalize_index(dates)
        self.principal = float(starting_loan)
        self.rate_config = rate_config
        self.previous_date: pd.Timestamp | None = None
        self._accrued_by_month: dict[pd.Period, float] = defaultdict(float)
        self._posting_dates = self._build_posting_dates(self.dates)
        self._base_series = self._prepare_base_series(rate_config)

    @staticmethod
    def _build_posting_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
        posting_dates: set[pd.Timestamp] = set()
        if dates.empty:
            return posting_dates
        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        periods = unique_dates.to_period("M")
        for period in periods.unique():
            month_dates = unique_dates[periods == period]
            if len(month_dates) >= 3:
                posting_dates.add(pd.Timestamp(month_dates[2]))
        return posting_dates

    @staticmethod
    def _prepare_base_series(rate_config: float | dict) -> pd.Series | None:
        if not isinstance(rate_config, dict):
            return None
        base = rate_config.get("base_series")
        if base is None:
            return None
        series = pd.Series(base, copy=True)
        series.index = _normalize_index(series.index)
        series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]

    def _base_rate_pct(self, day: pd.Timestamp, fallback: float = 5.0) -> float:
        if self._base_series is None or self._base_series.empty:
            return float(fallback)
        value = self._base_series.asof(day)
        return 0.0 if pd.isna(value) else float(value)

    def _interest_and_rate(self, day: pd.Timestamp) -> tuple[float, float]:
        balance = max(self.principal, 0.0)
        config = self.rate_config

        if not isinstance(config, dict):
            annual_pct = max(float(config), 0.0)
            return balance * annual_pct / 100.0 / USD_MARGIN_DAY_COUNT, annual_pct

        mode = str(config.get("type", "Fixed"))
        if mode == "Fixed":
            annual_pct = max(float(config.get("rate_pct", 5.0)), 0.0)
            return balance * annual_pct / 100.0 / USD_MARGIN_DAY_COUNT, annual_pct

        if mode == "Variable":
            if self._base_series is None:
                annual_pct = 5.0
            else:
                base_pct = max(self._base_rate_pct(day, fallback=0.0), 0.0)
                annual_pct = max(base_pct + float(config.get("spread_pct", 1.0)), 0.0)
            return balance * annual_pct / 100.0 / USD_MARGIN_DAY_COUNT, annual_pct

        if mode == "Tiered":
            base_pct = max(self._base_rate_pct(day), 0.0)
            tiers = sorted(
                [(float(limit), float(spread)) for limit, spread in config.get("tiers", [])],
                key=lambda item: item[0],
            )
            if not tiers:
                return balance * base_pct / 100.0 / USD_MARGIN_DAY_COUNT, base_pct

            interest = 0.0
            for idx, (limit, spread) in enumerate(tiers):
                next_limit = tiers[idx + 1][0] if idx + 1 < len(tiers) else float("inf")
                chunk = min(max(0.0, balance - limit), next_limit - limit)
                if chunk > 0:
                    tier_pct = max(base_pct + spread, 0.0)
                    interest += chunk * tier_pct / 100.0 / USD_MARGIN_DAY_COUNT
                if balance < next_limit:
                    break

            if balance > 0:
                effective_pct = interest / balance * USD_MARGIN_DAY_COUNT * 100.0
            else:
                effective_pct = max(base_pct + tiers[0][1], 0.0)
            return interest, effective_pct

        raise ValueError(f"Unsupported margin rate model: {mode}")

    def _post_prior_months(self, day: pd.Timestamp) -> None:
        if day not in self._posting_dates:
            return
        current_month = day.to_period("M")
        prior_months = [period for period in self._accrued_by_month if period < current_month]
        if not prior_months:
            return
        posted = sum(self._accrued_by_month.pop(period) for period in prior_months)
        self.principal += posted

    def _accrue_day(self, day: pd.Timestamp) -> float:
        interest, _ = self._interest_and_rate(day)
        if interest:
            self._accrued_by_month[day.to_period("M")] += interest
        return interest

    @property
    def accrued_interest(self) -> float:
        return float(sum(self._accrued_by_month.values()))

    @property
    def total_liability(self) -> float:
        return self.principal + self.accrued_interest

    def advance(self, date, loan_change: float = 0.0) -> MarginAccrualSnapshot:
        """Advance through ``date``, apply its cashflow, then accrue that day."""
        day = pd.Timestamp(date).normalize()
        if day.tz is not None:
            day = day.tz_localize(None)
        if self.previous_date is not None and day <= self.previous_date:
            raise ValueError("Margin ledger dates must be strictly increasing")

        if self.previous_date is not None:
            for interim in pd.date_range(self.previous_date + pd.Timedelta(days=1), day - pd.Timedelta(days=1), freq="D"):
                self._accrue_day(interim)

        self._post_prior_months(day)
        self.principal += float(loan_change)
        _, effective_rate_pct = self._interest_and_rate(day)
        self._accrue_day(day)
        self.previous_date = day

        return MarginAccrualSnapshot(
            total_liability=self.total_liability,
            settled_principal=self.principal,
            accrued_interest=self.accrued_interest,
            effective_rate_pct=effective_rate_pct,
        )
