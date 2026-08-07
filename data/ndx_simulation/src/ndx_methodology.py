"""Nasdaq-100 weighting rules used by the reconstruction pipeline.

The 2026 methodology became effective on 2026-05-01.  Keep these helpers
pure so the calculation can be tested independently from data downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


NEW_METHODOLOGY_EFFECTIVE_DATE = pd.Timestamp("2026-05-01")


def uses_2026_methodology(effective_date) -> bool:
    """Return whether an index event is governed by the 2026 methodology."""
    return pd.Timestamp(effective_date).normalize() >= NEW_METHODOLOGY_EFFECTIVE_DATE


def modified_market_cap_factor(
    total_shares,
    float_shares,
    effective_date,
) -> float:
    """Return Nasdaq's listed-share factor for a low-float security.

    From 2026-05-01, shares used for weighting are the lesser of total shares
    outstanding or three times free-floating shares. Missing share data falls
    back to 1.0 so a transient metadata failure cannot remove a constituent.
    """
    if not uses_2026_methodology(effective_date):
        return 1.0
    try:
        total = float(total_shares)
        floating = float(float_shares)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(total) or not np.isfinite(floating) or total <= 0 or floating <= 0:
        return 1.0
    return float(min(1.0, 3.0 * floating / total))


def build_validation_periods(weight_dates, data_end):
    """Pair each rebalance with the next one, carrying the last to data_end."""
    dates = sorted(pd.Timestamp(date) for date in weight_dates)
    if not dates:
        return []
    data_end = pd.Timestamp(data_end)
    return [
        (start, dates[index + 1] if index + 1 < len(dates) else data_end)
        for index, start in enumerate(dates)
        if start <= data_end
    ]


def interpolate_new_entry_values(values, new_entries):
    """Interpolate new-security values between adjacent existing constituents.

    Nasdaq ranks additions by Modified Market Capitalization, then assigns
    weights by linear interpolation between the next-largest and next-smallest
    existing constituents. Contiguous runs are evenly spaced between those
    boundaries. An edge entry without two boundaries keeps its input value.
    """
    values = pd.Series(values, dtype=float).copy()
    values = values[values.notna() & (values > 0)]
    additions = set(new_entries) & set(values.index)
    if not additions:
        return values, set()

    ordered = values.sort_values(ascending=False)
    tickers = ordered.index.tolist()
    adjusted = ordered.copy()
    interpolated = set()
    pos = 0

    while pos < len(tickers):
        if tickers[pos] not in additions:
            pos += 1
            continue

        run_start = pos
        while pos < len(tickers) and tickers[pos] in additions:
            pos += 1
        run_end = pos

        larger_pos = run_start - 1
        smaller_pos = run_end
        if larger_pos < 0 or smaller_pos >= len(tickers):
            continue

        larger = float(adjusted.iloc[larger_pos])
        smaller = float(adjusted.iloc[smaller_pos])
        count = run_end - run_start
        targets = np.linspace(larger, smaller, count + 2)[1:-1]
        for offset, target in enumerate(targets):
            ticker = tickers[run_start + offset]
            adjusted.loc[ticker] = float(target)
            interpolated.add(ticker)

    return adjusted.reindex(values.index), interpolated


def _normalize(weights):
    result = pd.Series(weights, dtype=float).clip(lower=0).fillna(0.0)
    total = float(result.sum())
    return result / total if total > 0 else result


def _allocate_with_cap(weights, target_total, cap):
    """Allocate a target total proportionally without crossing a hard cap."""
    source = pd.Series(weights, dtype=float).clip(lower=0).fillna(0.0)
    result = pd.Series(0.0, index=source.index)
    remaining = list(source.index)
    remaining_total = float(target_total)

    while remaining and remaining_total > 1e-14:
        base = source.loc[remaining]
        if float(base.sum()) <= 0:
            proposal = pd.Series(
                remaining_total / len(remaining),
                index=remaining,
            )
        else:
            proposal = base / float(base.sum()) * remaining_total
        over = proposal > cap + 1e-12
        if not over.any():
            result.loc[remaining] = proposal
            remaining_total = 0.0
            break

        capped = proposal.index[over]
        result.loc[capped] = cap
        remaining_total -= cap * len(capped)
        remaining = [name for name in remaining if name not in set(capped)]

    if remaining_total > 1e-10:
        raise ValueError("Weight cap has insufficient capacity for target allocation")
    return result


def _cap_and_redistribute(weights, cap):
    """Cap weights and redistribute excess proportionally to uncapped names."""
    return _allocate_with_cap(_normalize(weights), 1.0, cap)


def _apply_company_constraints(security_weights, company_by_ticker):
    security = _normalize(security_weights)
    companies = pd.Series(company_by_ticker).reindex(security.index)
    companies = companies.fillna(pd.Series(security.index, index=security.index))
    company_weights = security.groupby(companies).sum()

    if float(company_weights.max()) > 0.24 + 1e-12:
        company_weights = _cap_and_redistribute(company_weights, 0.20)

    cohort = company_weights > 0.045
    cohort_sum = float(company_weights.loc[cohort].sum())
    if cohort_sum >= 0.48 - 1e-12 and cohort.any() and (~cohort).any():
        company_weights.loc[cohort] *= 0.40 / cohort_sum
        outside = company_weights.index[~cohort]
        # Nasdaq preserves the initial company-weight rank order. After the
        # large-company cohort is reduced to 40%, an outside company therefore
        # cannot overtake the smallest member of that cohort. Keeping this cap
        # at or below 4.5% also prevents redistribution from creating a new
        # cohort that itself breaches the 48% trigger.
        outside_cap = min(0.045, float(company_weights.loc[cohort].min()))
        company_weights.loc[outside] = _allocate_with_cap(
            company_weights.loc[outside],
            0.60,
            outside_cap,
        )

    original_company = security.groupby(companies).sum()
    output = security.copy()
    for company, target in company_weights.items():
        members = companies.index[companies == company]
        original = float(original_company.loc[company])
        if original > 0:
            output.loc[members] *= float(target) / original
    return _normalize(output)


def _apply_security_constraints(security_weights):
    security = _normalize(security_weights)
    if float(security.max()) > 0.15 + 1e-12:
        security = _cap_and_redistribute(security, 0.14)

    ranked = security.sort_values(ascending=False)
    if len(ranked) >= 5 and float(ranked.iloc[:5].sum()) >= 0.40 - 1e-12:
        top_five = ranked.index[:5]
        top_sum = float(security.loc[top_five].sum())
        security.loc[top_five] *= 0.385 / top_sum
        outside = security.index.difference(top_five)
        outside_cap = min(0.044, float(security.loc[top_five].min()))
        security.loc[outside] = _allocate_with_cap(
            security.loc[outside],
            0.615,
            outside_cap,
        )
    return _normalize(security)


def apply_2026_ndx_capping(group, is_annual=False):
    """Apply the post-May-2026 Nasdaq-100 weighting constraints.

    Company-level constraints are applied at every scheduled rebalance. The
    December annual reconstitution then applies the security-level constraints.
    Multiple eligible share classes remain separate securities but are combined
    when evaluating company limits.
    """
    tickers = group["Ticker"].astype(str).tolist()
    initial = pd.Series(group["Weight"].astype(float).values, index=tickers)
    company_map = {
        ticker: getattr(config, "DUAL_CLASS_GROUPS", {}).get(ticker, ticker)
        for ticker in tickers
    }
    adjusted = _apply_company_constraints(initial, company_map)
    if is_annual:
        adjusted = _apply_security_constraints(adjusted)
    return pd.DataFrame(
        {
            "Ticker": adjusted.index,
            "CappedWeight": adjusted.values,
        }
    )
