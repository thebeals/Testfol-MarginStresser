"""Rebalance frequency verification and screener wiring.

Phase 2 step 8: Verified 2026-08-17. The local shadow-backtest engine
supports all screener-required frequencies per SCREENER-SPEC.md (None /
Monthly / Quarterly / Yearly). The testfol.io live API supports a broader
enum (Daily, Weekly, Bi-monthly, Every 4 Months, Semiannually, Every 2
Years, Every 5 Years) but the screener deliberately restricts to the
default four per spec --- calendar-based only, no exotic frequencies,
tested per candidate as a fixed parameter sweep rather than a search
dimension.

Verification detail:
  Local engine (shadow_backtest.py rebalance_freq) accepts:
    None, Monthly, Quarterly, Yearly, Custom, Threshold, Threshold+Calendar
  testfol.io live API accepts (confirmed via 422 validation):
    Daily, Weekly, Monthly, Bi-monthly, Every 4 Months, Quarterly,
    Semiannually, Yearly, Every 2 Years, Every 5 Years, None
  Overlap (screener default set): None, Monthly, Quarterly, Yearly.
  No bridging needed --- the local engine covers the screener's needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SCREENER_REBALANCE_FREQUENCIES: Final[tuple[str, ...]] = ("None", "Monthly", "Quarterly", "Yearly")

TESTFOLIO_API_FREQUENCIES: Final[tuple[str, ...]] = (
    "Daily",
    "Weekly",
    "Monthly",
    "Bi-monthly",
    "Every 4 Months",
    "Quarterly",
    "Semiannually",
    "Yearly",
    "Every 2 Years",
    "Every 5 Years",
    "None",
)

LOCAL_ENGINE_FREQUENCIES: Final[tuple[str, ...]] = (
    "None",
    "Monthly",
    "Quarterly",
    "Yearly",
    "Custom",
    "Threshold",
    "Threshold+Calendar",
)


@dataclass(frozen=True)
class RebalanceConfig:
    freq: str
    offset: int = 0


def screener_defaults() -> tuple[RebalanceConfig, ...]:
    return tuple(RebalanceConfig(freq) for freq in SCREENER_REBALANCE_FREQUENCIES)


def verify_engine_support() -> dict[str, bool]:
    return {freq: freq in LOCAL_ENGINE_FREQUENCIES for freq in SCREENER_REBALANCE_FREQUENCIES}
