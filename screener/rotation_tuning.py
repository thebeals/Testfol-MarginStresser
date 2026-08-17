"""Phase 5 — Deferred (do not build now).

See SCREENER-SPEC.md "Entry/exit and rotation" section.

Deferred in this pass (documented now so it isn't a surprise later):
  1. Full signal-based rotation exploration / parameter tuning. Phase 3 ships a
     small fixed, never-tuned set (SMA200/SPY, EMA100/SPY). Phase 5 is the
     exploration of new signal types and tuned periods.
  2. Contribution-timing variants (weekly DCA, dip-buy-threshold), a small
     curated hardcoded set, never optimizer-tuned.

General principle (SCREENER-SPEC.md): continuous + economically-smooth
dimensions get searched by the optimizer; discrete named-strategy choices where
free numerical tuning is a known overfitting trap use a small curated hardcoded
set, never tuned. Contribution timing changes the realized return sequence, so
DSR (and PBO, if/when sourced) must run on the full combined candidate
(allocation x timing-variant x rotation-variant), not allocation alone.

Nothing here is implemented yet. Revisit only when Phase 2/3 are complete and
Jonathan authorizes Phase 5 work.
"""

from __future__ import annotations


class _DeferredNotImplemented(NotImplementedError):
    """Marker for deferred Phase 5 functionality — do not call."""


def explore_rotation_variants() -> None:
    """(Deferred) Signal-based rotation exploration and parameter tuning."""
    raise _DeferredNotImplemented(
        "Phase 5 (deferred): full signal-rotation exploration/tuning. "
        "See SCREENER-SPEC.md 'Entry/exit and rotation'. Do not build in this pass."
    )


def contribution_timing_variants() -> None:
    """(Deferred) Contribution-timing variants: weekly DCA, dip-buy-threshold."""
    raise _DeferredNotImplemented(
        "Phase 5 (deferred): contribution-timing variants. "
        "See SCREENER-SPEC.md 'Entry/exit and rotation'. Do not build in this pass."
    )
