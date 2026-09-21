"""Canonical screened-allocation result loading."""

from __future__ import annotations

import json
from pathlib import Path


def load_screened_results(path: str | Path = "data/screener-results.json") -> list[dict[str, object]]:
    """Return only allocations that passed the local and Testfol gates."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("accepted", []))
