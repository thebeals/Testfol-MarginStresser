"""Sequential live testfol.io verification helper."""

from __future__ import annotations

import time
from typing import Callable, Iterable


def verify_sequential(
    candidates: Iterable[dict[str, object]],
    request: Callable[[dict[str, object]], dict[str, object]],
    *,
    delay_seconds: float = 1.5,
) -> list[dict[str, object]]:
    """Verify candidates sequentially; never parallelize this endpoint."""
    results = []
    for index, candidate in enumerate(candidates):
        if index:
            time.sleep(delay_seconds)
        results.append(request(candidate))
    return results
