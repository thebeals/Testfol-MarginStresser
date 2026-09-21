"""Readable structured generation summaries without LLM calls."""

from __future__ import annotations

import json
from pathlib import Path


def log_generation_summary(
    output_path: str | Path,
    *,
    generation: int,
    candidates_tested: int,
    survivor_pool_size: int,
    top_n_changed: bool,
    stale_generations: int,
) -> None:
    record = {
        "event": "generation_summary",
        "generation": generation,
        "candidates_tested": candidates_tested,
        "survivor_pool_size": survivor_pool_size,
        "top_n_changed": top_n_changed,
        "stale_generations": stale_generations,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
