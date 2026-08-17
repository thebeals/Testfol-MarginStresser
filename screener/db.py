"""SQLite persistence and canonical candidate deduplication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_hash TEXT PRIMARY KEY,
    allocation_json TEXT NOT NULL,
    rebalance_freq TEXT NOT NULL,
    rotation_variant TEXT NOT NULL,
    generation INTEGER NOT NULL,
    fitness REAL,
    mwrr REAL,
    dsr REAL,
    confidence_badge TEXT,
    diversification_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def canonical_hash(allocation: dict[str, float], rebalance_freq: str = "None", rotation_variant: str = "static") -> str:
    payload = {
        "allocation": {key: round(float(allocation[key]), 12) for key in sorted(allocation)},
        "rebalance_freq": rebalance_freq,
        "rotation_variant": rotation_variant,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CandidateStore:
    """Small SQLite store. Callers should use one writer process."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(SCHEMA)
        self.connection.commit()

    def contains(self, allocation: dict[str, float], rebalance_freq: str = "None", rotation_variant: str = "static") -> bool:
        key = canonical_hash(allocation, rebalance_freq, rotation_variant)
        return self.connection.execute("SELECT 1 FROM candidates WHERE candidate_hash = ?", (key,)).fetchone() is not None

    def save(self, result: dict[str, object]) -> bool:
        allocation = result["allocation"]
        key = canonical_hash(allocation, str(result.get("rebalance_freq", "None")), str(result.get("rotation_variant", "static")))
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO candidates
            (candidate_hash, allocation_json, rebalance_freq, rotation_variant,
             generation, fitness, mwrr, dsr, confidence_badge, diversification_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                json.dumps(allocation, sort_keys=True),
                result.get("rebalance_freq", "None"),
                result.get("rotation_variant", "static"),
                result.get("generation", 0),
                result.get("fitness"),
                result.get("mwrr"),
                result.get("dsr"),
                result.get("confidence_badge"),
                json.dumps(result.get("diversification", {}), sort_keys=True),
            ),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.close()
