"""Small statistical helpers used by chart modules.

SciPy is optional at runtime here. These functions intentionally use NumPy-only
moment formulas so chart rendering does not fail if a local SciPy wheel/conda
binary is broken or unavailable.
"""
from __future__ import annotations

import numpy as np


def _clean(values) -> np.ndarray:
    """Return a 1-D float array with NaN values omitted."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def skewness(values) -> float:
    """Biased Fisher-Pearson skewness, matching scipy.stats.skew defaults."""
    arr = _clean(values)
    if arr.size == 0:
        return float("nan")

    centered = arr - np.mean(arr)
    m2 = np.mean(centered ** 2)
    if not np.isfinite(m2) or m2 == 0:
        return float("nan")

    m3 = np.mean(centered ** 3)
    return float(m3 / (m2 ** 1.5))


def excess_kurtosis(values) -> float:
    """Biased excess kurtosis, matching scipy.stats.kurtosis defaults."""
    arr = _clean(values)
    if arr.size == 0:
        return float("nan")

    centered = arr - np.mean(arr)
    m2 = np.mean(centered ** 2)
    if not np.isfinite(m2) or m2 == 0:
        return float("nan")

    m4 = np.mean(centered ** 4)
    return float((m4 / (m2 ** 2)) - 3.0)
