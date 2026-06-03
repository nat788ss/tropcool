"""Wet-bulb temperature from dry-bulb T and relative humidity (weather CSV columns)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def wet_bulb_temperature_celsius(
    dry_bulb_celsius: float, relative_humidity_percent: float
) -> float:
    """Approximate wet-bulb temperature (°C) from dry bulb (°C) and RH (%).

    Stull (2011) empirical fit; suitable for typical weather-range *T* and RH.
    """
    t = dry_bulb_celsius
    rh = relative_humidity_percent
    return (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )


def wet_bulb_temperature_for_each_row(
    dry_bulb_c: np.ndarray | Any,
    relative_humidity_pct: np.ndarray | Any,
    *,
    decimals: int | None = 4,
) -> np.ndarray:
    """Compute wet bulb for each element of aligned 1-D (or flattened) arrays.

    Non-finite or missing inputs yield ``nan`` for that element.
    """
    t = np.asarray(dry_bulb_c, dtype=np.float64)
    rh = np.asarray(relative_humidity_pct, dtype=np.float64)
    if t.shape != rh.shape:
        raise ValueError("dry_bulb and relative_humidity must have the same shape")
    out = np.full(t.shape, np.nan, dtype=np.float64)
    flat_t = t.ravel()
    flat_rh = rh.ravel()
    flat_o = out.ravel()
    for i in range(flat_t.size):
        if not (np.isfinite(flat_t[i]) and np.isfinite(flat_rh[i])):
            continue
        try:
            w = wet_bulb_temperature_celsius(float(flat_t[i]), float(flat_rh[i]))
            if decimals is not None:
                w = round(w, int(decimals))
            flat_o[i] = w
        except (TypeError, ValueError):
            pass
    return out


def add_wet_bulb_column(
    df: Any,
    *,
    temp_col: str = "temperature_2m",
    rh_col: str = "relative_humidity_2m",
    out_col: str = "wet_bulb_temperature_2m",
    decimals: int | None = 4,
) -> Any:
    """Return *df* with ``out_col`` added from *temp_col* and *rh_col* (pandas DataFrame)."""
    if pd is None:
        raise ImportError("add_wet_bulb_column requires pandas")
    t = df[temp_col].to_numpy(dtype=np.float64, copy=False)
    rh = df[rh_col].to_numpy(dtype=np.float64, copy=False)
    wb = wet_bulb_temperature_for_each_row(t, rh, decimals=decimals)
    out = df.copy()
    out[out_col] = wb
    return out
