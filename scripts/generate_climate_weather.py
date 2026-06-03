#!/usr/bin/env python3
"""Synthetic future weather from historical CSVs (IPCC-style warming deltas).

Writes ``data/weather/projections/{slug}_{period}.csv`` for periods:
current (copy), 2030 (+0.5°C), 2040 (+1.2°C), 2050 (+2.0°C),
2040_stress (+2.5°C), 2050_stress (+3.0°C).

Humidity is reduced proportionally (~1.5% per °C warming) so wet-bulb rises
less than dry-bulb alone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_data import CITY_SLUGS
from predict_year import default_weather_path, load_weather_csv
from weather_utils import add_wet_bulb_column

PERIODS: dict[str, float] = {
    "current": 0.0,
    "2030": 0.5,
    "2040": 1.2,
    "2050": 2.0,
    "2040_stress": 2.5,
    "2050_stress": 3.0,
}


def apply_warming(df: pd.DataFrame, delta_c: float) -> pd.DataFrame:
    out = df.copy()
    if delta_c <= 0:
        return out
    t = out["temperature_2m"].astype(float) + delta_c
    out["temperature_2m"] = t
    rh = out["relative_humidity_2m"].astype(float)
    # Slight RH reduction with warming (keeps vapor pressure roughly plausible).
    factor = max(0.75, 1.0 - 0.015 * delta_c)
    out["relative_humidity_2m"] = (rh * factor).clip(25.0, 100.0)
    if "wet_bulb_temperature_2m" in out.columns:
        out = out.drop(columns=["wet_bulb_temperature_2m"])
    return add_wet_bulb_column(out)


def projection_path(out_dir: Path, slug: str, period: str) -> Path:
    return out_dir / f"{slug}_{period}.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weather-dir", type=Path, default=ROOT / "data/weather")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/weather/projections",
    )
    ap.add_argument(
        "--periods",
        type=str,
        default="current,2030,2040,2050,2040_stress,2050_stress",
        help="Comma-separated subset of period keys (incl. 2040_stress, 2050_stress)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = [p.strip() for p in args.periods.split(",") if p.strip()]
    for p in want:
        if p not in PERIODS:
            print(f"Unknown period {p!r}", file=sys.stderr)
            return 2

    for slug in CITY_SLUGS:
        src = default_weather_path(slug, args.weather_dir)
        if not src.is_file():
            print(f"MISSING {src}", file=sys.stderr)
            return 2
        base = pd.read_csv(src)
        for period in want:
            dest = projection_path(args.out_dir, slug, period)
            delta = PERIODS[period]
            if period == "current" and delta == 0:
                shutil.copy2(src, dest)
            else:
                warmed = apply_warming(base, delta)
                warmed.to_csv(dest, index=False)
            print(f"  {dest.name}  (+{delta:.1f}°C)" if delta else f"  {dest.name}  (baseline copy)")
    print(f"Wrote projections under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
