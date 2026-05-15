#!/usr/bin/env python3
"""Generate a large supervised dataset from the DC simulator + real hourly weather.

For **each hour** of weather (six tropical / SEA cities × ~20 years from
``fetch_weather_historical.py`` → ``data/weather/{slug}_hourly_2004_2024.csv``):

1. Read ``outdoor_temp`` (``temperature_2m``), ``humidity``, compute ``wet_bulb`` if missing.
2. **Uniform random controls:** ``chiller_setpoint`` ∈ [5, 12] °C; ``crah_fan_speed``,
   ``tower_fan_speed`` ∈ [0.3, 1.0].
3. **Uniform random facility:** ``IT_load`` ∈ [5, 50] MW; ``base_COP`` ∈ [5.0, 7.0] (passed as
   ``reference_cop``), mapped to ``equipment_age`` (years, newer ⇔ higher COP).

**Feature vector → targets (Parquet columns):**

- **[city_id, year, month, hour, outdoor_temp, humidity, wet_bulb, IT_load, equipment_age,
  chiller_setpoint, crah_fan_speed, tower_fan_speed]**
  → **[rack_inlet_temp, cooling_power, water_consumption, pue, cost]**

``cost`` = TOU electricity (RM/kWh by hour) × ``cooling_power`` + water price × liters.

**Scale:** default ``--target-rows`` **8_000_000** (aim for 5–10 M). Rows repeat weather hours in
**epochs** with fresh RNG samples until the target row count is reached.

Example::

    pip install pandas pyarrow tqdm
    python3 fetch_weather_historical.py
    python3 generate_data.py --target-rows 8000000 --output data/training/tropcool_train.parquet
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pip install pandas pyarrow", file=sys.stderr)
    raise

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("pip install pyarrow", file=sys.stderr)
    raise

from dc_simulator import DataCenterSimulator
from weather_utils import add_wet_bulb_column

# Align with fetch_weather_historical.LOCATIONS order → stable city_id
CITY_SLUGS: tuple[str, ...] = (
    "cyberjaya",
    "singapore",
    "jakarta",
    "bangkok",
    "johor_bahru",
    "ho_chi_minh_city",
)
SLUG_TO_ID: dict[str, int] = {s: i for i, s in enumerate(CITY_SLUGS)}


def _electricity_rm_per_kwh(hour_of_day: int) -> float:
    """Simple TOU tariff: peak 08–22, off otherwise (RM/kWh)."""
    h = int(hour_of_day) % 24
    return 0.40 if 8 <= h < 22 else 0.25


def _equipment_age_years(base_cop: np.ndarray) -> np.ndarray:
    """Higher COP ⇒ newer equipment (years ~ 0–12 for base_cop in [5, 7])."""
    return np.clip((7.0 - base_cop) * 6.0, 0.0, 15.0)


def _rated_capacity_mw(it_mw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slightly oversized chillers; small jitter for diversity."""
    spread = rng.uniform(0.98, 1.12, size=it_mw.shape)
    return np.maximum(it_mw * 1.22 * spread, it_mw + 0.5)


def _make_facility_episodes(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate piecewise-constant facility episodes + sticky control actions.

    This widens PUE across *facilities* by randomizing plant parasitics and tower/chiller
    parameters while keeping them stable for multi-day episodes.
    """
    # Preallocate signals
    ch_sp = np.empty(n, dtype=np.float64)
    crah_s = np.empty(n, dtype=np.float64)
    tow_s = np.empty(n, dtype=np.float64)
    it_mw = np.empty(n, dtype=np.float64)
    base_cop = np.empty(n, dtype=np.float64)

    # Plant parameters (episode-stable)
    overhead_frac = np.empty(n, dtype=np.float64)
    pump_per_kw = np.empty(n, dtype=np.float64)
    crah_rated_kw = np.empty(n, dtype=np.float64)
    ct_rated_kw = np.empty(n, dtype=np.float64)
    tower_eta0 = np.empty(n, dtype=np.float64)
    tower_eta_wb = np.empty(n, dtype=np.float64)
    tower_eta_max = np.empty(n, dtype=np.float64)
    cop_penalty_k = np.empty(n, dtype=np.float64)
    design_outdoor_c = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        ep_len = int(rng.integers(24, 24 * 7 + 1))  # 1–7 days
        j = min(n, i + ep_len)
        steps = j - i

        it0 = float(rng.uniform(5.0, 50.0))
        cop0 = float(rng.uniform(5.0, 7.0))
        sp0 = float(rng.uniform(5.0, 12.0))
        crah0 = float(rng.uniform(0.3, 1.0))
        tow0 = float(rng.uniform(0.3, 1.0))

        oh = float(rng.uniform(0.15, 0.45))
        pump = float(rng.uniform(0.03, 0.12))
        crah_r = float(rng.uniform(150.0, 650.0))
        ct_r = float(rng.uniform(80.0, 380.0))
        eta0 = float(rng.uniform(0.25, 0.55))
        eta_wb = float(rng.uniform(0.015, 0.060))
        eta_max = float(rng.uniform(0.70, 0.95))
        k_out = float(rng.uniform(0.001, 0.012))
        t_des = float(rng.uniform(24.0, 31.0))

        # Random walks within episode to create temporal structure
        ch_sp[i:j] = np.clip(sp0 + rng.normal(0.0, 0.08, size=steps).cumsum(), 5.0, 12.0)
        crah_s[i:j] = np.clip(crah0 + rng.normal(0.0, 0.02, size=steps).cumsum(), 0.3, 1.0)
        tow_s[i:j] = np.clip(tow0 + rng.normal(0.0, 0.02, size=steps).cumsum(), 0.3, 1.0)
        it_mw[i:j] = np.clip(it0 + rng.normal(0.0, 0.25, size=steps).cumsum(), 5.0, 50.0)
        base_cop[i:j] = np.clip(cop0 + rng.normal(0.0, 0.02, size=steps).cumsum(), 5.0, 7.0)

        overhead_frac[i:j] = oh
        pump_per_kw[i:j] = pump
        crah_rated_kw[i:j] = crah_r
        ct_rated_kw[i:j] = ct_r
        tower_eta0[i:j] = eta0
        tower_eta_wb[i:j] = eta_wb
        tower_eta_max[i:j] = eta_max
        cop_penalty_k[i:j] = k_out
        design_outdoor_c[i:j] = t_des

        i = j

    rated_mw = _rated_capacity_mw(it_mw, rng)
    return {
        "chiller_sp": ch_sp,
        "crah": crah_s,
        "tower": tow_s,
        "it_mw": it_mw,
        "base_cop": base_cop,
        "rated_mw": rated_mw,
        "facility_overhead_fraction_of_it": overhead_frac,
        "pump_kw_per_kw_load": pump_per_kw,
        "crah_fan_rated_power_kw": crah_rated_kw,
        "cooling_tower_fan_rated_power_kw": ct_rated_kw,
        "tower_effectiveness_base": tower_eta0,
        "tower_effectiveness_per_k_depression": tower_eta_wb,
        "tower_effectiveness_max": tower_eta_max,
        "outdoor_cop_penalty_per_k": cop_penalty_k,
        "design_outdoor_temp_c": design_outdoor_c,
    }


def _slug_from_filename(path: Path) -> str | None:
    stem = path.stem
    if "_hourly_" not in stem:
        return None
    return stem.split("_hourly_")[0]


def _weather_csv_per_city(weather_dir: Path) -> list[Path]:
    """Resolve exactly one ``*_hourly_*.csv`` per city, in ``CITY_SLUGS`` order.

    If multiple files exist for a slug (e.g. different year spans), pick the lexicographically
    last name so ``…_2004_2024`` wins over shorter-range exports.
    """
    paths: list[Path] = []
    missing: list[str] = []
    for slug in CITY_SLUGS:
        cand = sorted(weather_dir.glob(f"{slug}_hourly_*.csv"))
        if not cand:
            missing.append(slug)
            continue
        paths.append(cand[-1])
    if missing:
        print(
            f"No weather CSV for slug(s): {missing}. Expected under {weather_dir}",
            file=sys.stderr,
        )
    return paths


def _count_hours(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with p.open("rb") as f:
            total += sum(1 for _ in f) - 1
    return total


def _load_weather_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    slug = _slug_from_filename(path)
    if slug is None or slug not in SLUG_TO_ID:
        raise ValueError(f"unknown weather file slug for {path}")
    df["city_id"] = SLUG_TO_ID[slug]
    if "wet_bulb_temperature_2m" not in df.columns:
        df = add_wet_bulb_column(df)
    ts = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["hour"] = ts.dt.hour
    return df


def _simulate_chunk_arrays(
    outdoor_temp: np.ndarray,
    humidity: np.ndarray,
    wet_bulb: np.ndarray,
    year: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
    city_id: np.ndarray,
    chiller_sp: np.ndarray,
    crah: np.ndarray,
    tower: np.ndarray,
    it_mw: np.ndarray,
    base_cop: np.ndarray,
    rated_mw: np.ndarray,
    facility_overhead_fraction_of_it: np.ndarray,
    pump_kw_per_kw_load: np.ndarray,
    crah_fan_rated_power_kw: np.ndarray,
    cooling_tower_fan_rated_power_kw: np.ndarray,
    tower_effectiveness_base: np.ndarray,
    tower_effectiveness_per_k_depression: np.ndarray,
    tower_effectiveness_max: np.ndarray,
    outdoor_cop_penalty_per_k: np.ndarray,
    design_outdoor_temp_c: np.ndarray,
    water_price_per_l: float,
) -> dict[str, np.ndarray]:
    n = len(outdoor_temp)
    rack = np.empty(n)
    cool = np.empty(n)
    water = np.empty(n)
    pue = np.empty(n)
    cost = np.empty(n)

    for i in range(n):
        sim = DataCenterSimulator(
            reference_cop=float(base_cop[i]),
            chiller_rated_capacity_mw=float(rated_mw[i]),
            facility_overhead_fraction_of_it=float(facility_overhead_fraction_of_it[i]),
            pump_kw_per_kw_load=float(pump_kw_per_kw_load[i]),
            crah_fan_rated_power_kw=float(crah_fan_rated_power_kw[i]),
            cooling_tower_fan_rated_power_kw=float(cooling_tower_fan_rated_power_kw[i]),
            tower_effectiveness_base=float(tower_effectiveness_base[i]),
            tower_effectiveness_per_k_depression=float(tower_effectiveness_per_k_depression[i]),
            tower_effectiveness_max=float(tower_effectiveness_max[i]),
            outdoor_cop_penalty_per_k=float(outdoor_cop_penalty_per_k[i]),
            design_outdoor_temp_c=float(design_outdoor_temp_c[i]),
            thermal_storage=None,
        )
        r = sim.simulate_hour(
            float(outdoor_temp[i]),
            float(humidity[i]),
            float(it_mw[i]),
            float(chiller_sp[i]),
            float(crah[i]),
            float(tower[i]),
            wet_bulb_c=float(wet_bulb[i]),
        )
        rack[i] = r.rack_inlet_temperature_c
        cool[i] = r.total_cooling_power_kw
        water[i] = r.water_consumption_liters
        pue[i] = r.pue
        tariff = _electricity_rm_per_kwh(int(hour[i]))
        cost[i] = tariff * cool[i] + water_price_per_l * water[i]

    age = _equipment_age_years(base_cop)
    return {
        "city_id": city_id.astype(np.int16),
        "year": year.astype(np.int16),
        "month": month.astype(np.int8),
        "hour": hour.astype(np.int8),
        "outdoor_temp": outdoor_temp.astype(np.float32),
        "humidity": humidity.astype(np.float32),
        "wet_bulb": wet_bulb.astype(np.float32),
        "IT_load": it_mw.astype(np.float32),
        "equipment_age": age.astype(np.float32),
        "chiller_setpoint": chiller_sp.astype(np.float32),
        "crah_fan_speed": crah.astype(np.float32),
        "tower_fan_speed": tower.astype(np.float32),
        "facility_overhead_fraction_of_it": facility_overhead_fraction_of_it.astype(np.float32),
        "pump_kw_per_kw_load": pump_kw_per_kw_load.astype(np.float32),
        "crah_fan_rated_power_kw": crah_fan_rated_power_kw.astype(np.float32),
        "cooling_tower_fan_rated_power_kw": cooling_tower_fan_rated_power_kw.astype(np.float32),
        "tower_effectiveness_base": tower_effectiveness_base.astype(np.float32),
        "tower_effectiveness_per_k_depression": tower_effectiveness_per_k_depression.astype(np.float32),
        "tower_effectiveness_max": tower_effectiveness_max.astype(np.float32),
        "outdoor_cop_penalty_per_k": outdoor_cop_penalty_per_k.astype(np.float32),
        "design_outdoor_temp_c": design_outdoor_temp_c.astype(np.float32),
        "rack_inlet_temp": rack.astype(np.float32),
        "cooling_power": cool.astype(np.float32),
        "water_consumption": water.astype(np.float32),
        "pue": pue.astype(np.float32),
        "cost": cost.astype(np.float32),
    }


def _worker_batch(payload: tuple[Any, ...]) -> dict[str, np.ndarray]:
    (
        outdoor_temp,
        humidity,
        wet_bulb,
        year,
        month,
        hour,
        city_id,
        chiller_sp,
        crah,
        tower,
        it_mw,
        base_cop,
        rated_mw,
        facility_overhead_fraction_of_it,
        pump_kw_per_kw_load,
        crah_fan_rated_power_kw,
        cooling_tower_fan_rated_power_kw,
        tower_effectiveness_base,
        tower_effectiveness_per_k_depression,
        tower_effectiveness_max,
        outdoor_cop_penalty_per_k,
        design_outdoor_temp_c,
        water_price,
    ) = payload
    return _simulate_chunk_arrays(
        outdoor_temp,
        humidity,
        wet_bulb,
        year,
        month,
        hour,
        city_id,
        chiller_sp,
        crah,
        tower,
        it_mw,
        base_cop,
        rated_mw,
        facility_overhead_fraction_of_it,
        pump_kw_per_kw_load,
        crah_fan_rated_power_kw,
        cooling_tower_fan_rated_power_kw,
        tower_effectiveness_base,
        tower_effectiveness_per_k_depression,
        tower_effectiveness_max,
        outdoor_cop_penalty_per_k,
        design_outdoor_temp_c,
        water_price,
    )


def generate(
    *,
    weather_dir: Path,
    output_path: Path,
    target_rows: int,
    seed: int,
    chunk_rows: int,
    water_price_per_liter: float,
    jobs: int,
) -> None:
    rng = np.random.default_rng(seed)
    known = _weather_csv_per_city(weather_dir)
    if not known:
        print(
            f"No city weather CSVs under {weather_dir}. Run fetch_weather_historical.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if len(known) < len(CITY_SLUGS):
        print(
            f"Expected {len(CITY_SLUGS)} cities; found {len(known)} CSV(s). "
            "Continuing with available cities (fewer geographic diversity).",
            file=sys.stderr,
        )

    total_hours_one_epoch = _count_hours(known)
    if total_hours_one_epoch <= 0:
        print("Weather CSVs appear empty.", file=sys.stderr)
        raise SystemExit(1)

    epochs = max(1, math.ceil(target_rows / total_hours_one_epoch))
    expected = epochs * total_hours_one_epoch
    print(f"Weather hours / epoch: {total_hours_one_epoch:,}")
    print(f"Epochs for ≥{target_rows:,} rows: {epochs} (≈{expected:,} rows)")
    print(f"Writing parquet → {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    written = 0

    try:
        tqdm = None
        try:
            from tqdm import tqdm as tqdm_cls

            tqdm = tqdm_cls(total=min(target_rows, expected), unit="row")
        except ImportError:
            pass

        for epoch in range(epochs):
            if written >= target_rows:
                break
            epoch_seed = int(rng.integers(0, 2**31 - 1))
            ep_rng = np.random.default_rng(epoch_seed)

            for csv_path in known:
                if written >= target_rows:
                    break
                df = _load_weather_frame(csv_path)
                n = len(df)
                ep = _make_facility_episodes(n, ep_rng)

                arrays = dict(
                    outdoor_temp=df["temperature_2m"].to_numpy(dtype=np.float64),
                    humidity=df["relative_humidity_2m"].to_numpy(dtype=np.float64),
                    wet_bulb=df["wet_bulb_temperature_2m"].to_numpy(dtype=np.float64),
                    year=df["year"].to_numpy(),
                    month=df["month"].to_numpy(),
                    hour=df["hour"].to_numpy(),
                    city_id=df["city_id"].to_numpy(),
                    chiller_sp=ep["chiller_sp"],
                    crah=ep["crah"],
                    tower=ep["tower"],
                    it_mw=ep["it_mw"],
                    base_cop=ep["base_cop"],
                    rated_mw=ep["rated_mw"],
                    facility_overhead_fraction_of_it=ep["facility_overhead_fraction_of_it"],
                    pump_kw_per_kw_load=ep["pump_kw_per_kw_load"],
                    crah_fan_rated_power_kw=ep["crah_fan_rated_power_kw"],
                    cooling_tower_fan_rated_power_kw=ep["cooling_tower_fan_rated_power_kw"],
                    tower_effectiveness_base=ep["tower_effectiveness_base"],
                    tower_effectiveness_per_k_depression=ep["tower_effectiveness_per_k_depression"],
                    tower_effectiveness_max=ep["tower_effectiveness_max"],
                    outdoor_cop_penalty_per_k=ep["outdoor_cop_penalty_per_k"],
                    design_outdoor_temp_c=ep["design_outdoor_temp_c"],
                )

                # Split into sub-chunks for multiprocessing or sequential
                starts = range(0, n, chunk_rows)
                for start in starts:
                    if written >= target_rows:
                        break
                    end = min(start + chunk_rows, n)
                    sl = slice(start, end)
                    payload = (
                        arrays["outdoor_temp"][sl],
                        arrays["humidity"][sl],
                        arrays["wet_bulb"][sl],
                        arrays["year"][sl],
                        arrays["month"][sl],
                        arrays["hour"][sl],
                        arrays["city_id"][sl],
                        arrays["chiller_sp"][sl],
                        arrays["crah"][sl],
                        arrays["tower"][sl],
                        arrays["it_mw"][sl],
                        arrays["base_cop"][sl],
                        arrays["rated_mw"][sl],
                        arrays["facility_overhead_fraction_of_it"][sl],
                        arrays["pump_kw_per_kw_load"][sl],
                        arrays["crah_fan_rated_power_kw"][sl],
                        arrays["cooling_tower_fan_rated_power_kw"][sl],
                        arrays["tower_effectiveness_base"][sl],
                        arrays["tower_effectiveness_per_k_depression"][sl],
                        arrays["tower_effectiveness_max"][sl],
                        arrays["outdoor_cop_penalty_per_k"][sl],
                        arrays["design_outdoor_temp_c"][sl],
                        water_price_per_liter,
                    )

                    if jobs <= 1:
                        batch = _worker_batch(payload)
                    else:
                        # Further split payload for pool (avoid tiny tasks)
                        sub = end - start
                        step = max(sub // jobs, 5000)
                        futures = []
                        with ProcessPoolExecutor(max_workers=jobs) as ex:
                            for a in range(0, sub, step):
                                b = min(a + step, sub)
                                piece = tuple(
                                    x[a:b] if isinstance(x, np.ndarray) else x
                                    for x in payload[:-1]
                                ) + (payload[-1],)
                                futures.append(ex.submit(_worker_batch, piece))
                            parts = [f.result() for f in futures]
                        batch = {
                            k: np.concatenate([p[k] for p in parts]) for k in parts[0]
                        }

                    take = min(batch["city_id"].shape[0], target_rows - written)
                    if take <= 0:
                        continue
                    batch = {k: v[:take] for k, v in batch.items()}
                    table = pa.Table.from_pydict(batch)
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                    writer.write_table(table)
                    written += take
                    if tqdm is not None:
                        tqdm.update(take)

                    gc.collect()

                del df

        if tqdm is not None:
            tqdm.close()

    finally:
        if writer is not None:
            writer.close()

    print(f"Done. Wrote {written:,} rows to {output_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weather-dir",
        type=Path,
        default=Path("data/weather"),
        help="Folder with *_hourly_*.csv per city",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/tropcool_train.parquet"),
        help="Output parquet path",
    )
    ap.add_argument(
        "--target-rows",
        type=int,
        default=8_000_000,
        help="Minimum row target (may overshoot one epoch)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=50_000,
        help="Rows per simulation batch (memory vs parallelism)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes per batch (1 = disable)",
    )
    ap.add_argument(
        "--water-price",
        type=float,
        default=0.001,
        help="Water price per liter (same currency as electricity tariff below)",
    )
    args = ap.parse_args()

    generate(
        weather_dir=args.weather_dir.resolve(),
        output_path=args.output.resolve(),
        target_rows=args.target_rows,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
        water_price_per_liter=args.water_price,
        jobs=max(1, args.jobs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
