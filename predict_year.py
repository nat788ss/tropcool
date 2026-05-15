#!/usr/bin/env python3
"""Roll the world model forward for one standard year (8,760 h) on real weather.

Uses the same **24 h context + next-hour controls** tensor layout as training
(``world_model.SequenceWindowIndexDataset`` / ``evaluate_world_model.jakarta_48h_trace``).

**Weather window**

- ``--start-row N`` — after optional calendar filter, start the rollout at row ``N``
  (0-based) of the sorted hourly table; requires ``N + 24 + hours`` rows available.
- ``--calendar-year YYYY`` — restrict to rows whose timestamp falls in that civil
  year (local ``time`` column). If ``--hours`` is omitted, length is **365×24** or
  **366×24** for leap years.

**Simulator sanity band**

- By default, each predicted hour is compared to ``DataCenterSimulator.simulate_hour``
  using the same weather, IT load, controls, wet bulb, and plant parameters derived
  from ``FacilityConfig`` (plus chiller rating from IT load like ``generate_data``).
  Prints MAE / RMSE per target and simulator vs model annual totals.

Example::

    python3 predict_year.py --device cpu --calendar-year 2019 --start-row 0
    python3 predict_year.py --device cpu --no-compare-simulator
"""

from __future__ import annotations

import argparse
import calendar
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from dc_simulator import DataCenterSimulator
from evaluate_world_model import _load_model_and_scalers, _minmax_scale
from generate_data import CITY_SLUGS, SLUG_TO_ID
from sklearn.metrics import mean_absolute_error, mean_squared_error
from weather_utils import wet_bulb_temperature_for_each_row
from world_model import (
    BASE_FEATURE_NAMES,
    TARGET_NAMES,
    WorldModel,
    electricity_price_rm_per_kwh,
)

CityLike = Union[int, str]

HOURS_PER_YEAR = 8760
SEQUENCE_LEN = 24
WATER_PRICE_PER_L = 0.001  # matches ``generate_data`` default


def hours_in_civil_year(year: int) -> int:
    """Hours in Jan 1 00:00 … Dec 31 23:00 for ``year`` (8784 if leap)."""
    return 366 * 24 if calendar.isleap(year) else 365 * 24


def min_weather_rows(hours: int) -> int:
    return SEQUENCE_LEN + hours


@dataclass
class FacilityConfig:
    """Episode-stable plant inputs (matches ``dc_simulator.DataCenterSimulator`` defaults)."""

    equipment_age: float = 6.0
    facility_overhead_fraction_of_it: float = 0.32
    pump_kw_per_kw_load: float = 0.078
    crah_fan_rated_power_kw: float = 300.0
    cooling_tower_fan_rated_power_kw: float = 205.0
    tower_effectiveness_base: float = 0.42
    tower_effectiveness_per_k_depression: float = 0.038
    tower_effectiveness_max: float = 0.88
    outdoor_cop_penalty_per_k: float = 0.0025
    design_outdoor_temp_c: float = 27.5


@dataclass
class ControlStrategy:
    """Constant operational setpoints for the year (baseline)."""

    it_load_mw: float = 10.0
    chiller_setpoint_c: float = 7.0
    crah_fan_speed: float = 0.7
    tower_fan_speed: float = 0.7


@dataclass
class YearRollup:
    """Aggregates over ``hours`` consecutive next-hour predictions."""

    hours: int
    annual_pue: float
    mean_hourly_pue: float
    total_cost_rm: float
    total_water_liters: float
    total_cooling_kwh: float
    total_it_kwh: float


@dataclass
class SimulatorComparison:
    """Model vs ``DataCenterSimulator`` on the same hourly inputs."""

    per_target_mae: dict[str, float]
    per_target_rmse: dict[str, float]
    pue_abs_error_p50: float
    pue_abs_error_p95: float
    sim_rollup: YearRollup


def _reference_cop_from_equipment_age(age_years: float) -> float:
    """Invert ``generate_data._equipment_age_years`` for COP in [5, 7]."""
    cop = 7.0 - float(age_years) / 6.0
    return float(max(5.0, min(7.0, cop)))


def _rated_chiller_mw(it_mw: float) -> float:
    return float(max(it_mw * 1.22, it_mw + 0.5))


def _make_simulator(facility: FacilityConfig, strategy: ControlStrategy) -> DataCenterSimulator:
    cop = _reference_cop_from_equipment_age(facility.equipment_age)
    rated = _rated_chiller_mw(strategy.it_load_mw)
    return DataCenterSimulator(
        chiller_rated_capacity_mw=rated,
        reference_cop=cop,
        design_outdoor_temp_c=facility.design_outdoor_temp_c,
        outdoor_cop_penalty_per_k=facility.outdoor_cop_penalty_per_k,
        crah_fan_rated_power_kw=facility.crah_fan_rated_power_kw,
        cooling_tower_fan_rated_power_kw=facility.cooling_tower_fan_rated_power_kw,
        pump_kw_per_kw_load=facility.pump_kw_per_kw_load,
        facility_overhead_fraction_of_it=facility.facility_overhead_fraction_of_it,
        tower_effectiveness_base=facility.tower_effectiveness_base,
        tower_effectiveness_per_k_depression=facility.tower_effectiveness_per_k_depression,
        tower_effectiveness_max=facility.tower_effectiveness_max,
        thermal_storage=None,
    )


def _rollup_from_hourly_preds(
    preds: np.ndarray,
    strategy: ControlStrategy,
    *,
    hours: int,
) -> YearRollup:
    """``preds`` shape ``[hours, len(TARGET_NAMES)]`` in original units."""
    j_pue = list(TARGET_NAMES).index("pue")
    j_cost = list(TARGET_NAMES).index("cost")
    j_water = list(TARGET_NAMES).index("water_consumption")
    j_cool = list(TARGET_NAMES).index("cooling_power")
    pue_h = preds[:, j_pue]
    mean_hourly_pue = float(np.nanmean(pue_h))
    it_kw = strategy.it_load_mw * 1000.0
    total_it_kwh = it_kw * hours
    cool_kw_h = preds[:, j_cool]
    total_cooling_kwh = float(np.nansum(cool_kw_h))
    annual_pue = (
        (total_it_kwh + total_cooling_kwh) / total_it_kwh if total_it_kwh > 0 else float("nan")
    )
    return YearRollup(
        hours=hours,
        annual_pue=annual_pue,
        mean_hourly_pue=mean_hourly_pue,
        total_cost_rm=float(np.nansum(preds[:, j_cost])),
        total_water_liters=float(np.nansum(preds[:, j_water])),
        total_cooling_kwh=total_cooling_kwh,
        total_it_kwh=total_it_kwh,
    )


def select_weather_rollout_slice(
    wfull: pd.DataFrame,
    *,
    start_row: int = 0,
    calendar_year: int | None = None,
    hours: int,
) -> pd.DataFrame:
    """Return ``24 + hours`` contiguous hourly rows for a rollout.

    If ``calendar_year`` is set, keep only timestamps in that year (sorted), then
    apply ``start_row`` within that subset.
    """
    if "_ts" not in wfull.columns:
        raise ValueError("weather frame must come from load_weather_csv (needs _ts)")
    if start_row < 0:
        raise ValueError("start_row must be >= 0")
    need = min_weather_rows(hours)
    if calendar_year is not None:
        wy = wfull[wfull["_ts"].dt.year == int(calendar_year)].copy()
        wy = wy.sort_values("_ts").reset_index(drop=True)
        base = wy
    else:
        base = wfull
    if start_row + need > len(base):
        raise ValueError(
            f"Need at least {need} rows from start_row={start_row}, have {len(base)} "
            f"(calendar_year={calendar_year})"
        )
    return base.iloc[start_row : start_row + need].copy().reset_index(drop=True)


def _simulate_rollout_targets(
    w_slice: pd.DataFrame,
    facility: FacilityConfig,
    strategy: ControlStrategy,
    *,
    hours: int,
) -> np.ndarray:
    """Simulator targets for each predicted hour (rows ``SEQUENCE_LEN`` … inclusive)."""
    sim = _make_simulator(facility, strategy)
    out = np.empty((hours, len(TARGET_NAMES)), dtype=np.float64)
    j = 0
    for t in range(SEQUENCE_LEN, SEQUENCE_LEN + hours):
        row = w_slice.iloc[t]
        out_t = float(row["temperature_2m"])
        rh = float(row["relative_humidity_2m"])
        wb = float(row["wet_bulb_temperature_2m"])
        h = int(row["hour"])
        r = sim.simulate_hour(
            out_t,
            rh,
            strategy.it_load_mw,
            strategy.chiller_setpoint_c,
            strategy.crah_fan_speed,
            strategy.tower_fan_speed,
            wet_bulb_c=wb,
        )
        tariff = float(electricity_price_rm_per_kwh(np.array([h], dtype=np.int64))[0])
        cost = tariff * r.total_cooling_power_kw + WATER_PRICE_PER_L * r.water_consumption_liters
        out[j, :] = [
            r.rack_inlet_temperature_c,
            r.total_cooling_power_kw,
            r.water_consumption_liters,
            r.pue if math.isfinite(r.pue) else np.nan,
            cost,
        ]
        j += 1
    return out


def _compare_model_to_sim(
    preds: np.ndarray,
    sim_y: np.ndarray,
    strategy: ControlStrategy,
    *,
    hours: int,
) -> SimulatorComparison:
    mae_d: dict[str, float] = {}
    rmse_d: dict[str, float] = {}
    for j, name in enumerate(TARGET_NAMES):
        a = np.asarray(preds[:, j], dtype=np.float64)
        b = np.asarray(sim_y[:, j], dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            mae_d[name] = float("nan")
            rmse_d[name] = float("nan")
            continue
        mae_d[name] = float(mean_absolute_error(a[mask], b[mask]))
        rmse_d[name] = float(math.sqrt(mean_squared_error(a[mask], b[mask])))
    jp = list(TARGET_NAMES).index("pue")
    d = np.abs(preds[:, jp] - sim_y[:, jp])
    d = d[np.isfinite(d)]
    p50 = float(np.percentile(d, 50)) if d.size else float("nan")
    p95 = float(np.percentile(d, 95)) if d.size else float("nan")
    sim_rollup = _rollup_from_hourly_preds(sim_y, strategy, hours=hours)
    return SimulatorComparison(
        per_target_mae=mae_d,
        per_target_rmse=rmse_d,
        pue_abs_error_p50=p50,
        pue_abs_error_p95=p95,
        sim_rollup=sim_rollup,
    )


def _city_id(city: CityLike) -> int:
    if isinstance(city, int):
        if not (0 <= city < len(CITY_SLUGS)):
            raise ValueError(f"city_id must be in 0..{len(CITY_SLUGS)-1}, got {city}")
        return city
    key = str(city).lower().replace(" ", "_").replace("-", "_")
    if key not in SLUG_TO_ID:
        raise ValueError(f"Unknown city slug {city!r}; expected one of {list(SLUG_TO_ID)}")
    return SLUG_TO_ID[key]


def default_weather_path(city: CityLike, weather_dir: Path) -> Path:
    slug = CITY_SLUGS[_city_id(city)]
    return weather_dir / f"{slug}_hourly_2004_2024.csv"


def load_weather_csv(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    if "time" not in w.columns:
        raise ValueError(f"{path} missing 'time' column")
    if "wet_bulb_temperature_2m" not in w.columns:
        if not {"temperature_2m", "relative_humidity_2m"}.issubset(w.columns):
            raise ValueError(f"{path} needs temperature_2m, relative_humidity_2m or wet_bulb")
        w = w.copy()
        w["wet_bulb_temperature_2m"] = wet_bulb_temperature_for_each_row(
            w["temperature_2m"].to_numpy(dtype=np.float64),
            w["relative_humidity_2m"].to_numpy(dtype=np.float64),
            decimals=4,
        )
    ts = pd.to_datetime(w["time"], errors="coerce")
    w = w.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    w["year"] = w["_ts"].dt.year.astype(np.int16)
    w["month"] = w["_ts"].dt.month.astype(np.int8)
    w["hour"] = w["_ts"].dt.hour.astype(np.int8)
    return w


def _build_base_feature_matrix(
    w: pd.DataFrame,
    *,
    city_id: int,
    facility: FacilityConfig,
    strategy: ControlStrategy,
) -> np.ndarray:
    """Per-hour BASE features [N, len(BASE_FEATURE_NAMES)], aligned with ``w`` rows."""
    n = len(w)
    hour_f = w["hour"].to_numpy(dtype=np.float32)
    month_f = w["month"].to_numpy(dtype=np.float32)
    price = electricity_price_rm_per_kwh(hour_f.astype(np.int64))

    fp = facility
    cols: list[np.ndarray] = []
    for name in BASE_FEATURE_NAMES:
        if name == "city_id":
            cols.append(np.full(n, float(city_id), dtype=np.float32))
        elif name == "outdoor_temp":
            cols.append(w["temperature_2m"].to_numpy(dtype=np.float32))
        elif name == "humidity":
            cols.append(w["relative_humidity_2m"].to_numpy(dtype=np.float32))
        elif name == "wet_bulb":
            cols.append(w["wet_bulb_temperature_2m"].to_numpy(dtype=np.float32))
        elif name == "IT_load":
            cols.append(np.full(n, strategy.it_load_mw, dtype=np.float32))
        elif name == "equipment_age":
            cols.append(np.full(n, fp.equipment_age, dtype=np.float32))
        elif name == "chiller_setpoint":
            cols.append(np.full(n, strategy.chiller_setpoint_c, dtype=np.float32))
        elif name == "crah_fan_speed":
            cols.append(np.full(n, strategy.crah_fan_speed, dtype=np.float32))
        elif name == "tower_fan_speed":
            cols.append(np.full(n, strategy.tower_fan_speed, dtype=np.float32))
        elif name == "hour_of_day":
            cols.append(hour_f)
        elif name == "month":
            cols.append(month_f)
        elif name == "electricity_price":
            cols.append(price.astype(np.float32))
        elif name == "facility_overhead_fraction_of_it":
            cols.append(np.full(n, fp.facility_overhead_fraction_of_it, dtype=np.float32))
        elif name == "pump_kw_per_kw_load":
            cols.append(np.full(n, fp.pump_kw_per_kw_load, dtype=np.float32))
        elif name == "crah_fan_rated_power_kw":
            cols.append(np.full(n, fp.crah_fan_rated_power_kw, dtype=np.float32))
        elif name == "cooling_tower_fan_rated_power_kw":
            cols.append(np.full(n, fp.cooling_tower_fan_rated_power_kw, dtype=np.float32))
        elif name == "tower_effectiveness_base":
            cols.append(np.full(n, fp.tower_effectiveness_base, dtype=np.float32))
        elif name == "tower_effectiveness_per_k_depression":
            cols.append(np.full(n, fp.tower_effectiveness_per_k_depression, dtype=np.float32))
        elif name == "tower_effectiveness_max":
            cols.append(np.full(n, fp.tower_effectiveness_max, dtype=np.float32))
        elif name == "outdoor_cop_penalty_per_k":
            cols.append(np.full(n, fp.outdoor_cop_penalty_per_k, dtype=np.float32))
        elif name == "design_outdoor_temp_c":
            cols.append(np.full(n, fp.design_outdoor_temp_c, dtype=np.float32))
        else:
            raise ValueError(f"Unhandled BASE feature {name}")
    return np.column_stack(cols).astype(np.float32)


def _next_control_block(X_row: np.ndarray) -> np.ndarray:
    """Five next-hour scalars from one BASE row (same order as ``SequenceWindowIndexDataset``)."""
    return np.array(
        [
            float(X_row[BASE_FEATURE_NAMES.index("IT_load")]),
            float(X_row[BASE_FEATURE_NAMES.index("chiller_setpoint")]),
            float(X_row[BASE_FEATURE_NAMES.index("crah_fan_speed")]),
            float(X_row[BASE_FEATURE_NAMES.index("tower_fan_speed")]),
            float(X_row[BASE_FEATURE_NAMES.index("electricity_price")]),
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def predict_year(
    city: CityLike,
    facility: FacilityConfig,
    strategy: ControlStrategy,
    weather: pd.DataFrame,
    *,
    model: WorldModel,
    feat_scaler,
    targ_scaler,
    device: torch.device,
    hours: int = HOURS_PER_YEAR,
    batch_size: int = 512,
    start_row: int = 0,
    calendar_year: int | None = None,
    compare_simulator: bool = True,
) -> tuple[YearRollup, SimulatorComparison | None]:
    """Roll the world model forward for ``hours`` steps after a 24 h warm-up.

    ``weather`` must be the **full** sorted CSV frame from ``load_weather_csv`` (includes
    ``_ts``). Rows are selected with ``select_weather_rollout_slice`` using
    ``calendar_year`` and ``start_row``, then the first ``24 + hours`` rows of that
    slice define the rollout (same convention as training windows).

    Returns ``(model_rollup, sim_comparison)``. ``sim_comparison`` is ``None`` when
    ``compare_simulator`` is false.
    """
    w = select_weather_rollout_slice(
        weather,
        start_row=start_row,
        calendar_year=calendar_year,
        hours=hours,
    )
    cid = _city_id(city)
    X = _build_base_feature_matrix(w, city_id=cid, facility=facility, strategy=strategy)

    x_min = feat_scaler.data_min_.astype(np.float32)
    x_max = feat_scaler.data_max_.astype(np.float32)

    n_feat = x_min.shape[0]
    preds = np.empty((hours, len(TARGET_NAMES)), dtype=np.float64)
    model.eval()

    for start in range(0, hours, batch_size):
        end = min(start + batch_size, hours)
        b = end - start
        t_first = SEQUENCE_LEN + start
        wins = np.zeros((b, SEQUENCE_LEN, X.shape[1]), dtype=np.float32)
        for i in range(b):
            t = t_first + i
            wins[i] = X[t - SEQUENCE_LEN : t]
        next_rows = X[t_first : t_first + b]
        next_blk = np.stack([_next_control_block(next_rows[i]) for i in range(b)], axis=0)
        add = np.repeat(next_blk[:, np.newaxis, :], SEQUENCE_LEN, axis=1)
        full = np.concatenate([wins, add], axis=2)
        if full.shape[2] != n_feat:
            raise RuntimeError(
                f"Feature dim mismatch: window has {full.shape[2]} cols, scaler expects {n_feat}"
            )
        flat = full.reshape(-1, n_feat)
        scaled = _minmax_scale(flat, x_min, x_max).reshape(b, SEQUENCE_LEN, n_feat)
        xb = torch.from_numpy(scaled).to(device)
        yhat_s = model(xb).detach().cpu().numpy()
        yhat = targ_scaler.inverse_transform(yhat_s)
        preds[start:end] = yhat

    model_rollup = _rollup_from_hourly_preds(preds, strategy, hours=hours)

    if not compare_simulator:
        return model_rollup, None

    sim_y = _simulate_rollout_targets(w, facility, strategy, hours=hours)
    comparison = _compare_model_to_sim(preds, sim_y, strategy, hours=hours)
    return model_rollup, comparison


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weather-dir", type=Path, default=Path("data/weather"))
    ap.add_argument("--checkpoint", type=Path, default=Path("data/training/checkpoints/world_model_best.pt"))
    ap.add_argument("--scalers", type=Path, default=Path("data/training/checkpoints/world_model_scalers.joblib"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Rollout length (default: 8760, or 365/366×24 if --calendar-year is set).",
    )
    ap.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Row offset within the (calendar-filtered) sorted hourly table.",
    )
    ap.add_argument(
        "--calendar-year",
        type=int,
        default=None,
        metavar="YYYY",
        help="Use only timestamps in this civil year; default hours match that year.",
    )
    ap.add_argument(
        "--no-compare-simulator",
        action="store_true",
        help="Skip hourly DataCenterSimulator comparison (faster).",
    )
    args = ap.parse_args()

    hours = args.hours
    if hours is None:
        hours = hours_in_civil_year(args.calendar_year) if args.calendar_year is not None else HOURS_PER_YEAR

    device = torch.device(args.device)
    model, feat_scaler, targ_scaler = _load_model_and_scalers(
        args.checkpoint.resolve(),
        args.scalers.resolve(),
        device,
    )

    facility = FacilityConfig()
    strategy = ControlStrategy()
    compare = not args.no_compare_simulator

    print(
        "Baseline strategy: "
        f"IT={strategy.it_load_mw} MW, setpoint={strategy.chiller_setpoint_c} °C, "
        f"CRAH={strategy.crah_fan_speed}, tower={strategy.tower_fan_speed}",
        flush=True,
    )
    print(
        f"Window: hours={hours}  start_row={args.start_row}  "
        f"calendar_year={args.calendar_year}  compare_simulator={compare}",
        flush=True,
    )
    print("", flush=True)

    for slug in CITY_SLUGS:
        path = args.weather_dir / f"{slug}_hourly_2004_2024.csv"
        if not path.is_file():
            print(f"Missing weather file: {path}", file=sys.stderr)
            return 1
        wfull = load_weather_csv(path)
        try:
            select_weather_rollout_slice(
                wfull,
                start_row=args.start_row,
                calendar_year=args.calendar_year,
                hours=hours,
            )
        except ValueError as e:
            print(f"{slug}: {e}", file=sys.stderr)
            return 1

        r, cmp = predict_year(
            slug,
            facility,
            strategy,
            wfull,
            model=model,
            feat_scaler=feat_scaler,
            targ_scaler=targ_scaler,
            device=device,
            hours=hours,
            batch_size=args.batch_size,
            start_row=args.start_row,
            calendar_year=args.calendar_year,
            compare_simulator=compare,
        )
        print(
            f"{slug:20s}  [model] PUE_year={r.annual_pue:.3f}  PUE_mean_h={r.mean_hourly_pue:.3f}  "
            f"cost_RM={r.total_cost_rm:,.0f}  water_ML={r.total_water_liters/1e6:.2f}  "
            f"cool_GWh={r.total_cooling_kwh/1e6:.3f}",
            flush=True,
        )
        if cmp is not None:
            s = cmp.sim_rollup
            print(
                f"{'':20s}  [ sim ] PUE_year={s.annual_pue:.3f}  PUE_mean_h={s.mean_hourly_pue:.3f}  "
                f"cost_RM={s.total_cost_rm:,.0f}  water_ML={s.total_water_liters/1e6:.2f}  "
                f"cool_GWh={s.total_cooling_kwh/1e6:.3f}",
                flush=True,
            )
            print(
                f"{'':20s}  model vs sim — PUE |Δ| p50={cmp.pue_abs_error_p50:.4f}  p95={cmp.pue_abs_error_p95:.4f}",
                flush=True,
            )
            for name in TARGET_NAMES:
                print(
                    f"{'':20s}    {name:18s}  MAE={cmp.per_target_mae[name]:,.5g}  "
                    f"RMSE={cmp.per_target_rmse[name]:,.5g}",
                    flush=True,
                )
        print("", flush=True)

    print(
        "Notes: [sim] uses ``DataCenterSimulator`` with COP from equipment_age, chiller "
        "rating max(IT×1.22, IT+0.5) MW, and your ``FacilityConfig``. Hourly cost is "
        "TOU×cooling kW + water price × liters (same as training).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
