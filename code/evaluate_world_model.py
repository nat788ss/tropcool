#!/usr/bin/env python3
"""Evaluate the trained transformer world model and generate diagnostic plots.

Outputs:
- Console table of MAE / RMSE / R² for each target on the **test** split.
- Scatter plots: predicted vs actual for each target.
- Jakarta 48-hour PUE trace: predicted vs simulator actual under fixed controls.

This script expects:
- `data/training/checkpoints/world_model_best.pt`
- `data/training/checkpoints/world_model_scalers.joblib`
- training parquet: `data/training/tropcool_train.parquet`

Requires: pip install torch pandas pyarrow scikit-learn joblib matplotlib
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dc_simulator import DataCenterSimulator
from weather_utils import wet_bulb_temperature_for_each_row
from world_model import (
    INPUT_FEATURE_NAMES,
    TARGET_NAMES,
    SEQUENCE_LEN,
    BASE_FEATURE_NAMES,
    SequenceWindowIndexDataset,
    WorldModel,
    electricity_price_rm_per_kwh,
    load_parquet_features,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_model_and_scalers(ckpt_path: Path, scaler_path: Path, device: torch.device) -> tuple[WorldModel, object, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not ckpt.get("mse_on_normalized_scale", False):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not look like a normalized run "
            "(missing mse_on_normalized_scale=True). Re-train with the updated world_model.py."
        )
    model = WorldModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bundle = joblib.load(scaler_path)
    feat_scaler = bundle["feature_scaler"]
    targ_scaler = bundle["target_scaler"]
    return model, feat_scaler, targ_scaler


def _minmax_scale(x: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    data_min = np.asarray(data_min, dtype=np.float32)
    data_max = np.asarray(data_max, dtype=np.float32)
    denom = np.maximum(data_max - data_min, 1e-12).astype(np.float32)
    return ((x.astype(np.float32) - data_min) / denom).astype(np.float32)


@torch.no_grad()
def _predict_on_loader(
    model: WorldModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_test_set(
    *,
    parquet_path: Path,
    ckpt_path: Path,
    scaler_path: Path,
    output_dir: Path,
    seed: int,
    batch_size: int,
    max_rows: int | None,
    max_windows: int | None,
    device: str,
) -> None:
    dev = torch.device(device)
    model, feat_scaler, targ_scaler = _load_model_and_scalers(ckpt_path, scaler_path, dev)

    df, X, Y = load_parquet_features(parquet_path, max_rows=max_rows)

    # Build raw (unscaled) window dataset. It already appends next-hour controls, yielding
    # windows with len(INPUT_FEATURE_NAMES) features. We'll apply manual min/max per batch.
    ds = SequenceWindowIndexDataset(df, X, Y, max_samples=max_windows, seed=seed, include_next_controls=True)
    if len(ds) < 100:
        raise RuntimeError(f"Too few windows for evaluation: {len(ds)}")

    # 80/10/10 split on windows.
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    _train, _val, test = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=g)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    x_min = getattr(feat_scaler, "data_min_", None)
    x_max = getattr(feat_scaler, "data_max_", None)
    y_min = getattr(targ_scaler, "data_min_", None)
    y_max = getattr(targ_scaler, "data_max_", None)
    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise RuntimeError("Scalers missing data_min_/data_max_ fields.")

    preds_s = []
    trues_s = []
    for xb, yb in test_loader:
        xb_np = xb.numpy()
        yb_np = yb.numpy()
        xb_s = _minmax_scale(xb_np, x_min, x_max)
        yb_s = _minmax_scale(yb_np, y_min, y_max)
        with torch.no_grad():
            yhat = model(torch.from_numpy(xb_s).to(dev)).detach().cpu().numpy()
        preds_s.append(yhat)
        trues_s.append(yb_s)
    yhat_s = np.concatenate(preds_s, axis=0)
    y_s = np.concatenate(trues_s, axis=0)

    # Inverse-transform to original units for metrics and plots.
    yhat = targ_scaler.inverse_transform(yhat_s)
    y = targ_scaler.inverse_transform(y_s)

    _ensure_dir(output_dir)

    print("\n## Test-set metrics (original units)")
    rows = []
    for j, name in enumerate(TARGET_NAMES):
        m = _metrics(y[:, j], yhat[:, j])
        rows.append((name, m["mae"], m["rmse"], m["r2"]))
    # Pretty print (no markdown table to avoid canvas requirement)
    for name, mae, rmse, r2 in rows:
        print(f"- {name:17s}  MAE={mae:,.4f}  RMSE={rmse:,.4f}  R²={r2:.4f}")

    pue_r2 = dict((n, r2) for n, _, _, r2 in rows).get("pue", float("nan"))
    print(f"\nKey check: PUE R² = {pue_r2:.4f}")

    # Scatter plots
    for j, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(6, 6))
        plt.scatter(y[:, j], yhat[:, j], s=4, alpha=0.15)
        lo = float(min(y[:, j].min(), yhat[:, j].min()))
        hi = float(max(y[:, j].max(), yhat[:, j].max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Predicted vs Actual — {name}")
        out = output_dir / f"scatter_{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()

    print(f"\nSaved scatter plots to {output_dir}")


def jakarta_48h_trace(
    *,
    weather_csv: Path,
    ckpt_path: Path,
    scaler_path: Path,
    output_dir: Path,
    start_row: int,
    device: str,
    it_load_mw: float,
    equipment_age: float,
    chiller_setpoint_c: float,
    crah_fan_speed: float,
    tower_fan_speed: float,
) -> None:
    dev = torch.device(device)
    model, feat_scaler, targ_scaler = _load_model_and_scalers(ckpt_path, scaler_path, dev)

    w = pd.read_csv(weather_csv)
    need = {"temperature_2m", "relative_humidity_2m", "wet_bulb_temperature_2m", "time"}
    miss = need - set(w.columns)
    if "wet_bulb_temperature_2m" in miss:
        # Backfill wet bulb from dry bulb + RH if missing.
        w["wet_bulb_temperature_2m"] = wet_bulb_temperature_for_each_row(
            w["temperature_2m"].to_numpy(dtype=np.float64),
            w["relative_humidity_2m"].to_numpy(dtype=np.float64),
            decimals=4,
        )
        miss = need - set(w.columns)
    if miss:
        raise ValueError(f"Weather CSV missing columns: {sorted(miss)}")
    ts = pd.to_datetime(w["time"], errors="coerce")
    w["year"] = ts.dt.year
    w["month"] = ts.dt.month
    w["hour"] = ts.dt.hour

    # Jakarta city_id in generate_data.py mapping: ("cyberjaya","singapore","jakarta",...)
    city_id = 2.0

    # Build 24 + 48 hours = 72 rows of **base** features.
    sl = slice(start_row, start_row + SEQUENCE_LEN + 48)
    ww = w.iloc[sl].copy()
    if len(ww) < SEQUENCE_LEN + 48:
        raise ValueError("Not enough weather rows for 24+48 trace at chosen start_row.")

    hour_arr = ww["hour"].to_numpy(dtype=np.float32)
    month_arr = ww["month"].to_numpy(dtype=np.float32)
    price_arr = electricity_price_rm_per_kwh(hour_arr.astype(np.int64))

    # Facility parameters: choose a fixed, representative plant (you can vary later).
    facility_params = {
        "facility_overhead_fraction_of_it": 0.30,
        "pump_kw_per_kw_load": 0.078,
        "crah_fan_rated_power_kw": 300.0,
        "cooling_tower_fan_rated_power_kw": 205.0,
        "tower_effectiveness_base": 0.42,
        "tower_effectiveness_per_k_depression": 0.038,
        "tower_effectiveness_max": 0.88,
        "outdoor_cop_penalty_per_k": 0.006,
        "design_outdoor_temp_c": 27.5,
    }

    # Build BASE_FEATURE_NAMES in order.
    base_cols: list[np.ndarray] = []
    for name in BASE_FEATURE_NAMES:
        if name == "city_id":
            base_cols.append(np.full(len(ww), city_id, dtype=np.float32))
        elif name == "outdoor_temp":
            base_cols.append(ww["temperature_2m"].to_numpy(dtype=np.float32))
        elif name == "humidity":
            base_cols.append(ww["relative_humidity_2m"].to_numpy(dtype=np.float32))
        elif name == "wet_bulb":
            base_cols.append(ww["wet_bulb_temperature_2m"].to_numpy(dtype=np.float32))
        elif name == "IT_load":
            base_cols.append(np.full(len(ww), float(it_load_mw), dtype=np.float32))
        elif name == "equipment_age":
            base_cols.append(np.full(len(ww), float(equipment_age), dtype=np.float32))
        elif name == "chiller_setpoint":
            base_cols.append(np.full(len(ww), float(chiller_setpoint_c), dtype=np.float32))
        elif name == "crah_fan_speed":
            base_cols.append(np.full(len(ww), float(crah_fan_speed), dtype=np.float32))
        elif name == "tower_fan_speed":
            base_cols.append(np.full(len(ww), float(tower_fan_speed), dtype=np.float32))
        elif name == "hour_of_day":
            base_cols.append(hour_arr.astype(np.float32))
        elif name == "month":
            base_cols.append(month_arr.astype(np.float32))
        elif name == "electricity_price":
            base_cols.append(price_arr.astype(np.float32))
        elif name in facility_params:
            base_cols.append(np.full(len(ww), float(facility_params[name]), dtype=np.float32))
        else:
            raise ValueError(f"Unhandled base feature {name}")

    X = np.column_stack(base_cols).astype(np.float32)
    x_min = feat_scaler.data_min_.astype(np.float32)
    x_max = feat_scaler.data_max_.astype(np.float32)

    sim = DataCenterSimulator(thermal_storage=None)
    actual_pue = []
    pred_pue = []
    times = ww["time"].to_numpy()

    for t in range(SEQUENCE_LEN, SEQUENCE_LEN + 48):
        x_hist = X[t - SEQUENCE_LEN : t].astype(np.float32, copy=False)
        x_next = X[t].astype(np.float32, copy=False)
        next_feats = np.array(
            [
                float(x_next[BASE_FEATURE_NAMES.index("IT_load")]),
                float(x_next[BASE_FEATURE_NAMES.index("chiller_setpoint")]),
                float(x_next[BASE_FEATURE_NAMES.index("crah_fan_speed")]),
                float(x_next[BASE_FEATURE_NAMES.index("tower_fan_speed")]),
                float(x_next[BASE_FEATURE_NAMES.index("electricity_price")]),
            ],
            dtype=np.float32,
        )
        add = np.repeat(next_feats.reshape(1, -1), SEQUENCE_LEN, axis=0)
        x_win = np.concatenate([x_hist, add], axis=1)
        x_win_s = _minmax_scale(x_win, x_min, x_max)
        xb = torch.from_numpy(x_win_s).unsqueeze(0).to(dev)
        yhat_s = model(xb).detach().cpu().numpy()
        yhat = targ_scaler.inverse_transform(yhat_s)[0]
        pred_pue.append(float(yhat[list(TARGET_NAMES).index("pue")]))

        # Simulator actual for hour t (the next-hour target for window ending at t-1)
        out_temp = float(X[t, list(INPUT_FEATURE_NAMES).index("outdoor_temp")])
        rh = float(X[t, list(INPUT_FEATURE_NAMES).index("humidity")])
        wb = float(X[t, list(INPUT_FEATURE_NAMES).index("wet_bulb")])
        r = sim.simulate_hour(
            outdoor_temp_c=out_temp,
            outdoor_relative_humidity_percent=rh,
            it_load_mw=float(it_load_mw),
            chiller_setpoint_c=float(chiller_setpoint_c),
            crah_fan_speed=float(crah_fan_speed),
            cooling_tower_fan_speed=float(tower_fan_speed),
            wet_bulb_c=wb,
        )
        actual_pue.append(float(r.pue))

    t48 = times[SEQUENCE_LEN : SEQUENCE_LEN + 48]
    _ensure_dir(output_dir)
    plt.figure(figsize=(10, 4))
    plt.plot(actual_pue, label="Simulator actual PUE", linewidth=2)
    plt.plot(pred_pue, label="World model predicted PUE", linewidth=2)
    plt.title("Jakarta 48-hour PUE trace (fixed controls + IT load)")
    plt.xlabel("Hour index (next 48 hours)")
    plt.ylabel("PUE")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out = output_dir / "jakarta_48h_pue_trace.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

    print(f"\nSaved 48-hour Jakarta PUE trace plot to {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=Path("data/training/tropcool_train.parquet"))
    ap.add_argument("--checkpoint", type=Path, default=Path("data/training/checkpoints/world_model_best.pt"))
    ap.add_argument("--scalers", type=Path, default=Path("data/training/checkpoints/world_model_scalers.joblib"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/training/eval"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--max-windows", type=int, default=60_000)

    ap.add_argument("--jakarta-weather", type=Path, default=Path("data/weather/jakarta_hourly_2004_2024.csv"))
    ap.add_argument("--jakarta-start-row", type=int, default=10_000)
    ap.add_argument("--jakarta-it-mw", type=float, default=10.0)
    ap.add_argument("--jakarta-equipment-age", type=float, default=6.0)
    ap.add_argument("--jakarta-setpoint", type=float, default=7.0)
    ap.add_argument("--jakarta-crah", type=float, default=0.7)
    ap.add_argument("--jakarta-tower", type=float, default=0.7)

    args = ap.parse_args()

    evaluate_test_set(
        parquet_path=args.parquet.resolve(),
        ckpt_path=args.checkpoint.resolve(),
        scaler_path=args.scalers.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        batch_size=args.batch_size,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        max_windows=args.max_windows if args.max_windows and args.max_windows > 0 else None,
        device=args.device,
    )

    jakarta_48h_trace(
        weather_csv=args.jakarta_weather.resolve(),
        ckpt_path=args.checkpoint.resolve(),
        scaler_path=args.scalers.resolve(),
        output_dir=args.output_dir.resolve(),
        start_row=args.jakarta_start_row,
        device=args.device,
        it_load_mw=args.jakarta_it_mw,
        equipment_age=args.jakarta_equipment_age,
        chiller_setpoint_c=args.jakarta_setpoint,
        crah_fan_speed=args.jakarta_crah,
        tower_fan_speed=args.jakarta_tower,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

