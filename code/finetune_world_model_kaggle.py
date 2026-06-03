#!/usr/bin/env python3
"""Fine-tune the pretrained world model on real Kaggle-style DC CSVs.

Maps:
  - ``data/real/HVAC Energy Data.csv`` — resampled to **1 h** (native data is ~30 min).
    Fahrenheit → °C, RT → MW for IT load, Stull wet bulb from T/RH. Plant controls use
    defaults (chiller setpoint 7 °C, CRAH/tower speeds 0.72). Episode facility columns
    match ``DataCenterSimulator`` defaults (same as synthetic training).
  - ``data/real/cold_source_control_dataset.csv`` — hourly ambient/server workload/chiller
    and AHU usage → IT load (MW), fan speeds, humidity default 65 % when absent.

**Targets** are **simulator-consistent** labels: for each mapped row we call
``DataCenterSimulator.simulate_hour`` with that row’s weather, IT, controls, and wet
bulb, then set ``cost`` = TOU RM/kWh × cooling_power + water_price × water (same
water price default as ``generate_data.py``).

Uses **existing** ``world_model_scalers.joblib`` from pretraining so inputs/targets
stay in the same normalized space. Chronological 80/20 split **within each city_id**
(window order), then ``ConcatDataset`` for train/val.

Example::

    python3 finetune_world_model_kaggle.py \\
        --checkpoint-dir data/training/checkpoints \\
        --epochs 10 --lr 1e-5 --batch-size 128 --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dc_simulator import DataCenterSimulator
from weather_utils import wet_bulb_temperature_for_each_row
from world_model import (
    TARGET_NAMES,
    ManualMinMaxDataset,
    SequenceWindowIndexDataset,
    WorldModel,
    electricity_price_rm_per_kwh,
    train_epoch,
)

WATER_PRICE_PER_L = 0.001  # matches generate_data.py default


def _default_facility_row(n: int) -> dict[str, np.ndarray]:
    """Episode-stable facility parameters (``dc_simulator.DataCenterSimulator`` defaults)."""
    return {
        "facility_overhead_fraction_of_it": np.full(n, 0.32, dtype=np.float32),
        "pump_kw_per_kw_load": np.full(n, 0.078, dtype=np.float32),
        "crah_fan_rated_power_kw": np.full(n, 300.0, dtype=np.float32),
        "cooling_tower_fan_rated_power_kw": np.full(n, 205.0, dtype=np.float32),
        "tower_effectiveness_base": np.full(n, 0.42, dtype=np.float32),
        "tower_effectiveness_per_k_depression": np.full(n, 0.038, dtype=np.float32),
        "tower_effectiveness_max": np.full(n, 0.88, dtype=np.float32),
        "outdoor_cop_penalty_per_k": np.full(n, 0.0025, dtype=np.float32),
        "design_outdoor_temp_c": np.full(n, 27.5, dtype=np.float32),
    }


def _stack_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build X [N, len(BASE)] and Y [N, 5] aligned with ``world_model.load_parquet_features``."""
    ep = _default_facility_row(len(df))
    X = np.column_stack(
        [
            df["city_id"].astype(np.float32),
            df["outdoor_temp"].astype(np.float32),
            df["humidity"].astype(np.float32),
            df["wet_bulb"].astype(np.float32),
            df["IT_load"].astype(np.float32),
            df["equipment_age"].astype(np.float32),
            df["chiller_setpoint"].astype(np.float32),
            df["crah_fan_speed"].astype(np.float32),
            df["tower_fan_speed"].astype(np.float32),
            df["hour_of_day"].astype(np.float32),
            df["month"].astype(np.float32),
            df["electricity_price"].astype(np.float32),
            ep["facility_overhead_fraction_of_it"],
            ep["pump_kw_per_kw_load"],
            ep["crah_fan_rated_power_kw"],
            ep["cooling_tower_fan_rated_power_kw"],
            ep["tower_effectiveness_base"],
            ep["tower_effectiveness_per_k_depression"],
            ep["tower_effectiveness_max"],
            ep["outdoor_cop_penalty_per_k"],
            ep["design_outdoor_temp_c"],
        ]
    )
    Y = np.column_stack([df[c].astype(np.float32) for c in TARGET_NAMES])
    return X.astype(np.float32), Y.astype(np.float32)


def _simulate_targets(
    outdoor_c: np.ndarray,
    rh: np.ndarray,
    wb: np.ndarray,
    it_mw: np.ndarray,
    ch_sp: np.ndarray,
    crah: np.ndarray,
    tower: np.ndarray,
    hour: np.ndarray,
) -> np.ndarray:
    """Return Y columns [rack, cool_kw, water_L/h, pue, cost] using default plant."""
    sim = DataCenterSimulator(thermal_storage=None)
    n = len(outdoor_c)
    rack = np.empty(n, dtype=np.float32)
    cool = np.empty(n, dtype=np.float32)
    water = np.empty(n, dtype=np.float32)
    pue = np.empty(n, dtype=np.float32)
    cost = np.empty(n, dtype=np.float32)
    for i in range(n):
        r = sim.simulate_hour(
            float(outdoor_c[i]),
            float(rh[i]),
            float(it_mw[i]),
            float(ch_sp[i]),
            float(crah[i]),
            float(tower[i]),
            wet_bulb_c=float(wb[i]),
        )
        rack[i] = r.rack_inlet_temperature_c
        cool[i] = r.total_cooling_power_kw
        water[i] = r.water_consumption_liters
        pue[i] = r.pue if math.isfinite(r.pue) else np.nan
        tariff = float(electricity_price_rm_per_kwh(np.array([hour[i]], dtype=np.int64))[0])
        cost[i] = tariff * cool[i] + WATER_PRICE_PER_L * water[i]
    return np.column_stack([rack, cool, water, pue, cost])


def load_hvac_hourly_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["t"] = pd.to_datetime(raw["Local Time (Timezone : GMT+8h)"])
    raw = raw.set_index("t").sort_index()
    h = raw.resample("1h").mean(numeric_only=True)
    h = h.dropna(subset=["Building Load (RT)", "Outside Temperature (F)", "Humidity (%)"])
    h = h.reset_index().rename(columns={"t": "ts"})

    t_f = h["Outside Temperature (F)"].to_numpy(dtype=np.float64)
    outdoor_c = ((t_f - 32.0) * (5.0 / 9.0)).astype(np.float32)
    rh = h["Humidity (%)"].to_numpy(dtype=np.float64)
    wb = wet_bulb_temperature_for_each_row(outdoor_c.astype(np.float64), rh, decimals=4)
    wb = np.clip(wb.astype(np.float32), outdoor_c - 30.0, outdoor_c)

    # IT load: refrigeration tonnage → MW (1 RT ≈ 3.517 kW).
    rt = h["Building Load (RT)"].to_numpy(dtype=np.float64)
    it_mw = np.clip((rt * 3.517) / 1000.0, 0.05, 20.0).astype(np.float32)

    ts = pd.to_datetime(h["ts"])
    year = ts.dt.year.astype(np.int16)
    month = ts.dt.month.astype(np.int8)
    hour = ts.dt.hour.astype(np.int8)

    n = len(h)
    ch_sp = np.full(n, 7.0, dtype=np.float32)
    crah = np.full(n, 0.72, dtype=np.float32)
    tower = np.full(n, 0.72, dtype=np.float32)
    equipment_age = np.full(n, 6.0, dtype=np.float32)

    Y = _simulate_targets(
        outdoor_c.astype(np.float64),
        rh,
        wb.astype(np.float64),
        it_mw.astype(np.float64),
        ch_sp.astype(np.float64),
        crah.astype(np.float64),
        tower.astype(np.float64),
        hour.to_numpy(dtype=np.int64),
    )

    df = pd.DataFrame(
        {
            "city_id": np.zeros(n, dtype=np.int16),
            "year": year,
            "month": month.astype(np.float32),
            "hour": hour,
            "outdoor_temp": outdoor_c,
            "humidity": rh.astype(np.float32),
            "wet_bulb": wb.astype(np.float32),
            "IT_load": it_mw,
            "equipment_age": equipment_age,
            "chiller_setpoint": ch_sp,
            "crah_fan_speed": crah,
            "tower_fan_speed": tower,
            "rack_inlet_temp": Y[:, 0],
            "cooling_power": Y[:, 1],
            "water_consumption": Y[:, 2],
            "pue": Y[:, 3],
            "cost": Y[:, 4],
        }
    )
    df["hour_of_day"] = df["hour"].astype(np.float32)
    df["electricity_price"] = electricity_price_rm_per_kwh(df["hour"].to_numpy())
    df["_row"] = np.arange(len(df), dtype=np.int64)
    return df


def load_cold_source_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["ts"] = pd.to_datetime(raw["Timestamp"])
    raw = raw.sort_values("ts").dropna(subset=["Ambient_Temperature(°C)", "Server_Workload(%)"])
    n = len(raw)
    outdoor_c = raw["Ambient_Temperature(°C)"].to_numpy(dtype=np.float32)
    # Kaggle file has no RH; assume moderate humidity for wet-bulb / simulator.
    rh = np.full(n, 65.0, dtype=np.float64)
    wb = wet_bulb_temperature_for_each_row(
        outdoor_c.astype(np.float64), rh, decimals=4
    ).astype(np.float32)
    wb = np.clip(wb, outdoor_c - 30.0, outdoor_c)

    wl = raw["Server_Workload(%)"].to_numpy(dtype=np.float64)
    it_mw = np.clip(12.0 * (wl / 100.0), 0.2, 12.0).astype(np.float32)

    crah = np.clip(raw["AHU_Usage(%)"].to_numpy(dtype=np.float64) / 100.0, 0.25, 1.0).astype(
        np.float32
    )
    tower = np.clip(raw["Chiller_Usage(%)"].to_numpy(dtype=np.float64) / 100.0, 0.25, 1.0).astype(
        np.float32
    )

    ts = pd.to_datetime(raw["ts"])
    year = ts.dt.year.astype(np.int16)
    month = ts.dt.month.astype(np.int8)
    hour = ts.dt.hour.astype(np.int8)

    ch_sp = np.full(n, 7.0, dtype=np.float32)
    equipment_age = np.full(n, 5.0, dtype=np.float32)

    Y = _simulate_targets(
        outdoor_c.astype(np.float64),
        rh,
        wb.astype(np.float64),
        it_mw.astype(np.float64),
        ch_sp.astype(np.float64),
        crah.astype(np.float64),
        tower.astype(np.float64),
        hour.to_numpy(dtype=np.int64),
    )

    df = pd.DataFrame(
        {
            "city_id": np.ones(n, dtype=np.int16),
            "year": year,
            "month": month.astype(np.float32),
            "hour": hour,
            "outdoor_temp": outdoor_c,
            "humidity": rh.astype(np.float32),
            "wet_bulb": wb,
            "IT_load": it_mw,
            "equipment_age": equipment_age,
            "chiller_setpoint": ch_sp,
            "crah_fan_speed": crah,
            "tower_fan_speed": tower,
            "rack_inlet_temp": Y[:, 0],
            "cooling_power": Y[:, 1],
            "water_consumption": Y[:, 2],
            "pue": Y[:, 3],
            "cost": Y[:, 4],
        }
    )
    df["hour_of_day"] = df["hour"].astype(np.float32)
    df["electricity_price"] = electricity_price_rm_per_kwh(df["hour"].to_numpy())
    df["_row"] = np.arange(len(df), dtype=np.int64)
    return df


def _merge_kaggle_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    full = pd.concat(frames, axis=0, ignore_index=True)
    full["_row"] = np.arange(len(full), dtype=np.int64)
    return full


def _chronological_split_indices(ds_len: int, train_frac: float) -> tuple[list[int], list[int]]:
    cut = max(1, int(ds_len * train_frac))
    cut = min(cut, ds_len - 1)
    train_idx = list(range(cut))
    val_idx = list(range(cut, ds_len))
    return train_idx, val_idx


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate_physical(
    model: WorldModel,
    loader: DataLoader,
    device: torch.device,
    targ_scaler,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb).detach().cpu().numpy()
        yb_np = yb.detach().cpu().numpy()
        yhat_phys = targ_scaler.inverse_transform(yhat)
        y_phys = targ_scaler.inverse_transform(yb_np)
        preds.append(yhat_phys)
        trues.append(y_phys)
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def _print_metrics_block(title: str, y: np.ndarray, yhat: np.ndarray) -> None:
    print(f"\n## {title}")
    for j, name in enumerate(TARGET_NAMES):
        m = _metrics(y[:, j], yhat[:, j])
        print(
            f"- {name:17s}  MAE={m['mae']:,.4f}  RMSE={m['rmse']:,.4f}  R²={m['r2']:.4f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/training/checkpoints"),
    )
    ap.add_argument("--hvac-csv", type=Path, default=Path("data/real/HVAC Energy Data.csv"))
    ap.add_argument(
        "--cold-csv",
        type=Path,
        default=Path("data/real/cold_source_control_dataset.csv"),
    )
    ap.add_argument("--skip-cold", action="store_true", help="Train on HVAC hourly only.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--pue-loss-weight", type=float, default=5.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt_path = args.checkpoint_dir / "world_model_best.pt"
    scaler_path = args.checkpoint_dir / "world_model_scalers.joblib"
    if not ckpt_path.is_file():
        print(f"Missing checkpoint {ckpt_path}", file=sys.stderr)
        return 1
    if not scaler_path.is_file():
        print(f"Missing scalers {scaler_path}", file=sys.stderr)
        return 1

    frames: list[pd.DataFrame] = []
    if args.hvac_csv.is_file():
        print(f"Loading & resampling HVAC → hourly: {args.hvac_csv}", flush=True)
        frames.append(load_hvac_hourly_frame(args.hvac_csv))
        print(f"  HVAC hourly rows: {len(frames[-1])}", flush=True)
    else:
        print(f"Warning: missing {args.hvac_csv}", file=sys.stderr)

    if not args.skip_cold and args.cold_csv.is_file():
        print(f"Loading cold-source DC: {args.cold_csv}", flush=True)
        frames.append(load_cold_source_frame(args.cold_csv))
        print(f"  Cold-source rows: {len(frames[-1])}", flush=True)
    elif not args.skip_cold:
        print(f"Warning: missing {args.cold_csv}", file=sys.stderr)

    if not frames:
        print("No input frames — provide CSV paths.", file=sys.stderr)
        return 1

    merged = _merge_kaggle_frames(frames)
    print(
        f"Merged rows: {len(merged)}  cities: {sorted(merged['city_id'].unique().tolist())}",
        flush=True,
    )

    bundle = joblib.load(scaler_path)
    feat_scaler = bundle["feature_scaler"]
    targ_scaler = bundle["target_scaler"]
    x_min = feat_scaler.data_min_.astype(np.float32)
    x_max = feat_scaler.data_max_.astype(np.float32)
    y_min = targ_scaler.data_min_.astype(np.float32)
    y_max = targ_scaler.data_max_.astype(np.float32)

    # Per-city chronological split, then concat.
    parts_train: list = []
    parts_val: list = []
    total_windows = 0
    for cid in sorted(merged["city_id"].unique()):
        sub_df = merged[merged["city_id"] == cid].copy().reset_index(drop=True)
        sub_df["_row"] = np.arange(len(sub_df), dtype=np.int64)
        xi, yi = _stack_xy(sub_df)
        ds_c = SequenceWindowIndexDataset(
            sub_df,
            xi,
            yi,
            max_samples=None,
            seed=args.seed,
            include_next_controls=True,
        )
        if len(ds_c) < 2:
            continue
        tr_idx, va_idx = _chronological_split_indices(len(ds_c), args.train_frac)
        parts_train.append(
            ManualMinMaxDataset(
                Subset(ds_c, tr_idx),
                x_data_min=x_min,
                x_data_max=x_max,
                y_data_min=y_min,
                y_data_max=y_max,
            )
        )
        parts_val.append(
            ManualMinMaxDataset(
                Subset(ds_c, va_idx),
                x_data_min=x_min,
                x_data_max=x_max,
                y_data_min=y_min,
                y_data_max=y_max,
            )
        )
        total_windows += len(ds_c)
        print(f"  city_id={cid}: windows={len(ds_c)}  train={len(tr_idx)}  val={len(va_idx)}", flush=True)

    if total_windows < 50:
        print(f"Too few windows ({total_windows}). Need >50.", file=sys.stderr)
        return 1
    print(f"Total sequence windows: {total_windows}", flush=True)

    if not parts_train:
        print("No per-city windows after split.", file=sys.stderr)
        return 1

    train_ds = ConcatDataset(parts_train) if len(parts_train) > 1 else parts_train[0]
    val_ds = ConcatDataset(parts_val) if len(parts_val) > 1 else parts_val[0]

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not ckpt.get("mse_on_normalized_scale", False):
        print("Checkpoint missing mse_on_normalized_scale=True — aborting.", file=sys.stderr)
        return 1

    model = WorldModel().to(device)
    model.load_state_dict(ckpt["model"])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    w = torch.ones(len(TARGET_NAMES), dtype=torch.float32, device=device)
    w[list(TARGET_NAMES).index("pue")] = float(args.pue_loss_weight)
    w = w.view(1, -1)

    y_val_0, yhat_val_0 = evaluate_physical(model, val_loader, device, targ_scaler)
    _print_metrics_block("Validation — **before** fine-tuning (original units)", y_val_0, yhat_val_0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"\nFine-tuning {args.epochs} epochs @ lr={args.lr} …", flush=True)
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, w)
        model.eval()
        val_sse = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_sse += torch.sum((pred - yb) ** 2).item()
                val_n += pred.numel()
        val_mse = val_sse / max(val_n, 1)
        print(f"  epoch {ep:02d}/{args.epochs}  train_loss={tr_loss:.6f}  val_mse_norm={val_mse:.6f}", flush=True)

    y_val_1, yhat_val_1 = evaluate_physical(model, val_loader, device, targ_scaler)
    _print_metrics_block("Validation — **after** fine-tuning (original units)", y_val_1, yhat_val_1)

    out_ckpt = args.checkpoint_dir / "world_model_kaggle_finetuned.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "mse_on_normalized_scale": True,
            "scalers_path": str(scaler_path.resolve()),
            "config": {
                "finetune": "kaggle_dc",
                "epochs": args.epochs,
                "lr": args.lr,
                "train_frac": args.train_frac,
                "pue_loss_weight": args.pue_loss_weight,
            },
        },
        out_ckpt,
    )
    print(f"\nSaved fine-tuned weights → {out_ckpt}", flush=True)

    summary = {
        "before": {
            TARGET_NAMES[j]: _metrics(y_val_0[:, j], yhat_val_0[:, j])
            for j in range(len(TARGET_NAMES))
        },
        "after": {
            TARGET_NAMES[j]: _metrics(y_val_1[:, j], yhat_val_1[:, j])
            for j in range(len(TARGET_NAMES))
        },
    }
    summ_path = args.checkpoint_dir / "kaggle_finetune_metrics.json"
    summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics JSON → {summ_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
