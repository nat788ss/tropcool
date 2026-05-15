#!/usr/bin/env python3
"""Transformer world model for next-hour DC cooling targets from 24 h context.

**Input window:** last **24 hours**, each timestep **12 features**:

``city_id``, ``outdoor_temp``, ``humidity``, ``wet_bulb``, ``IT_load``, ``equipment_age``,
``chiller_setpoint``, ``crah_fan_speed``, ``tower_fan_speed``, ``hour_of_day``, ``month``,
``electricity_price``

``electricity_price`` is TOU RM/kWh from ``hour`` (same rule as ``generate_data.py``).

**Output:** next-hour ``rack_inlet_temp``, ``cooling_power``, ``water_consumption``, ``pue``,
``cost`` — **MSE** over all five.

**Data:** Builds sliding windows **within each ``city_id``**, sorted by
``year → month → hour → row order`` (tie-break preserves approximate chronology from the
generator). Use the **raw** simulator parquet (e.g. ``data/training/tropcool_train.parquet``),
not row-shuffled processed splits, so sequences stay meaningful.

Requires: ``pip install torch pandas pyarrow scikit-learn joblib``

**Normalization:** ``MinMaxScaler((0,1))`` fit **only on training-window rows**; then every
parquet row of ``X`` / ``Y`` is transformed and the sliding-window dataset is rebuilt so **every
training run** serves only scaled tensors (no lazy sklearn in ``__getitem__``). Scalers saved as
``world_model_scalers.joblib`` for ``inverse_transform``. By default any stale
``world_model_best.pt`` in ``--checkpoint-dir`` is removed before training.

Default training: **Adam** ``lr=1e-4``, batch **256**, **30** epochs, ``ReduceLROnPlateau``
(×0.5 after **3** epochs without val improvement).

Example::

    python3 world_model.py train --parquet data/training/tropcool_train.parquet \\
        --epochs 30 --batch-size 256 --lr 1e-4 --device cuda
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
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("pip install torch", file=sys.stderr)
    raise

SEQUENCE_LEN = 24

# Base per-timestep feature order (12); hour_of_day aliases ``hour`` from parquet.
#
# With facility episodes, we also include plant parameters as part of the per-timestep state
# (stable across episodes). This widens the learnable PUE distribution without making the
# next-hour mapping unidentifiable.
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "city_id",
    "outdoor_temp",
    "humidity",
    "wet_bulb",
    "IT_load",
    "equipment_age",
    "chiller_setpoint",
    "crah_fan_speed",
    "tower_fan_speed",
    "hour_of_day",
    "month",
    "electricity_price",
    # Episode-stable facility parameters (from generate_data.py)
    "facility_overhead_fraction_of_it",
    "pump_kw_per_kw_load",
    "crah_fan_rated_power_kw",
    "cooling_tower_fan_rated_power_kw",
    "tower_effectiveness_base",
    "tower_effectiveness_per_k_depression",
    "tower_effectiveness_max",
    "outdoor_cop_penalty_per_k",
    "design_outdoor_temp_c",
)

# OPTION 2: include *next-hour planned actions/state* in the input so next-hour targets are
# predictable even when actions/IT load are sampled independently each hour.
#
# These are appended to every timestep (constant across the 24h window) so the model sees:
# - history window (weather + DC state + last actions) AND
# - the planned next-hour actions/state that will drive the next-hour outcome.
INPUT_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + (
    "next_IT_load",
    "next_chiller_setpoint",
    "next_crah_fan_speed",
    "next_tower_fan_speed",
    "next_electricity_price",
)

TARGET_NAMES: tuple[str, ...] = (
    "rack_inlet_temp",
    "cooling_power",
    "water_consumption",
    "pue",
    "cost",
)


def electricity_price_rm_per_kwh(hour_of_day: np.ndarray) -> np.ndarray:
    """Peak 08–22, off otherwise (matches ``generate_data._electricity_rm_per_kwh``)."""
    h = hour_of_day.astype(np.int64) % 24
    return np.where((h >= 8) & (h < 22), 0.40, 0.25).astype(np.float32)


def load_parquet_features(
    path: Path,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return dataframe with electricity_price and stacked ``X``, ``Y``.

    If ``max_rows`` is set, stop after that many rows using Parquet row-groups (avoids
    decoding the full file when you only need a prefix).
    """
    import pyarrow.parquet as pq

    cols_needed = list(
        {
            # time / grouping
            "city_id",
            "year",
            "hour",
            "month",
            # base features
            "outdoor_temp",
            "humidity",
            "wet_bulb",
            "IT_load",
            "equipment_age",
            "chiller_setpoint",
            "crah_fan_speed",
            "tower_fan_speed",
            # optional facility episode params (may not exist on older datasets)
            "facility_overhead_fraction_of_it",
            "pump_kw_per_kw_load",
            "crah_fan_rated_power_kw",
            "cooling_tower_fan_rated_power_kw",
            "tower_effectiveness_base",
            "tower_effectiveness_per_k_depression",
            "tower_effectiveness_max",
            "outdoor_cop_penalty_per_k",
            "design_outdoor_temp_c",
            # targets
            *TARGET_NAMES,
        }
    )
    pf = pq.ParquetFile(path)
    chunks: list = []
    seen = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=cols_needed)
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            if table.num_rows > remain:
                table = table.slice(0, remain)
        chunks.append(table)
        seen += table.num_rows
        if max_rows is not None and seen >= max_rows:
            break
    if not chunks:
        df = pd.read_parquet(path, columns=cols_needed)
        if max_rows is not None:
            df = df.iloc[:max_rows]
    else:
        import pyarrow as pa

        df = pa.concat_tables(chunks).to_pandas()  # type: ignore[arg-type]
    need = {
        "city_id",
        "hour",
        "month",
        "outdoor_temp",
        "humidity",
        "wet_bulb",
        "IT_load",
        "equipment_age",
        "chiller_setpoint",
        "crah_fan_speed",
        "tower_fan_speed",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Parquet missing columns: {sorted(miss)}")
    df = df.copy()
    df["_row"] = np.arange(len(df), dtype=np.int64)
    df["hour_of_day"] = df["hour"].astype(np.float32)
    df["electricity_price"] = electricity_price_rm_per_kwh(df["hour"].to_numpy())
    # Backfill missing facility episode params with NaNs (older datasets).
    for col in (
        "facility_overhead_fraction_of_it",
        "pump_kw_per_kw_load",
        "crah_fan_rated_power_kw",
        "cooling_tower_fan_rated_power_kw",
        "tower_effectiveness_base",
        "tower_effectiveness_per_k_depression",
        "tower_effectiveness_max",
        "outdoor_cop_penalty_per_k",
        "design_outdoor_temp_c",
    ):
        if col not in df.columns:
            df[col] = np.nan
    # Build [N, len(BASE_FEATURE_NAMES)] in fixed order
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
            df["facility_overhead_fraction_of_it"].astype(np.float32),
            df["pump_kw_per_kw_load"].astype(np.float32),
            df["crah_fan_rated_power_kw"].astype(np.float32),
            df["cooling_tower_fan_rated_power_kw"].astype(np.float32),
            df["tower_effectiveness_base"].astype(np.float32),
            df["tower_effectiveness_per_k_depression"].astype(np.float32),
            df["tower_effectiveness_max"].astype(np.float32),
            df["outdoor_cop_penalty_per_k"].astype(np.float32),
            df["design_outdoor_temp_c"].astype(np.float32),
        ]
    )
    Y = np.column_stack([df[c].astype(np.float32) for c in TARGET_NAMES])
    return df, X.astype(np.float32), Y.astype(np.float32)


class SequenceWindowIndexDataset(Dataset):
    """Sliding windows X[t:t+24] -> Y[t+24], indexed by linear window id (prefix sums).

    Avoids materializing millions of ``(block, offset)`` tuples. Optional ``max_samples``
    subsamples **uniformly over all windows** without enumerating them first.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        max_samples: int | None = None,
        seed: int = 42,
        include_next_controls: bool = True,
    ) -> None:
        if "year" not in df.columns:
            raise ValueError("Parquet must include 'year' for temporal sorting within city.")
        rng = np.random.default_rng(seed)
        self.X_blocks: list[np.ndarray] = []
        self.Y_blocks: list[np.ndarray] = []
        counts: list[int] = []
        self._include_next = bool(include_next_controls)

        for _cid, g in df.groupby("city_id", sort=False):
            order = np.lexsort(
                (
                    g["_row"].to_numpy(),
                    g["hour"].to_numpy(),
                    g["month"].to_numpy(),
                    g["year"].to_numpy(),
                )
            )
            idx = g.index.to_numpy()[order]
            xi = X[idx]
            yi = Y[idx]
            t = xi.shape[0]
            if t <= SEQUENCE_LEN:
                continue
            self.X_blocks.append(xi)
            self.Y_blocks.append(yi)
            counts.append(t - SEQUENCE_LEN)

        if not counts:
            self._offsets = np.array([0, 0], dtype=np.int64)
            self._lin = None
            return

        cnt = np.array(counts, dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(cnt)))
        total = int(self._offsets[-1])

        if max_samples is not None and max_samples < total:
            pick = rng.choice(total, size=max_samples, replace=False)
            self._lin = np.sort(pick.astype(np.int64))
        else:
            self._lin = None

    def __len__(self) -> int:
        total = int(self._offsets[-1]) if self._offsets.size else 0
        if total == 0:
            return 0
        if self._lin is not None:
            return int(self._lin.shape[0])
        return total

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._lin is not None:
            lin = int(self._lin[i])
        else:
            lin = int(i)
        bid = int(np.searchsorted(self._offsets, lin, side="right") - 1)
        s = lin - int(self._offsets[bid])
        x = self.X_blocks[bid][s : s + SEQUENCE_LEN]
        y = self.Y_blocks[bid][s + SEQUENCE_LEN]
        if self._include_next:
            # Next-hour controls/state are taken from the *target hour* row.
            x_next = self.X_blocks[bid][s + SEQUENCE_LEN]
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
            x = np.concatenate([x.astype(np.float32, copy=False), add], axis=1)
        return torch.from_numpy(x), torch.from_numpy(y)


def _fit_scalers_on_train_subset(
    train_subset: Subset,
    *,
    max_fit_windows: int | None,
    seed: int,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Collect training windows and fit two MinMaxScalers (features: [N*24, 12], targets: [N, 5])."""
    n = len(train_subset)
    if max_fit_windows is not None and max_fit_windows < n:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n, size=max_fit_windows, replace=False))
    else:
        pick = np.arange(n, dtype=np.int64)

    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    for j in pick:
        x, y = train_subset[int(j)]
        X_parts.append(x.numpy())
        Y_parts.append(y.numpy())
    X_st = np.stack(X_parts, axis=0)
    Y_st = np.stack(Y_parts, axis=0)
    feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    targ_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    feat_scaler.fit(X_st.reshape(-1, X_st.shape[-1]))
    targ_scaler.fit(Y_st)
    return feat_scaler, targ_scaler


def save_scalers(
    path: Path,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
) -> None:
    payload = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_data_min": feature_scaler.data_min_.astype(np.float32),
        "feature_data_max": feature_scaler.data_max_.astype(np.float32),
        "target_data_min": target_scaler.data_min_.astype(np.float32),
        "target_data_max": target_scaler.data_max_.astype(np.float32),
        "feature_names": list(INPUT_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "sequence_len": SEQUENCE_LEN,
    }
    joblib.dump(payload, path)
    meta = {
        "feature_names": list(INPUT_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "sequence_len": SEQUENCE_LEN,
        "feature_scale_note": "Apply feature_scaler to each [T, 12] window flattened as (T*12, 12) or row-wise per timestep.",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _minmax_params(data_min: np.ndarray, data_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (scale, min) for y = (x - min) * scale, safe for constant columns."""
    data_min = np.asarray(data_min, dtype=np.float32)
    data_max = np.asarray(data_max, dtype=np.float32)
    denom = np.maximum(data_max - data_min, 1e-12).astype(np.float32)
    scale = (1.0 / denom).astype(np.float32)
    return scale, data_min


class ManualMinMaxDataset(Dataset):
    """Scale windows using stored min/max arrays (no sklearn inside DataLoader)."""

    def __init__(
        self,
        base_subset: torch.utils.data.Subset,
        *,
        x_data_min: np.ndarray,
        x_data_max: np.ndarray,
        y_data_min: np.ndarray,
        y_data_max: np.ndarray,
    ) -> None:
        self.base = base_subset
        self.x_scale, self.x_min = _minmax_params(x_data_min, x_data_max)
        self.y_scale, self.y_min = _minmax_params(y_data_min, y_data_max)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[i]
        x_np = x.numpy().astype(np.float32, copy=False)
        y_np = y.numpy().astype(np.float32, copy=False)
        x_s = (x_np - self.x_min) * self.x_scale
        y_s = (y_np - self.y_min) * self.y_scale
        return torch.from_numpy(x_s), torch.from_numpy(y_s)


@torch.no_grad()
def _assert_loader_normalized(loader: DataLoader, device: torch.device, tag: str) -> None:
    xb, yb = next(iter(loader))
    xb = xb.to(device)
    yb = yb.to(device)
    xmax, xmin = float(xb.max()), float(xb.min())
    ymax, ymin = float(yb.max()), float(yb.min())
    if xmax > 1.5 or xmin < -0.5 or ymax > 1.5 or ymin < -0.5:
        raise RuntimeError(
            f"{tag}: batch out of expected normalized range "
            f"x∈[{xmin:.4g},{xmax:.4g}] y∈[{ymin:.4g},{ymax:.4g}] — scaling not applied?"
        )
    print(
        f"Sanity [{tag}]: sample batch x∈[{xmin:.4f},{xmax:.4f}] y∈[{ymin:.4f},{ymax:.4f}]",
        flush=True,
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class WorldModel(nn.Module):
    """Transformer encoder over 24 timesteps → linear head → 5 next-hour targets."""

    def __init__(
        self,
        n_features: int = len(INPUT_FEATURE_NAMES),
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_targets: int = 5,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=SEQUENCE_LEN + 16, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: torch.Tensor,
) -> float:
    model.train()
    sse, n_elem = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        # Weighted MSE on normalized targets (all in ~[0,1]).
        se = (pred - yb) ** 2
        loss = torch.sum(se * loss_weights)
        loss.backward()
        optimizer.step()
        sse += loss.item()
        n_elem += pred.numel()
    return sse / max(n_elem, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sse, n_elem = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        se = (pred - yb) ** 2
        sse += torch.sum(se).item()
        n_elem += pred.numel()
    return sse / max(n_elem, 1)


def cmd_train(args: argparse.Namespace) -> int:
    path = Path(args.parquet).resolve()
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    print(f"Loading {path} …", flush=True)
    df, X, Y = load_parquet_features(path, max_rows=args.max_rows)
    ds_full = SequenceWindowIndexDataset(
        df,
        X,
        Y,
        max_samples=args.max_samples,
        seed=args.seed,
        include_next_controls=True,
    )
    if len(ds_full) < 10:
        print("Too few sequence samples — check parquet / city coverage.", file=sys.stderr)
        return 1

    n_train = int(len(ds_full) * args.train_frac)
    n_train = max(1, min(n_train, len(ds_full) - 1))
    n_val = len(ds_full) - n_train
    train_raw, val_raw = random_split(
        ds_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    out_dir = Path(args.checkpoint_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_fresh_start:
        stale = out_dir / "world_model_best.pt"
        if stale.is_file():
            stale.unlink()
            print(f"Removed stale checkpoint {stale}", flush=True)

    print("Fitting MinMaxScalers on training windows …", flush=True)
    feat_scaler, targ_scaler = _fit_scalers_on_train_subset(
        train_raw,
        max_fit_windows=args.scale_fit_max_windows,
        seed=args.seed,
    )
    scaler_path = out_dir / "world_model_scalers.joblib"
    save_scalers(scaler_path, feat_scaler, targ_scaler)
    print(f"Saved scalers → {scaler_path}", flush=True)

    # Apply normalization via manual min/max inside the dataset wrapper so it is guaranteed
    # for every batch, without calling sklearn inside DataLoader workers.
    train_ds = ManualMinMaxDataset(
        train_raw,
        x_data_min=feat_scaler.data_min_,
        x_data_max=feat_scaler.data_max_,
        y_data_min=targ_scaler.data_min_,
        y_data_max=targ_scaler.data_max_,
    )
    val_ds = ManualMinMaxDataset(
        val_raw,
        x_data_min=feat_scaler.data_min_,
        x_data_max=feat_scaler.data_max_,
        y_data_min=targ_scaler.data_min_,
        y_data_max=targ_scaler.data_max_,
    )
    print("Wrapped train/val datasets with manual MinMax scaling.", flush=True)

    device = torch.device(args.device)
    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.plateau_epochs,
        threshold=1e-8,
        min_lr=args.min_lr,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    _assert_loader_normalized(train_loader, device, "train")

    best_val = float("inf")
    # Emphasize PUE (index 3) so it achieves high R² even though variance is small.
    w = torch.ones(len(TARGET_NAMES), dtype=torch.float32, device=device)
    w[list(TARGET_NAMES).index("pue")] = float(args.pue_loss_weight)
    w = w.view(1, -1)  # broadcast over batch

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, w)
        va = evaluate(model, val_loader, device)
        scheduler.step(va)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}  train_mse {tr:.6f}  val_mse {va:.6f}  lr {lr_now:.2e}",
            flush=True,
        )
        if va < best_val:
            best_val = va
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mse": best_val,
                "scalers_path": str(scaler_path),
                "mse_on_normalized_scale": True,
                "config": {
                    "n_features": 12,
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 4,
                    "dim_feedforward": 512,
                    "sequence_len": SEQUENCE_LEN,
                    "input_features": INPUT_FEATURE_NAMES,
                    "targets": TARGET_NAMES,
                },
            }
            torch.save(ckpt, out_dir / "world_model_best.pt")

    print(
        f"Best val MSE ≈ {best_val:.6f}; checkpoint → {out_dir / 'world_model_best.pt'}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train world model")
    tr.add_argument("--parquet", type=Path, default=Path("data/training/tropcool_train.parquet"))
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument(
        "--plateau-epochs",
        type=int,
        default=3,
        help="ReduceLROnPlateau: wait this many epochs with no val improvement before halving lr",
    )
    tr.add_argument("--min-lr", type=float, default=1e-7, dest="min_lr")
    tr.add_argument("--device", type=str, default="cpu")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--train-frac", type=float, default=0.9, dest="train_frac")
    tr.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Load only the first N rows of parquet (faster than reading full 8M-row file)",
    )
    tr.add_argument("--max-samples", type=int, default=None, help="Cap windows (debug / RAM)")
    tr.add_argument("--checkpoint-dir", type=Path, default=Path("data/training/checkpoints"))
    tr.add_argument(
        "--scale-fit-max-windows",
        type=int,
        default=None,
        help="Max training windows used to fit MinMax (default: all train windows)",
    )
    tr.add_argument(
        "--pue-loss-weight",
        type=float,
        default=25.0,
        help="Multiply PUE squared-error by this factor during training (normalized scale).",
    )
    tr.add_argument("--workers", type=int, default=0)
    tr.add_argument(
        "--no-fresh-start",
        action="store_true",
        help="Do not delete existing world_model_best.pt before training",
    )
    tr.set_defaults(func=cmd_train)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
