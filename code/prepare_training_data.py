#!/usr/bin/env python3
"""Load Parquet training data, MinMax scale every column to [0, 1], split 80/10/10, save scalers.

Scalers are **fit on the training split only** (no leakage). Validation and test sets are
transformed with the same ``MinMaxScaler``.

Outputs (default ``--output-dir`` = ``data/training/processed``)::

    train.parquet
    val.parquet
    test.parquet
    minmax_scaler.joblib   # sklearn objects + column metadata

Requires: pip install pandas pyarrow scikit-learn joblib

Example::

    python3 prepare_training_data.py --input data/training/tropcool_train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS: tuple[str, ...] = (
    "city_id",
    "year",
    "month",
    "hour",
    "outdoor_temp",
    "humidity",
    "wet_bulb",
    "IT_load",
    "equipment_age",
    "chiller_setpoint",
    "crah_fan_speed",
    "tower_fan_speed",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "rack_inlet_temp",
    "cooling_power",
    "water_consumption",
    "pue",
    "cost",
)

ALL_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + TARGET_COLUMNS


def prepare(
    input_path: Path,
    output_dir: Path,
    random_state: int,
    max_rows: int | None,
) -> None:
    print(f"Loading {input_path} …")
    df = pd.read_parquet(input_path)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[:max_rows].copy()
        print(f"  (truncated to {len(df):,} rows for --max-rows)")

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        raise SystemExit(1)

    X = df[list(FEATURE_COLUMNS)]
    y = df[list(TARGET_COLUMNS)]

    # First split: 80% train, 20% holdout
    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    # Second split: holdout → 50% val, 50% test (10% each of full set)
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    cols = list(ALL_COLUMNS)
    scaler.fit(train_df[cols].values)

    def _transform(part: pd.DataFrame) -> pd.DataFrame:
        z = scaler.transform(part[cols].values)
        return pd.DataFrame(z, columns=cols, index=part.index, dtype=np.float32)

    train_s = _transform(train_df)
    val_s = _transform(val_df)
    test_s = _transform(test_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_s.to_parquet(output_dir / "train.parquet", index=False)
    val_s.to_parquet(output_dir / "val.parquet", index=False)
    test_s.to_parquet(output_dir / "test.parquet", index=False)

    payload = {
        "scaler": scaler,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "all_columns": list(ALL_COLUMNS),
        "random_state": random_state,
        "input_path": str(input_path.resolve()),
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
    }
    joblib.dump(payload, output_dir / "minmax_scaler.joblib")

    meta_path = output_dir / "split_metadata.json"
    meta_path.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "scaler"}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'train.parquet'} ({len(train_s):,} rows)")
    print(f"Wrote {output_dir / 'val.parquet'} ({len(val_s):,} rows)")
    print(f"Wrote {output_dir / 'test.parquet'} ({len(test_s):,} rows)")
    print(f"Wrote {output_dir / 'minmax_scaler.joblib'} and {meta_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/training/tropcool_train.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/processed"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If > 0, only use first N rows (debug / smoke test).",
    )
    args = ap.parse_args()
    max_rows = args.max_rows if args.max_rows > 0 else None
    prepare(
        args.input.resolve(),
        args.output_dir.resolve(),
        random_state=args.seed,
        max_rows=max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
