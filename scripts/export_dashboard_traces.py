#!/usr/bin/env python3
"""Export 24h baseline vs SAC traces for dashboard animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_data import CITY_SLUGS
from predict_year import default_weather_path, load_weather_csv, select_weather_rollout_slice
from tropcool_rl import (
    load_sb3_sac_policy_fn,
    make_world_model_env,
    policy_fixed_baseline,
)
from world_model import TARGET_NAMES


def hottest_day_start_row(wfull: pd.DataFrame, year: int) -> int:
    """Row index in ``year`` slice where daily max dry-bulb is highest."""
    wy = wfull[wfull["_ts"].dt.year == year].sort_values("_ts").reset_index(drop=True)
    wy = wy.assign(_date=wy["_ts"].dt.date)
    daily = wy.groupby("_date")["temperature_2m"].max()
    if daily.empty:
        return 0
    hot_date = daily.idxmax()
    first = wy.index[wy["_date"] == hot_date][0]
    # Need 24h context before the day + 24 steps
    return max(0, int(first) - 24)


def export_city_trace(
    slug: str,
    *,
    rl_dir: Path,
    weather_dir: Path,
    out_dir: Path,
    calendar_year: int,
    model_path: str,
    device: str,
    action_box: str | None,
) -> dict | None:
    wpath = default_weather_path(slug, weather_dir)
    if not wpath.is_file():
        return None
    wfull = load_weather_csv(wpath)
    start = hottest_day_start_row(wfull, calendar_year)
    hours = 24
    sac_zip = rl_dir / f"{slug}.zip"

    def _roll(policy_fn) -> list[dict]:
        env = make_world_model_env(
            slug,
            max_episode_steps=hours,
            calendar_year=calendar_year,
            weather_start_row=start,
            model_path=model_path,
            device=device,
            action_box=action_box,
        )
        obs, _ = env.reset(seed=0)
        rows: list[dict] = []
        cum_cost = 0.0
        cum_water = 0.0
        j_cost = list(TARGET_NAMES).index("cost")
        j_pue = list(TARGET_NAMES).index("pue")
        j_water = list(TARGET_NAMES).index("water_consumption")
        for h in range(hours):
            act = policy_fn(obs, env)
            obs, _r, term, trunc, info = env.step(act)
            ps = info.get("predicted_next_state") or {}
            cost = float(ps.get("cost", 0.0))
            water = float(ps.get("water_consumption", ps.get("water", 0.0)))
            if "water_consumption" in ps:
                water = float(ps["water_consumption"])
            cum_cost += cost
            cum_water += water
            rows.append(
                {
                    "hour": h,
                    "controls": [float(act[0]), float(act[1]), float(act[2])],
                    "pue": float(ps.get("pue", float("nan"))),
                    "cost_rm": cost,
                    "water_l": water,
                    "cumulative_cost_rm": cum_cost,
                    "cumulative_water_l": cum_water,
                }
            )
            if term or trunc:
                break
        return rows

    baseline_rows = _roll(policy_fixed_baseline)
    sac_rows: list[dict] = []
    if sac_zip.is_file():
        sac_pol = load_sb3_sac_policy_fn(
            sac_zip,
            eval_city_slug=slug,
            max_episode_steps=hours,
            calendar_year=calendar_year,
            weather_start_row=start,
            model_path=model_path,
            device=device,
            action_box=action_box,
        )
        sac_rows = _roll(sac_pol)

    b_total = baseline_rows[-1]["cumulative_cost_rm"] if baseline_rows else 0.0
    s_total = sac_rows[-1]["cumulative_cost_rm"] if sac_rows else b_total
    savings_pct = 100.0 * (1.0 - s_total / b_total) if b_total > 0 else 0.0

    w_slice = select_weather_rollout_slice(
        wfull, start_row=start, calendar_year=calendar_year, hours=hours
    )
    temps = w_slice.iloc[24 : 24 + hours]["temperature_2m"].tolist()

    payload = {
        "city": slug,
        "calendar_year": calendar_year,
        "weather_start_row": start,
        "outdoor_temp_c": temps,
        "baseline": baseline_rows,
        "sac": sac_rows,
        "daily_savings_pct": round(savings_pct, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}_24h.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  {out_path.name}  savings={savings_pct:.1f}%")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl-per-city-dir", type=Path, default=ROOT / "data/rl/per_city_v2")
    ap.add_argument("--weather-dir", type=Path, default=ROOT / "data/weather")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dashboard/traces")
    ap.add_argument("--calendar-year", type=int, default=2019)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model-path", type=str, default="world_model_best.pt")
    ap.add_argument("--action-box", type=str, default="baseline")
    args = ap.parse_args()
    ab = None if args.action_box == "full" else args.action_box

    for slug in CITY_SLUGS:
        export_city_trace(
            slug,
            rl_dir=args.rl_per_city_dir,
            weather_dir=args.weather_dir,
            out_dir=args.out_dir,
            calendar_year=args.calendar_year,
            model_path=args.model_path,
            device=args.device,
            action_box=ab,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
