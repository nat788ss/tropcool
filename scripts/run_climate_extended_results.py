#!/usr/bin/env python3
"""Roll baseline vs SAC across cities × climate periods → climate_results.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_data import CITY_SLUGS
PERIODS: dict[str, float] = {
    "current": 0.0,
    "2030": 0.5,
    "2040": 1.2,
    "2050": 2.0,
    "2040_stress": 2.5,
    "2050_stress": 3.0,
}


def projection_path(out_dir: Path, slug: str, period: str) -> Path:
    return out_dir / f"{slug}_{period}.csv"
from predict_year import SEQUENCE_LEN, hours_in_civil_year
from tropcool_rl import (
    load_sb3_sac_policy_fn,
    make_world_model_env,
    policy_fixed_baseline,
    rollout_episode,
)


def _clamp_episode_hours(episode_hours: int, calendar_year: int) -> int:
    """World-model rollouts need SEQUENCE_LEN context rows inside the calendar-year slice."""
    max_episode = hours_in_civil_year(calendar_year) - SEQUENCE_LEN
    if episode_hours > max_episode:
        print(
            f"WARNING: episode-hours {episode_hours} > {max_episode} for calendar_year={calendar_year} "
            f"(requires {SEQUENCE_LEN}h context); using {max_episode}",
            file=sys.stderr,
        )
        return max_episode
    return episode_hours

MW_PER_10MW_IT = 0.01  # placeholder scale for regional headline
CO2_KG_PER_MWH = 450.0  # grid intensity placeholder
VALIDATED_PORTFOLIO_SLUGS: tuple[str, ...] = tuple(s for s in CITY_SLUGS if s != "jakarta")
IPCC_FUTURE_PERIODS: tuple[str, ...] = ("2030", "2040", "2050")
STRESS_FUTURE_PERIODS: tuple[str, ...] = ("2040_stress", "2050_stress")
FUTURE_PERIODS: tuple[str, ...] = IPCC_FUTURE_PERIODS + STRESS_FUTURE_PERIODS


def _annualize(cost_window: float, hours: int) -> float:
    if hours <= 0:
        return float(cost_window)
    return float(cost_window) * (8760.0 / hours)


def _run_cell(
    slug: str,
    weather_csv: Path,
    *,
    episode_hours: int,
    calendar_year: int,
    weather_start_row: int,
    seed: int,
    model_path: str,
    device: str,
    sac_zip: Path | None,
    action_box: str | None,
) -> dict[str, float]:
    base_env = make_world_model_env(
        slug,
        max_episode_steps=episode_hours,
        calendar_year=calendar_year,
        weather_start_row=weather_start_row,
        weather_csv=weather_csv,
        model_path=model_path,
        device=device,
        action_box=action_box,
    )
    b = rollout_episode(base_env, policy_fixed_baseline, seed=seed)
    out = {
        "cost_RM": float(b["total_cost_rm"]),
        "cost_RM_annualized": _annualize(float(b["total_cost_rm"]), episode_hours),
        "mean_pue": float(b["mean_pue"]),
        "violations": int(b["violations"]),
    }
    if sac_zip and sac_zip.is_file():
        sac_pol = load_sb3_sac_policy_fn(
            sac_zip,
            eval_city_slug=slug,
            max_episode_steps=episode_hours,
            calendar_year=calendar_year,
            weather_start_row=weather_start_row,
            model_path=model_path,
            device=device,
            action_box=action_box,
        )
        sac_env = make_world_model_env(
            slug,
            max_episode_steps=episode_hours,
            calendar_year=calendar_year,
            weather_start_row=weather_start_row,
            weather_csv=weather_csv,
            model_path=model_path,
            device=device,
            action_box=action_box,
        )
        s = rollout_episode(sac_env, sac_pol, seed=seed)
        out["sac_cost_RM"] = float(s["total_cost_rm"])
        out["sac_cost_RM_annualized"] = _annualize(float(s["total_cost_rm"]), episode_hours)
        out["sac_mean_pue"] = float(s["mean_pue"])
        out["sac_violations"] = int(s["violations"])
    return out


def _policy_annual_cost(cell: dict, policy: str) -> float | None:
    data = cell.get(policy, {})
    if not data:
        return None
    val = data.get("cost_RM_annualized")
    if val is None:
        val = data.get("cost_RM")
    return float(val) if val is not None else None


def _pct_cost_increase(future: float | None, current: float | None) -> float:
    if future is None or current is None or current == 0:
        return float("nan")
    return 100.0 * (float(future) - float(current)) / float(current)


def _savings_at_weather_pct(baseline_at_period: float | None, sac_at_period: float | None) -> float:
    """SAC savings vs baseline on the same future weather (dramatic pitch metric)."""
    if baseline_at_period is None or sac_at_period is None or baseline_at_period <= 0:
        return float("nan")
    return 100.0 * (1.0 - float(sac_at_period) / float(baseline_at_period))


def _headlines_for_periods(periods: dict) -> dict[str, dict]:
    """X/Y climate headlines from annualized rollout costs (current → future)."""
    cur_cell = periods.get("current", {})
    base_cur = _policy_annual_cost(cur_cell, "baseline")
    sac_cur = _policy_annual_cost(cur_cell, "sac")
    out: dict[str, dict] = {}
    for period in FUTURE_PERIODS:
        if period not in periods:
            continue
        fut_cell = periods[period]
        base_fut = _policy_annual_cost(fut_cell, "baseline")
        sac_fut = _policy_annual_cost(fut_cell, "sac")
        if base_cur is None or base_fut is None:
            continue
        x_pct = _pct_cost_increase(base_fut, base_cur)
        y_pct = _pct_cost_increase(sac_fut, sac_cur) if sac_cur is not None and sac_fut is not None else float("nan")
        mitigation = x_pct - y_pct if np.isfinite(x_pct) and np.isfinite(y_pct) else float("nan")
        sav = _savings_at_weather_pct(base_fut, sac_fut)
        entry: dict[str, object] = {
            "X_baseline_pct": round(x_pct, 2),
            "Y_sac_pct": round(y_pct, 2) if np.isfinite(y_pct) else None,
            "mitigation_pp": round(mitigation, 2) if np.isfinite(mitigation) else None,
            "climate_mitigation_pp": round(mitigation, 2) if np.isfinite(mitigation) else None,
            "savings_at_weather_pct": round(sav, 2) if np.isfinite(sav) else None,
            "warming_c": PERIODS.get(period),
        }
        if np.isfinite(x_pct) and np.isfinite(y_pct):
            entry["pitch"] = (
                f"By {period}, baseline cooling cost rises {x_pct:.1f}%; "
                f"with TropCool SAC, the rise is only {y_pct:.1f}%."
            )
        if np.isfinite(sav):
            entry["savings_pitch"] = (
                f"At that horizon's weather, SAC cuts cooling cost {sav:.1f}% vs baseline."
            )
        out[period] = entry
    return out


def _validated_portfolio_periods(cities: dict) -> dict:
    """Mean annualized baseline/SAC costs across five validated cities (excludes Jakarta)."""
    merged: dict[str, dict] = {}
    for period in ("current",) + FUTURE_PERIODS:
        b_vals: list[float] = []
        s_vals: list[float] = []
        for slug in VALIDATED_PORTFOLIO_SLUGS:
            cell = cities.get(slug, {}).get(period, {})
            b = _policy_annual_cost(cell, "baseline")
            s = _policy_annual_cost(cell, "sac")
            if b is not None:
                b_vals.append(b)
            if s is not None:
                s_vals.append(s)
        if not b_vals:
            continue
        period_cell: dict[str, dict] = {
            "baseline": {"cost_RM_annualized": float(np.mean(b_vals))},
        }
        if s_vals:
            period_cell["sac"] = {"cost_RM_annualized": float(np.mean(s_vals))}
        merged[period] = period_cell
    return merged


def _headlines(cities: dict) -> dict:
    """Legacy flat keys (baseline vs current; SAC line uses SAC/current for compat)."""
    headlines: dict[str, dict] = {}
    for slug, periods in cities.items():
        if slug == "validated_portfolio_mean":
            continue
        block = _headlines_for_periods(periods)
        for period, entry in block.items():
            headlines[f"{slug}_{period}"] = {
                "baseline_increase_pct_vs_current": entry["X_baseline_pct"],
                "sac_increase_pct_vs_current": entry["Y_sac_pct"],
            }
    return headlines


def _headlines_v2(cities: dict) -> dict:
    out: dict[str, dict] = {}
    for slug in CITY_SLUGS:
        if slug in cities:
            out[slug] = _headlines_for_periods(cities[slug])
    out["validated_portfolio_mean"] = _headlines_for_periods(_validated_portfolio_periods(cities))
    return out


def _regional(cities: dict, episode_hours: int) -> dict:
    saved_rm = 0.0
    saved_water_l = 0.0
    for slug, periods in cities.items():
        cur = periods.get("current", {})
        if "baseline" not in cur or "sac" not in cur:
            continue
        b = cur["baseline"].get("cost_RM", 0)
        s = cur["sac"].get("sac_cost_RM", cur["sac"].get("cost_RM", 0))
        saved_rm += max(0.0, b - s)
    scale = 8760.0 / max(1, episode_hours)
    # Rough MW recovered from cooling kWh savings (very approximate demo metric).
    mw_saved = saved_rm * scale / 1e6 * MW_PER_10MW_IT
    water_saved_m3 = saved_water_l * scale / 1000.0
    mwh_saved = mw_saved * 8760.0
    return {
        "annual_cost_RM_saved_vs_baseline_sac_current": round(saved_rm * scale, 0),
        "mw_capacity_recovered_estimate": round(mw_saved, 2),
        "water_saved_m3_estimate": round(water_saved_m3, 1),
        "co2_avoided_tonnes_estimate": round(mwh_saved * CO2_KG_PER_MWH / 1000.0, 0),
        "note": "Regional figures are illustrative aggregates from model rollouts.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections-dir", type=Path, default=ROOT / "data/weather/projections")
    ap.add_argument("--rl-per-city-dir", type=Path, default=ROOT / "data/rl/per_city_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "dashboard/climate_results.json")
    ap.add_argument(
        "--episode-hours",
        type=int,
        default=None,
        help="Rollout length in hours (default 8760 full year; use --fast for 720).",
    )
    ap.add_argument(
        "--episode-len",
        type=int,
        default=None,
        help="Deprecated alias for --episode-hours.",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Dev mode: 720 h episodes instead of 8760.",
    )
    ap.add_argument("--calendar-year", type=int, default=2019)
    ap.add_argument(
        "--weather-start-row",
        type=int,
        default=0,
        help="First weather row in projection CSV (default 0 = calendar start).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model-path", type=str, default="world_model_best.pt")
    ap.add_argument("--action-box", type=str, default="baseline")
    ap.add_argument(
        "--periods",
        type=str,
        default="current,2030,2040,2050,2040_stress,2050_stress",
    )
    args = ap.parse_args()
    if args.fast:
        episode_hours = 720
    elif args.episode_hours is not None:
        episode_hours = args.episode_hours
    elif args.episode_len is not None:
        episode_hours = args.episode_len
    else:
        episode_hours = 8760
    episode_hours = _clamp_episode_hours(episode_hours, args.calendar_year)
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    rl_dir = args.rl_per_city_dir

    cities_out: dict[str, dict] = {}
    for slug in CITY_SLUGS:
        cities_out[slug] = {}
        sac_zip = rl_dir / f"{slug}.zip" if rl_dir.is_dir() else None
        for period in periods:
            if period not in PERIODS:
                print(f"skip unknown period {period}", file=sys.stderr)
                continue
            wcsv = projection_path(args.projections_dir, slug, period)
            if not wcsv.is_file():
                print(f"MISSING {wcsv} — run generate_climate_weather.py first", file=sys.stderr)
                return 2
            print(f"  {slug} / {period} …", flush=True)
            base_metrics = _run_cell(
                slug,
                wcsv,
                episode_hours=episode_hours,
                calendar_year=args.calendar_year,
                weather_start_row=args.weather_start_row,
                seed=args.seed,
                model_path=args.model_path,
                device=args.device,
                sac_zip=None,
                action_box=args.action_box if args.action_box != "full" else None,
            )
            cell = {"baseline": {k: v for k, v in base_metrics.items() if not k.startswith("sac_")}}
            if sac_zip and sac_zip.is_file():
                full = _run_cell(
                    slug,
                    wcsv,
                    episode_hours=episode_hours,
                    calendar_year=args.calendar_year,
                    weather_start_row=args.weather_start_row,
                    seed=args.seed,
                    model_path=args.model_path,
                    device=args.device,
                    sac_zip=sac_zip,
                    action_box=args.action_box if args.action_box != "full" else None,
                )
                cell["sac"] = {
                    "cost_RM": full.get("sac_cost_RM"),
                    "cost_RM_annualized": full.get("sac_cost_RM_annualized"),
                    "mean_pue": full.get("sac_mean_pue"),
                    "violations": full.get("sac_violations"),
                }
            cities_out[slug][period] = cell

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episode_hours": episode_hours,
        "calendar_year": args.calendar_year,
        "weather_start_row": args.weather_start_row,
        "periods": periods,
        "warming_c": {k: PERIODS[k] for k in periods if k in PERIODS},
        "cities": cities_out,
        "headlines": _headlines(cities_out),
        "headlines_v2": _headlines_v2(cities_out),
        "headlines_v2_note": (
            "X/Y = cost growth vs current weather; savings_at_weather_pct = 100×(1−SAC/baseline) "
            "on the same future weather. IPCC central: 2030/2040/2050; stress: 2040_stress (+2.5°C), "
            "2050_stress (+3.0°C). validated_portfolio_mean excludes Jakarta."
        ),
        "regional": _regional(cities_out, episode_hours),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
