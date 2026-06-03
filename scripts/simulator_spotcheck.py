#!/usr/bin/env python3
"""Spot-check: WorldModelEnv predicted cost vs dc_simulator on same actions (Singapore)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from generate_data import _electricity_rm_per_kwh
from predict_year import (
    ControlStrategy,
    FacilityConfig,
    _make_simulator,
    default_weather_path,
    load_weather_csv,
    select_weather_rollout_slice,
)
from tropcool_rl import (
    load_sb3_sac_policy_fn,
    make_world_model_env,
    policy_fixed_baseline,
    rollout_episode,
)


def rollout_sim_cost(
    *,
    city: str,
    calendar_year: int,
    weather_start_row: int,
    episode_hours: int,
    policy_fn,
    seed: int,
    weather_dir: Path,
) -> tuple[float, int]:
    wfull = load_weather_csv(default_weather_path(city, weather_dir))
    w = select_weather_rollout_slice(
        wfull,
        calendar_year=calendar_year,
        start_row=weather_start_row,
        hours=episode_hours,
    )
    fac = FacilityConfig()
    strat = ControlStrategy()
    sim = _make_simulator(fac, strat)
    water_price_per_l = 0.002  # matches generate_data default
    env = make_world_model_env(
        city,
        max_episode_steps=episode_hours,
        calendar_year=calendar_year,
        weather_start_row=weather_start_row,
        weather_dir=weather_dir,
        action_box="baseline",
    )
    obs, _ = env.reset(seed=seed)
    total_sim = 0.0
    steps = 0
    for t in range(episode_hours):
        act = policy_fn(obs, env)
        row = w.iloc[24 + t]
        tout = float(row["temperature_2m"])
        rh = float(row["relative_humidity_2m"])
        wb = float(row.get("wet_bulb_temperature_2m", np.nan))
        if not np.isfinite(wb):
            from weather_utils import wet_bulb_temperature_celsius

            wb = wet_bulb_temperature_celsius(tout, rh)
        h = int(row.get("hour", t % 24))
        r = sim.simulate_hour(
            tout,
            rh,
            strat.it_load_mw,
            float(act[0]),
            float(act[1]),
            float(act[2]),
            wet_bulb_c=wb,
        )
        tariff = _electricity_rm_per_kwh(h)
        total_sim += tariff * r.total_cooling_power_kw + water_price_per_l * r.water_consumption_liters
        obs, _, term, trunc, _ = env.step(act)
        steps += 1
        if term or trunc:
            break
    return total_sim, steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="singapore")
    p.add_argument("--calendar-year", type=int, default=2019)
    p.add_argument("--weather-start-row", type=int, default=0)
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sac-zip", default="data/rl/pilot/singapore_ab.zip")
    p.add_argument("--out", default="data/rl/simulator_spotcheck.log")
    args = p.parse_args()

    weather_dir = Path("data/weather")
    lines: list[str] = []
    lines.append(
        f"Simulator spot-check: {args.city} year={args.calendar_year} "
        f"start_row={args.weather_start_row} hours={args.episode_hours} seed={args.seed}"
    )

    for name, pfn in [
        ("baseline", policy_fixed_baseline),
        (
            "SAC",
            load_sb3_sac_policy_fn(
                args.sac_zip,
                eval_city_slug=args.city,
                max_episode_steps=args.episode_hours,
                calendar_year=args.calendar_year,
                weather_start_row=args.weather_start_row,
                model_path="world_model_best.pt",
                device="cpu",
                action_box="baseline",
            ),
        ),
    ]:
        env = make_world_model_env(
            args.city,
            max_episode_steps=args.episode_hours,
            calendar_year=args.calendar_year,
            weather_start_row=args.weather_start_row,
            weather_dir=weather_dir,
            action_box="baseline",
        )
        wm = rollout_episode(env, pfn, seed=args.seed)
        sim_cost, n = rollout_sim_cost(
            city=args.city,
            calendar_year=args.calendar_year,
            weather_start_row=args.weather_start_row,
            episode_hours=args.episode_hours,
            policy_fn=pfn,
            seed=args.seed,
            weather_dir=weather_dir,
        )
        wm_cost = float(wm["total_cost_rm"])
        lines.append(
            f"{name}: world_model_cost_RM={wm_cost:,.0f}  simulator_cost_RM={sim_cost:,.0f}  steps={n}"
        )

    b_wm = float(
        rollout_episode(
            make_world_model_env(
                args.city,
                max_episode_steps=args.episode_hours,
                calendar_year=args.calendar_year,
                weather_start_row=args.weather_start_row,
                action_box="baseline",
            ),
            policy_fixed_baseline,
            seed=args.seed,
        )["total_cost_rm"]
    )
    sac_policy = load_sb3_sac_policy_fn(
        args.sac_zip,
        eval_city_slug=args.city,
        max_episode_steps=args.episode_hours,
        calendar_year=args.calendar_year,
        weather_start_row=args.weather_start_row,
        model_path="world_model_best.pt",
        device="cpu",
        action_box="baseline",
    )
    s_wm = float(
        rollout_episode(
            make_world_model_env(
                args.city,
                max_episode_steps=args.episode_hours,
                calendar_year=args.calendar_year,
                weather_start_row=args.weather_start_row,
                action_box="baseline",
            ),
            sac_policy,
            seed=args.seed,
        )["total_cost_rm"]
    )
    # Re-parse per-policy lines for simulator ordering
    sim_base = sim_sac = None
    for ln in lines:
        if ln.startswith("baseline:"):
            sim_base = float(ln.split("simulator_cost_RM=")[1].split()[0].replace(",", ""))
        if ln.startswith("SAC:"):
            sim_sac = float(ln.split("simulator_cost_RM=")[1].split()[0].replace(",", ""))
    sac_wins_wm = s_wm < b_wm
    sac_wins_sim = sim_sac is not None and sim_base is not None and sim_sac < sim_base
    lines.append("")
    lines.append(f"Direction on world model: SAC cheaper than baseline = {sac_wins_wm}")
    if sim_base is not None and sim_sac is not None:
        lines.append(f"Direction on physics simulator: SAC cheaper than baseline = {sac_wins_sim}")
        lines.append(
            f"Agreement: {'YES' if sac_wins_wm == sac_wins_sim else 'NO (surrogate vs sim disagree)'}"
        )
    lines.append(
        "Note: simulator absolute RM differs from surrogate; agreement is on cost ordering only."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
