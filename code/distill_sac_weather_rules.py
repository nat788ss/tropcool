#!/usr/bin/env python3
"""Distill a trained SAC policy into simple decision-tree rules (deployment-friendly).

We imitate SAC actions using **compact weather features** at the decision hour:
``[temperature_2m, relative_humidity_2m, wet_bulb_temperature_2m, hour, month]``.

This is not identical to the full 512-dim policy input, but it yields a transparent
controller that tracks SAC reasonably on the same ``WorldModelEnv`` distribution.

Example::

    python3 distill_sac_weather_rules.py \\
        --sac-zip data/rl/tropcool_sac_1m.zip \\
        --vecnorm data/rl/tropcool_sac_1m.vecnorm.pkl \\
        --out data/rl/sac_distilled_trees.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from generate_data import CITY_SLUGS
from tropcool_rl import make_world_model_env


def _weather_feats(env) -> np.ndarray:
    row = env._weather.iloc[env._t]  # noqa: SLF001 (intentional: compact features)
    return np.array(
        [
            float(row["temperature_2m"]),
            float(row["relative_humidity_2m"]),
            float(row["wet_bulb_temperature_2m"]),
            float(row["hour"]),
            float(row["month"]),
        ],
        dtype=np.float32,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sac-zip", type=Path, default=Path("data/rl/tropcool_sac_1m.zip"))
    ap.add_argument("--vecnorm", type=Path, default=Path("data/rl/tropcool_sac_1m.vecnorm.pkl"))
    ap.add_argument("--model-path", type=str, default="world_model_best.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--episode-len", type=int, default=256)
    ap.add_argument("--episodes-per-city", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/rl/sac_distilled_trees.txt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    def _make() -> object:
        return make_world_model_env(
            CITY_SLUGS[0],
            max_episode_steps=args.episode_len,
            calendar_year=2019,
            calendar_year_choices=[2017, 2018, 2019, 2020, 2021],
            random_start_row_max=50_000,
            model_path=args.model_path,
            device=args.device,
        )

    base_vec = DummyVecEnv([_make])
    vec = VecNormalize.load(str(args.vecnorm), base_vec)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(args.sac_zip), env=vec)

    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []

    for slug in CITY_SLUGS:
        for _ep in range(int(args.episodes_per_city)):
            env = make_world_model_env(
                slug,
                max_episode_steps=args.episode_len,
                calendar_year=2019,
                calendar_year_choices=[2017, 2018, 2019, 2020, 2021],
                random_start_row_max=50_000,
                model_path=args.model_path,
                device=args.device,
            )
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for _t in range(args.episode_len):
                if env._t > env._t_max:  # noqa: SLF001
                    break
                feat = _weather_feats(env)
                obs_b = obs.reshape(1, -1)
                obs_n = vec.normalize_obs(obs_b)
                act, _ = model.policy.predict(obs_n, deterministic=True)
                act = np.asarray(act, dtype=np.float32).reshape(3,)
                Xs.append(feat)
                Ys.append(act)

                a_rand = rng.uniform(low=np.array([5.0, 0.3, 0.3]), high=np.array([12.0, 1.0, 1.0]))
                obs, _r, term, trunc, _info = env.step(a_rand.astype(np.float32))
                if term or trunc:
                    break

    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)

    reg = MultiOutputRegressor(
        DecisionTreeRegressor(max_depth=int(args.max_depth), random_state=args.seed)
    )
    reg.fit(X, Y)

    feature_names = ["T_out", "RH", "Twb", "hour", "month"]
    lines: list[str] = []
    lines.append(f"Trained on N={len(X)} samples from SAC teacher.\n")
    for j, name in enumerate(["chiller_setpoint", "crah_fan_speed", "tower_fan_speed"]):
        tree = reg.estimators_[j]
        lines.append(f"=== Tree: {name} ===\n")
        lines.append(export_text(tree, feature_names=feature_names))
        lines.append("\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} (N={len(X)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
