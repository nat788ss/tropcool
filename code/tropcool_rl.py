"""Policies and rollouts for RL / baselines on ``WorldModelEnv``.

Used by ``train_sac_tropcool.py`` and ``compare_policies.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from generate_data import CITY_SLUGS
from world_model_env import ACTION_HIGH, ACTION_LOW, WorldModelEnv

PolicyFn = Callable[[np.ndarray, WorldModelEnv], np.ndarray]


def make_world_model_env(
    city: str | int,
    *,
    max_episode_steps: int,
    calendar_year: int = 2019,
    calendar_year_choices: list[int] | None = None,
    weather_start_row: int = 0,
    random_start_row_max: int | None = None,
    training_city_slugs: list[str] | tuple[str, ...] | None = None,
    action_box: str | None = None,
    eval_aligned_fraction: float = 0.0,
    eval_start_row_choices: list[int] | tuple[int, ...] | None = None,
    model_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    weather_dir: Path | str = Path("data/weather"),
    weather_csv: str | Path | None = None,
    device: str = "cpu",
    it_load_mw: float = 10.0,
) -> WorldModelEnv:
    """Factory for the world-model MDP (short horizons recommended for SB3).

    If ``training_city_slugs`` is set, each ``reset`` samples a city and weather window
    from cached per-city CSVs (``city`` is only used for the initial ctor state).
    """
    return WorldModelEnv(
        city=city,
        weather_dir=weather_dir,
        weather_csv=weather_csv,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        max_episode_steps=max_episode_steps,
        calendar_year=calendar_year,
        calendar_year_choices=calendar_year_choices,
        weather_start_row=weather_start_row,
        random_start_row_max=random_start_row_max,
        training_city_slugs=training_city_slugs,
        action_box=action_box,
        eval_aligned_fraction=eval_aligned_fraction,
        eval_start_row_choices=eval_start_row_choices,
        device=device,
        it_load_mw=it_load_mw,
    )


def policy_fixed_baseline(_obs: np.ndarray, _env: WorldModelEnv) -> np.ndarray:
    """Constant controls (same as typical evaluation baseline)."""
    return np.array([7.0, 0.7, 0.7], dtype=np.float32)


def policy_rules_weather(_obs: np.ndarray, env: WorldModelEnv) -> np.ndarray:
    """Heuristic: more fan when hot/wet-bulb depression is small (humid)."""
    tout = env.current_ambient_temp_c()
    twb = env.current_wet_bulb_c()
    dep = max(0.5, tout - twb)
    # Tighter approach temperature when ambient is hot; lift fans when humid (low depression).
    ch = float(np.clip(6.8 + 0.06 * (tout - 26.0), 6.0, 8.5))
    if tout >= 32.0:
        crah, tow = 0.86, 0.92
    elif tout >= 29.0:
        crah, tow = 0.74, 0.78
    else:
        crah, tow = 0.62, 0.62
    if dep < 4.0:
        tow = float(min(1.0, tow + 0.06))
        crah = float(min(1.0, crah + 0.04))
    a = np.array([ch, crah, tow], dtype=np.float32)
    return np.clip(a, ACTION_LOW, ACTION_HIGH)


def rollout_episode(
    env: WorldModelEnv,
    policy: PolicyFn,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """One episode; higher ``return`` is better (reward = −cost − violation penalty)."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0
    violations = 0
    pue_vals: list[float] = []
    steps = 0
    while True:
        act = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += float(reward)
        ps = info.get("predicted_next_state") or {}
        total_cost += float(ps.get("cost", 0.0))
        violations += int(info.get("rack_violation", False))
        if "pue" in ps and np.isfinite(ps["pue"]):
            pue_vals.append(float(ps["pue"]))
        steps += 1
        if terminated or truncated:
            break
    return {
        "return": total_reward,
        "total_cost_rm": total_cost,
        "violations": violations,
        "steps": steps,
        "mean_pue": float(np.mean(pue_vals)) if pue_vals else float("nan"),
    }


def make_sb3_policy(model: Any) -> PolicyFn:
    """Wrap a Stable-Baselines3 ``predict`` for ``(obs, env)`` signature."""

    def _f(obs: np.ndarray, env: WorldModelEnv) -> np.ndarray:
        act, _ = model.predict(obs, deterministic=True)
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        return np.clip(np.asarray(act, dtype=np.float32).reshape(3,), low, high)

    return _f


def load_sb3_sac_policy_fn(
    checkpoint_zip: str | Path,
    *,
    eval_city_slug: str,
    max_episode_steps: int,
    calendar_year: int,
    weather_start_row: int,
    model_path: str | Path,
    device: str,
    vecnorm_path: str | Path | None = None,
    action_box: str | None = None,
) -> PolicyFn:
    """Load SB3 SAC + optional VecNormalize for evaluation on ``eval_city_slug``.

    ``checkpoint_zip`` should be the ``.zip`` path written by ``model.save()``; if
    ``vecnorm_path`` is None, looks for ``<stem>.vecnorm.pkl`` next to the zip.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    z = Path(checkpoint_zip)
    if not z.is_file() and z.with_suffix(".zip").is_file():
        z = z.with_suffix(".zip")
    vn = Path(vecnorm_path) if vecnorm_path is not None else z.with_suffix(".vecnorm.pkl")

    def _make() -> object:
        return make_world_model_env(
            eval_city_slug,
            max_episode_steps=max_episode_steps,
            calendar_year=calendar_year,
            weather_start_row=weather_start_row,
            model_path=model_path,
            device=device,
            action_box=action_box,
        )

    base_vec = DummyVecEnv([_make])
    if vn.is_file():
        venv = VecNormalize.load(str(vn), base_vec)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = base_vec
    model = SAC.load(str(z), env=venv)
    return make_sb3_policy(model)
