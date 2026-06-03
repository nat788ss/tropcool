#!/usr/bin/env python3
"""Gymnasium environment that steps a trained transformer world model on real weather.

Each ``step`` applies **(chiller_setpoint, crah_fan_speed, tower_fan_speed)** for the
current hour, runs the same **24 h history + next-hour controls** forward pass as
training, and returns predicted targets. Reward is **negative predicted cost** (RM),
with an extra **−1000** when predicted ``rack_inlet_temp`` exceeds **27 °C**.

Requires: ``pip install gymnasium torch pandas numpy scikit-learn joblib``

Example::

    from world_model_env import WorldModelEnv

    env = WorldModelEnv(
        city=\"jakarta\",
        model_path=\"world_model_best.pt\",
        max_episode_steps=168,
        calendar_year=2019,
    )
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from evaluate_world_model import _load_model_and_scalers, _minmax_scale
from generate_data import CITY_SLUGS
from predict_year import (
    FacilityConfig,
    ControlStrategy,
    _build_base_feature_matrix,
    _city_id,
    _next_control_block,
    default_weather_path,
    load_weather_csv,
    select_weather_rollout_slice,
)
from world_model import BASE_FEATURE_NAMES, TARGET_NAMES, SEQUENCE_LEN

RACK_INLET_LIMIT_C = 27.0
RACK_VIOLATION_PENALTY = 1000.0

# Action bounds aligned with ``generate_data.py`` sampling ranges.
ACTION_LOW = np.array([5.0, 0.3, 0.3], dtype=np.float32)
ACTION_HIGH = np.array([12.0, 1.0, 1.0], dtype=np.float32)

# Narrow box around fixed baseline [7 °C, 0.7, 0.7] for local RL search (plan B).
BASELINE_ACTION_BOX_LOW = np.array([6.5, 0.55, 0.55], dtype=np.float32)
BASELINE_ACTION_BOX_HIGH = np.array([8.5, 0.85, 0.85], dtype=np.float32)

DEFAULT_EVAL_START_ROWS: tuple[int, ...] = (0, 2160, 4320)


def resolve_action_bounds(action_box: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) action vectors. ``baseline`` = local search around [7, 0.7, 0.7]."""
    if action_box is None or str(action_box).strip().lower() in ("", "full", "default"):
        return ACTION_LOW.copy(), ACTION_HIGH.copy()
    key = str(action_box).strip().lower()
    if key == "baseline":
        return BASELINE_ACTION_BOX_LOW.copy(), BASELINE_ACTION_BOX_HIGH.copy()
    raise ValueError(f"Unknown action_box {action_box!r}; use 'full' or 'baseline'.")

# Observation: last 24 BASE rows (flattened) + current-hour exogenous scalars (no controls).
_CUR_OBS_NAMES: tuple[str, ...] = (
    "outdoor_temp",
    "humidity",
    "wet_bulb",
    "hour_of_day",
    "month",
    "electricity_price",
    "IT_load",
    "equipment_age",
)
_CUR_OBS_IDX = np.array([BASE_FEATURE_NAMES.index(n) for n in _CUR_OBS_NAMES], dtype=np.int64)
_CONTROL_IDX = np.array(
    [
        BASE_FEATURE_NAMES.index("chiller_setpoint"),
        BASE_FEATURE_NAMES.index("crah_fan_speed"),
        BASE_FEATURE_NAMES.index("tower_fan_speed"),
    ],
    dtype=np.int64,
)

OBS_DIM = SEQUENCE_LEN * len(BASE_FEATURE_NAMES) + len(_CUR_OBS_NAMES)

_VALID_SLUGS = frozenset(CITY_SLUGS)


def _resolve_checkpoint(path: str | Path | None) -> Path:
    """Resolve world-model weights: try cwd, then ``data/training/checkpoints/`` by basename."""
    if path is None:
        p = Path("data/training/checkpoints/world_model_best.pt")
    else:
        p = Path(path)
    if p.is_file():
        return p.resolve()
    alt = Path("data/training/checkpoints") / p.name
    if alt.is_file():
        return alt.resolve()
    return p.resolve()


class WorldModelEnv(gym.Env):
    """Roll forward one hour per step using the pretrained ``WorldModel``."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        city: str | int,
        weather_csv: str | Path | None = None,
        weather_dir: str | Path = Path("data/weather"),
        model_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        device: str = "cpu",
        facility: FacilityConfig | None = None,
        it_load_mw: float = 10.0,
        initial_chiller_setpoint_c: float = 7.0,
        initial_crah_fan_speed: float = 0.7,
        initial_tower_fan_speed: float = 0.7,
        max_episode_steps: int = 8760,
        calendar_year: int | None = None,
        calendar_year_choices: list[int] | None = None,
        weather_start_row: int = 0,
        random_start_row_max: int | None = None,
        training_city_slugs: list[str] | tuple[str, ...] | None = None,
        action_box: str | None = None,
        eval_aligned_fraction: float = 0.0,
        eval_start_row_choices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if model_path is not None and checkpoint_path is not None:
            raise ValueError("Pass at most one of model_path and checkpoint_path.")
        if training_city_slugs is not None and weather_csv is not None:
            raise ValueError("Pass weather_csv=None when using training_city_slugs (per-city CSVs are implied).")
        self._training_city_slugs: tuple[str, ...] | None = None
        self._weather_cache: dict[str, Any] = {}
        if training_city_slugs:
            slugs = [str(s).strip().lower().replace(" ", "_") for s in training_city_slugs if str(s).strip()]
            bad = [s for s in slugs if s not in _VALID_SLUGS]
            if bad:
                raise ValueError(f"Unknown training_city_slugs {bad}; expected subset of {list(CITY_SLUGS)}")
            self._training_city_slugs = tuple(dict.fromkeys(slugs))  # stable unique

        self._city = city
        self._city_id = int(_city_id(city))
        self._device = torch.device(device)
        self.facility = facility or FacilityConfig()
        self.it_load_mw = float(it_load_mw)
        self._init_ch = float(initial_chiller_setpoint_c)
        self._init_crah = float(initial_crah_fan_speed)
        self._init_tow = float(initial_tower_fan_speed)
        self.max_episode_steps = int(max_episode_steps)
        self._calendar_year = calendar_year
        self._calendar_year_choices = calendar_year_choices[:] if calendar_year_choices else None
        self._weather_start_row = int(weather_start_row)
        self._random_start_row_max = None if random_start_row_max is None else int(random_start_row_max)
        self._eval_aligned_fraction = float(max(0.0, min(1.0, eval_aligned_fraction)))
        if eval_start_row_choices is None:
            self._eval_start_row_choices: tuple[int, ...] | None = (
                DEFAULT_EVAL_START_ROWS if self._eval_aligned_fraction > 0 else None
            )
        else:
            self._eval_start_row_choices = tuple(int(x) for x in eval_start_row_choices)
        self._action_low, self._action_high = resolve_action_bounds(action_box)

        ckpt = _resolve_checkpoint(checkpoint_path or model_path)
        if scaler_path is None:
            scp = ckpt.parent / "world_model_scalers.joblib"
        else:
            scp = Path(scaler_path).resolve()
        self._model, self._feat_scaler, self._targ_scaler = _load_model_and_scalers(
            ckpt,
            scp,
            self._device,
        )
        self._model.eval()
        self._x_min = self._feat_scaler.data_min_.astype(np.float32)
        self._x_max = self._feat_scaler.data_max_.astype(np.float32)
        self._n_feat = int(self._x_min.shape[0])

        wd = Path(weather_dir)
        if self._training_city_slugs is not None:
            for s in self._training_city_slugs:
                self._weather_cache[s] = load_weather_csv(default_weather_path(s, wd))
            init_slug = self._training_city_slugs[0]
            self._city = init_slug
            self._city_id = int(_city_id(init_slug))
            self._weather_full = self._weather_cache[init_slug]
        elif weather_csv is None:
            wcsv = default_weather_path(city, wd)
            self._weather_full = load_weather_csv(wcsv)
        else:
            self._weather_full = load_weather_csv(Path(weather_csv))
        self._weather = select_weather_rollout_slice(
            self._weather_full,
            start_row=self._weather_start_row,
            calendar_year=self._calendar_year,
            hours=self.max_episode_steps,
        )
        self._n_rows = len(self._weather)
        self._t_max = self._n_rows - 1

        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self._X: np.ndarray | None = None
        self._t: int = SEQUENCE_LEN
        self._episode_steps: int = 0

    def _resample_weather_window(self) -> None:
        """Reselect the weather slice (calendar year / start row) for the next episode."""
        if self._calendar_year_choices:
            # Deterministic given Gym seed via self.np_random.
            pick = int(self.np_random.integers(0, len(self._calendar_year_choices)))
            self._calendar_year = int(self._calendar_year_choices[pick])

        # Compute the maximum feasible start row for the selected calendar filter.
        if self._calendar_year is not None:
            base_len = int((self._weather_full["_ts"].dt.year == int(self._calendar_year)).sum())
        else:
            base_len = int(len(self._weather_full))
        need = int(SEQUENCE_LEN + self.max_episode_steps)
        feasible_max = max(0, base_len - need)

        if self._random_start_row_max is not None:
            hi = min(int(self._random_start_row_max), int(feasible_max))
        elif self._training_city_slugs is not None:
            hi = int(feasible_max)
        else:
            hi = None

        picked_eval = False
        if (
            self._eval_aligned_fraction > 0.0
            and self._eval_start_row_choices
            and float(self.np_random.random()) < self._eval_aligned_fraction
        ):
            feasible_eval = [int(sr) for sr in self._eval_start_row_choices if int(sr) <= feasible_max]
            if feasible_eval:
                self._weather_start_row = int(self.np_random.choice(feasible_eval))
                picked_eval = True

        if not picked_eval:
            if hi is not None:
                self._weather_start_row = int(self.np_random.integers(0, hi + 1))
            else:
                self._weather_start_row = int(min(self._weather_start_row, feasible_max))

        self._weather = select_weather_rollout_slice(
            self._weather_full,
            start_row=self._weather_start_row,
            calendar_year=self._calendar_year,
            hours=self.max_episode_steps,
        )
        self._n_rows = len(self._weather)
        self._t_max = self._n_rows - 1

    def current_ambient_temp_c(self) -> float:
        """Outdoor dry-bulb (°C) for the current decision row ``_t`` (valid after ``reset``)."""
        if self._weather is None or len(self._weather) == 0:
            return float("nan")
        ti = min(max(0, self._t), len(self._weather) - 1)
        return float(self._weather.iloc[ti]["temperature_2m"])

    def current_wet_bulb_c(self) -> float:
        if self._weather is None or len(self._weather) == 0:
            return float("nan")
        ti = min(max(0, self._t), len(self._weather) - 1)
        return float(self._weather.iloc[ti]["wet_bulb_temperature_2m"])

    def _initial_strategy(self) -> ControlStrategy:
        return ControlStrategy(
            it_load_mw=self.it_load_mw,
            chiller_setpoint_c=self._init_ch,
            crah_fan_speed=self._init_crah,
            tower_fan_speed=self._init_tow,
        )

    def _rebuild_state_matrix(self) -> None:
        strat = self._initial_strategy()
        self._X = _build_base_feature_matrix(
            self._weather,
            city_id=self._city_id,
            facility=self.facility,
            strategy=strat,
        )

    def _get_obs(self) -> np.ndarray:
        assert self._X is not None
        t = self._t
        hist = self._X[t - SEQUENCE_LEN : t].reshape(-1).astype(np.float32, copy=False)
        cur = self._X[t, _CUR_OBS_IDX].astype(np.float32, copy=False)
        return np.concatenate([hist, cur], axis=0)

    @torch.no_grad()
    def _predict(self) -> np.ndarray:
        assert self._X is not None
        t = self._t
        x_hist = self._X[t - SEQUENCE_LEN : t].astype(np.float32, copy=False)
        next_blk = _next_control_block(self._X[t])
        add = np.repeat(next_blk.reshape(1, -1), SEQUENCE_LEN, axis=0)
        full = np.concatenate([x_hist, add], axis=1)
        flat = full.reshape(-1, self._n_feat)
        scaled = _minmax_scale(flat, self._x_min, self._x_max).reshape(1, SEQUENCE_LEN, self._n_feat)
        xb = torch.from_numpy(scaled).to(self._device)
        y_s = self._model(xb).detach().cpu().numpy()
        y = self._targ_scaler.inverse_transform(y_s)[0]
        return y.astype(np.float64, copy=False)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._training_city_slugs is not None:
            slug = str(self.np_random.choice(self._training_city_slugs))
            self._city = slug
            self._city_id = int(_city_id(slug))
            self._weather_full = self._weather_cache[slug]
        if (
            self._training_city_slugs is not None
            or self._calendar_year_choices is not None
            or self._random_start_row_max is not None
            or self._eval_aligned_fraction > 0.0
        ):
            self._resample_weather_window()
        self._rebuild_state_matrix()
        self._t = SEQUENCE_LEN
        self._episode_steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._X is None:
            raise RuntimeError("Call reset() before step().")
        if self._t > self._t_max:
            raise RuntimeError("Episode has finished; call reset().")
        err_msg = self._validate_action(action)
        if err_msg:
            raise ValueError(err_msg)

        a = np.clip(np.asarray(action, dtype=np.float32).reshape(3,), self._action_low, self._action_high)
        self._X[self._t, _CONTROL_IDX] = a

        pred = self._predict()
        names = list(TARGET_NAMES)
        info_pred = {names[j]: float(pred[j]) for j in range(len(names))}

        j_cost = names.index("cost")
        j_rack = names.index("rack_inlet_temp")
        cost = float(pred[j_cost])
        rack = float(pred[j_rack])

        reward = -cost
        if rack > RACK_INLET_LIMIT_C:
            reward -= RACK_VIOLATION_PENALTY

        self._t += 1
        self._episode_steps += 1

        # Natural end: no further weather rows for a next decision.
        terminated = bool(self._t > self._t_max)
        # Time-limit (reserved for longer CSVs with ``max_episode_steps`` shorter than available).
        truncated = bool(
            not terminated and self._episode_steps >= self.max_episode_steps
        )

        if terminated or truncated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info: dict[str, Any] = {
            "predicted_next_state": info_pred,
            "rack_violation": bool(rack > RACK_INLET_LIMIT_C),
            "timestep_index": int(self._t),
        }
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _validate_action(action: np.ndarray) -> str | None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 3:
            return f"action must have shape (3,), got {np.asarray(action).shape}"
        if not np.all(np.isfinite(a)):
            return "action must be finite"
        return None


if __name__ == "__main__":
    env = WorldModelEnv(
        city="jakarta",
        weather_csv=Path("data/weather/jakarta_hourly_2004_2024.csv"),
        max_episode_steps=48,
        calendar_year=2019,
        weather_start_row=0,
        device="cpu",
    )
    obs, _ = env.reset(seed=1)
    print("obs_dim", obs.shape, "action_space", env.action_space)
    total_r = 0.0
    for k in range(5):
        a = np.array([7.0, 0.72, 0.72], dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        print(
            f"step {k}  reward={r:,.2f}  rack={info['predicted_next_state']['rack_inlet_temp']:.2f}C  "
            f"cost={info['predicted_next_state']['cost']:.2f}  term={term}",
        )
        if term:
            break
    print("sum_reward", total_r)
