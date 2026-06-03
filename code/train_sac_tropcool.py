#!/usr/bin/env python3
"""Train SAC on ``WorldModelEnv`` (transformer rollout; no physics simulator).

Example::

    python3 train_sac_tropcool.py --timesteps 500000 --save data/rl/tropcool_sac

Per-city checkpoints (one SAC + VecNormalize per city, for strict per-city baseline
beat under ``compare_policies.py --rl-per-city-dir``)::

    python3 train_sac_tropcool.py --each-city --out-dir data/rl/per_city \\
        --timesteps 1000000 --vecnormalize --episode-len 256 --n-envs 4 \\
        --calendar-year-choices 2018,2019,2020 --random-start-row-max 8760

Resume after an interrupted run (skip cities that already have ``<slug>.zip``,
and ``<slug>.vecnorm.pkl`` when ``--vecnormalize``)::

    python3 train_sac_tropcool.py --each-city --out-dir data/rl/per_city --resume ...

Singapore pilot (eval-aligned sampling + baseline action box, no reward norm)::

    python3 train_sac_tropcool.py --city singapore --save data/rl/pilot/singapore_ab \\
        --timesteps 500000 --episode-len 512 --n-envs 4 --vecnormalize --no-norm-reward \\
        --calendar-year-choices 2018,2019,2020 --random-start-row-max 8760 \\
        --eval-aligned-fraction 0.4 --action-box baseline

Requires: ``pip install stable_baselines3 gymnasium``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from generate_data import CITY_SLUGS
from tropcool_rl import make_world_model_env


def train_sac_once(
    *,
    city: str,
    save_prefix: Path,
    timesteps: int,
    episode_len: int,
    n_envs: int,
    calendar_year: int,
    calendar_year_choices: list[int] | None,
    random_start_row_max: int | None,
    train_slugs: list[str] | None,
    model_path: str,
    device: str,
    vecnormalize: bool,
    norm_reward: bool,
    action_box: str | None,
    eval_aligned_fraction: float,
    eval_start_row_choices: list[int] | None,
    learning_rate: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    seed: int,
    verbose: int,
) -> None:
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    vecnorm_path = save_prefix.with_suffix(".vecnorm.pkl")
    init_city = train_slugs[0] if train_slugs else city

    def _make() -> object:
        return make_world_model_env(
            init_city,
            max_episode_steps=episode_len,
            calendar_year=calendar_year,
            calendar_year_choices=calendar_year_choices,
            random_start_row_max=random_start_row_max,
            training_city_slugs=train_slugs,
            model_path=model_path,
            device=device,
            action_box=action_box,
            eval_aligned_fraction=eval_aligned_fraction,
            eval_start_row_choices=eval_start_row_choices,
        )

    n_envs = max(1, int(n_envs))
    vec = DummyVecEnv([_make for _ in range(n_envs)])
    if vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)
    model = SAC(
        "MlpPolicy",
        vec,
        verbose=verbose,
        seed=seed,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_freq=1,
        gradient_steps=1,
        learning_rate=learning_rate,
        tensorboard_log=None,
    )
    try:
        model.learn(total_timesteps=timesteps, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(save_prefix))
    if vecnormalize:
        vec.save(str(vecnorm_path))
    print(f"Saved → {save_prefix}.zip", flush=True)
    if vecnormalize:
        print(f"Saved VecNormalize stats → {vecnorm_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--city",
        type=str,
        default="cyberjaya",
        help="Training city slug (ignored when --train-cities or --each-city is set)",
    )
    ap.add_argument(
        "--train-cities",
        type=str,
        default="",
        help="Multi-city training: comma-separated slugs, or 'all' for all six cities. Each reset picks a random city and window.",
    )
    ap.add_argument(
        "--each-city",
        action="store_true",
        help="Train one SAC per city in CITY_SLUGS; writes <out-dir>/<slug>.zip (+ .vecnorm.pkl if --vecnormalize).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="With --each-city: skip training for a slug if <out-dir>/<slug>.zip already exists (and .vecnorm.pkl if --vecnormalize).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Required with --each-city: directory for per-city checkpoints (filenames are <slug>.zip).",
    )
    ap.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="SB3 learn() total_timesteps (per city when --each-city)",
    )
    ap.add_argument(
        "--episode-len",
        type=int,
        default=512,
        help="max_episode_steps per reset (shorter = faster RL iterations)",
    )
    ap.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel envs (DummyVecEnv). Higher is usually faster.",
    )
    ap.add_argument("--calendar-year", type=int, default=2019)
    ap.add_argument(
        "--calendar-year-choices",
        type=str,
        default="",
        help="Comma-separated years to sample from on each reset (optional).",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--model-path",
        type=str,
        default="world_model_best.pt",
        help="Checkpoint file or name under data/training/checkpoints/",
    )
    ap.add_argument(
        "--random-start-row-max",
        type=int,
        default=None,
        help="If set, pick a random start_row in [0,max] each reset (clamped per year).",
    )
    ap.add_argument("--vecnormalize", action="store_true", help="Enable VecNormalize on observations (and rewards unless --no-norm-reward).")
    ap.add_argument(
        "--no-norm-reward",
        action="store_true",
        help="With --vecnormalize: normalize observations only (norm_reward=False).",
    )
    ap.add_argument(
        "--action-box",
        type=str,
        default="full",
        choices=("full", "baseline"),
        help="'baseline' = chiller [6.5,8.5], fans [0.55,0.85] around fixed baseline.",
    )
    ap.add_argument(
        "--eval-aligned-fraction",
        type=float,
        default=0.0,
        help="On each reset, probability of sampling start_row from --eval-start-rows (eval contract).",
    )
    ap.add_argument(
        "--eval-start-rows",
        type=str,
        default="0,2160,4320",
        help="Comma-separated start_row values for eval-aligned resets.",
    )
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument(
        "--save",
        type=Path,
        default=Path("data/rl/tropcool_sac"),
        help="SB3 save prefix (writes ``<save>.zip``); not used with --each-city (use --out-dir).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    action_box = None if args.action_box == "full" else args.action_box
    eval_start_rows: list[int] | None = None
    if float(args.eval_aligned_fraction) > 0.0:
        eval_start_rows = [
            int(x.strip()) for x in str(args.eval_start_rows).split(",") if x.strip()
        ]
    norm_reward = not args.no_norm_reward

    if args.each_city:
        if args.out_dir is None:
            ap.error("--each-city requires --out-dir")
        if str(args.train_cities).strip():
            ap.error("Do not combine --each-city with --train-cities")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        year_choices: list[int] | None = None
        if str(args.calendar_year_choices).strip():
            year_choices = [int(x.strip()) for x in str(args.calendar_year_choices).split(",") if x.strip()]
        else:
            year_choices = [2018, 2019, 2020]
        rsrm = args.random_start_row_max
        if rsrm is None:
            rsrm = 8760
        for i, slug in enumerate(CITY_SLUGS):
            print(f"\n========== Training SAC for {slug} ({i + 1}/{len(CITY_SLUGS)}) ==========\n", flush=True)
            zip_path = args.out_dir / f"{slug}.zip"
            vn_path = args.out_dir / f"{slug}.vecnorm.pkl"
            if args.resume and zip_path.is_file():
                if args.vecnormalize and not vn_path.is_file():
                    print(
                        f"Re-training {slug}: --resume but missing {vn_path.name}",
                        flush=True,
                    )
                else:
                    print(f"Skipping {slug} (--resume: found {zip_path.name})", flush=True)
                    continue
            train_sac_once(
                city=slug,
                save_prefix=args.out_dir / slug,
                timesteps=args.timesteps,
                episode_len=args.episode_len,
                n_envs=args.n_envs,
                calendar_year=args.calendar_year,
                calendar_year_choices=year_choices,
                random_start_row_max=rsrm,
                train_slugs=None,
                model_path=args.model_path,
                device=args.device,
                vecnormalize=args.vecnormalize,
                norm_reward=norm_reward,
                action_box=action_box,
                eval_aligned_fraction=float(args.eval_aligned_fraction),
                eval_start_row_choices=eval_start_rows,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                seed=args.seed + i * 10_000,
                verbose=args.verbose,
            )
        return 0

    args.save.parent.mkdir(parents=True, exist_ok=True)
    year_choices = None
    if str(args.calendar_year_choices).strip():
        year_choices = [int(x.strip()) for x in str(args.calendar_year_choices).split(",") if x.strip()]

    tc_raw = str(args.train_cities).strip().lower()
    train_slugs: list[str] | None = None
    if tc_raw == "all":
        train_slugs = list(CITY_SLUGS)
    elif str(args.train_cities).strip():
        train_slugs = [
            x.strip().lower().replace(" ", "_")
            for x in str(args.train_cities).split(",")
            if x.strip()
        ]
    init_city = train_slugs[0] if train_slugs else args.city

    train_sac_once(
        city=init_city,
        save_prefix=args.save,
        timesteps=args.timesteps,
        episode_len=args.episode_len,
        n_envs=args.n_envs,
        calendar_year=args.calendar_year,
        calendar_year_choices=year_choices,
        random_start_row_max=args.random_start_row_max,
        train_slugs=train_slugs,
        model_path=args.model_path,
        device=args.device,
        vecnormalize=args.vecnormalize,
        norm_reward=norm_reward,
        action_box=action_box,
        eval_aligned_fraction=float(args.eval_aligned_fraction),
        eval_start_row_choices=eval_start_rows,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        seed=args.seed,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
