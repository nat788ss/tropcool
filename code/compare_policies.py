#!/usr/bin/env python3
"""Compare fixed baseline vs rules-based vs SAC agent across all six cities.

Runs ``WorldModelEnv`` episodes and prints returns, predicted cost, rack violations,
and mean PUE.

Single-slice mode (default):: one ``calendar_year``, ``weather_start_row``, ``seed``.

Mean-over-scenarios mode (**--mean-over-scenarios**) averages over a grid of:

- multiple **calendar years** ``--eval-years`` (default ``2018,2019,2020``),
- multiple **start rows** ``--eval-start-rows`` (default ``0,2160,4320`` hours into the filtered year),
- multiple episode **RNG seeds** ``--eval-seeds`` (default ``42,43,44``),

i.e. the **Cartesian product** — **per-city mean cost_RM** (and std) over all scenarios.

**Wide coverage** (**--eval-wide-coverage**): calendar **years = intersection** across all six city CSVs; **start_rows** = evenly spaced in ``[0, gmax]`` where ``gmax`` is the minimum feasible offset **across cities** for that year and episode length. Baseline, rules, and SAC use the **same** scenario list. Prefer ``--eval-seeds 42`` (single seed) unless you need replication—the scenario count grows quickly.

Examples::

    python3 compare_policies.py --rl-ckpt data/rl/tropcool_sac --episode-len 720

    python3 compare_policies.py --rl-per-city-dir data/rl/per_city --episode-len 256 \\
        --mean-over-scenarios --require-sac-beats-baseline-all

If ``--rl-ckpt`` is missing or load fails, the RL column is omitted / marked N/A.
``--rl-per-city-dir`` takes precedence over ``--rl-ckpt`` for SAC.

Requires: ``pip install stable_baselines3`` (only for RL column).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_data import CITY_SLUGS
from predict_year import default_weather_path, load_weather_csv, min_weather_rows

from tropcool_rl import (
    load_sb3_sac_policy_fn,
    make_world_model_env,
    policy_fixed_baseline,
    policy_rules_weather,
    rollout_episode,
)


def _parse_csv_ints(s: str) -> list[int]:
    t = str(s).strip()
    if not t:
        return []
    return [int(x.strip()) for x in t.split(",") if x.strip()]


def _parse_csv_slugs(s: str) -> list[str]:
    t = str(s).strip()
    if not t:
        return []
    out: list[str] = []
    for x in t.split(","):
        slug = x.strip()
        if slug:
            out.append(slug)
    return out


def _mean_y_pct(costs: dict[str, dict[str, list[float]]], slugs: list[str]) -> float:
    base_c = float(np.mean([float(np.mean(costs[sl]["baseline"])) for sl in slugs]))
    sac_c = float(np.mean([float(np.mean(costs[sl]["SAC"])) for sl in slugs]))
    return 100.0 * (1.0 - sac_c / base_c) if base_c > 0 else float("nan")


def _load_weather_cache(weather_dir: Path) -> dict[str, Any]:
    return {s: load_weather_csv(default_weather_path(s, weather_dir)) for s in CITY_SLUGS}


def _years_intersection_from_cache(cache: dict[str, Any]) -> list[int]:
    sets = [{int(y) for y in cache[s]["_ts"].dt.year.unique().tolist()} for s in CITY_SLUGS]
    return sorted(set.intersection(*sets))


def _feasible_max_start_row(wfull, calendar_year: int, episode_hours: int) -> int:
    need = int(min_weather_rows(episode_hours))
    wy = wfull[wfull["_ts"].dt.year == int(calendar_year)]
    base_len = len(wy)
    return max(0, base_len - need)


def _global_feasible_max_start_cached(cache: dict[str, Any], calendar_year: int, episode_hours: int) -> int:
    """Max start_row such that every city can run ``episode_hours`` from that offset in ``calendar_year``."""
    return min(_feasible_max_start_row(cache[s], calendar_year, episode_hours) for s in CITY_SLUGS)


def _evenly_spaced_starts(gmax: int, n_positions: int) -> list[int]:
    """Unique integers from 0 .. gmax inclusive, approximately evenly spaced (inclusive endpoints)."""
    gmax = int(max(0, gmax))
    if gmax == 0:
        return [0]
    n = int(min(max(1, n_positions), gmax + 1))
    if n == 1:
        return [0]
    raw = [round(i * gmax / (n - 1)) for i in range(n)]
    return sorted({int(min(gmax, max(0, x))) for x in raw})


def build_wide_coverage_scenarios(
    *,
    weather_dir: Path,
    episode_hours: int,
    starts_per_year: int,
    seeds: list[int],
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    cache = _load_weather_cache(weather_dir)
    years = _years_intersection_from_cache(cache)
    scenarios: list[tuple[int, int, int]] = []
    for year in years:
        gmax = _global_feasible_max_start_cached(cache, year, episode_hours)
        for sr in _evenly_spaced_starts(gmax, starts_per_year):
            for sd in seeds:
                scenarios.append((year, sr, sd))
    meta = {
        "years": years,
        "starts_per_year": starts_per_year,
        "seeds": seeds,
        "n_scenarios": len(scenarios),
    }
    return scenarios, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-len", type=int, default=720, help="Hours per episode")
    ap.add_argument("--calendar-year", type=int, default=2019)
    ap.add_argument("--weather-start-row", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--action-box",
        type=str,
        default="full",
        choices=("full", "baseline"),
        help="SAC env action bounds when loading SB3 (must match training). Baseline policy unchanged.",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default="world_model_best.pt",
    )
    ap.add_argument(
        "--rl-ckpt",
        type=Path,
        default=Path("data/rl/tropcool_sac.zip"),
        help="Path to SAC zip (SB3 adds .zip to save()); ignored if --rl-per-city-dir is set.",
    )
    ap.add_argument(
        "--rl-per-city-dir",
        type=Path,
        default=None,
        help="Directory with <slug>.zip (+ optional <slug>.vecnorm.pkl) per city.",
    )
    ap.add_argument(
        "--vecnorm",
        type=Path,
        default=None,
        help="VecNormalize stats for single --rl-ckpt (defaults to <rl-ckpt base>.vecnorm.pkl).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--weather-dir",
        type=Path,
        default=Path("data/weather"),
        help="Hourly CSV directory (used for --eval-wide-coverage year intersection and feasibility).",
    )
    ap.add_argument(
        "--mean-over-scenarios",
        action="store_true",
        help="Average over Cartesian product of --eval-years × --eval-start-rows × --eval-seeds.",
    )
    ap.add_argument(
        "--eval-wide-coverage",
        action="store_true",
        help="Average over all calendar years common to every city + evenly spaced start_rows per year (see --eval-coverage-starts-per-year); implies multi-scenario averaging.",
    )
    ap.add_argument(
        "--eval-coverage-starts-per-year",
        type=int,
        default=24,
        help="With --eval-wide-coverage: number of evenly spaced start_row positions per year.",
    )
    ap.add_argument(
        "--eval-years",
        type=str,
        default="2018,2019,2020",
        help="Comma-separated calendar years (mean-over-scenarios only).",
    )
    ap.add_argument(
        "--eval-start-rows",
        type=str,
        default="0,2160,4320",
        help="Comma-separated start_row values after year filter (hours offsets into slice).",
    )
    ap.add_argument(
        "--eval-seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated RNG seeds for env.reset (Monte Carlo-style replication).",
    )
    ap.add_argument(
        "--print-each-scenario",
        action="store_true",
        help="Print every rollout line (can be very verbose).",
    )
    ap.add_argument(
        "--require-sac-beats-baseline-all",
        action="store_true",
        help="Exit 1 if SAC mean cost is not strictly below baseline mean cost on any validated city.",
    )
    ap.add_argument(
        "--exclude-cities",
        type=str,
        default="",
        help=(
            "Comma-separated slugs excluded from validated portfolio mean Y and "
            "--require-sac-beats-baseline-all (still evaluated and printed). "
            "Example: jakarta"
        ),
    )
    args = ap.parse_args()

    excluded_slugs = _parse_csv_slugs(args.exclude_cities)
    unknown_excluded = [s for s in excluded_slugs if s not in CITY_SLUGS]
    if unknown_excluded:
        print(f"ERROR: unknown --exclude-cities slug(s): {unknown_excluded}", file=sys.stderr)
        return 2
    excluded_set = frozenset(excluded_slugs)
    validated_slugs = [s for s in CITY_SLUGS if s not in excluded_set]

    sac_action_box = None if args.action_box == "full" else args.action_box

    wide_meta: dict[str, Any] | None = None
    if args.eval_wide_coverage:
        seeds_w = _parse_csv_ints(args.eval_seeds)
        if not seeds_w:
            seeds_w = [42]
        scenarios, wide_meta = build_wide_coverage_scenarios(
            weather_dir=args.weather_dir,
            episode_hours=args.episode_len,
            starts_per_year=max(1, int(args.eval_coverage_starts_per_year)),
            seeds=seeds_w,
        )
        if not scenarios:
            print(
                "ERROR: --eval-wide-coverage produced no scenarios (empty year intersection or infeasible episode length).",
                file=sys.stderr,
            )
            return 2
    elif args.mean_over_scenarios:
        years = _parse_csv_ints(args.eval_years)
        starts = _parse_csv_ints(args.eval_start_rows)
        seeds = _parse_csv_ints(args.eval_seeds)
        if not years or not starts or not seeds:
            print("ERROR: --mean-over-scenarios needs non-empty --eval-years, --eval-start-rows, --eval-seeds", file=sys.stderr)
            return 2
        scenarios = [(y, sr, sd) for y, sr, sd in itertools.product(years, starts, seeds)]
    else:
        scenarios = [(args.calendar_year, args.weather_start_row, args.seed)]

    per_city_sac: dict[str, Any] | None = None
    sac_single = None

    cy0, sr0, sd0 = scenarios[0]

    if args.rl_per_city_dir is not None:
        d = Path(args.rl_per_city_dir)
        missing: list[str] = []
        for slug in CITY_SLUGS:
            zp = d / f"{slug}.zip"
            if not zp.is_file():
                missing.append(str(zp))
        if missing:
            print("ERROR: --rl-per-city-dir requires every city zip:", file=sys.stderr)
            for m in missing:
                print(f"  missing {m}", file=sys.stderr)
            return 2
        try:
            per_city_sac = {}
            for slug in CITY_SLUGS:
                per_city_sac[slug] = load_sb3_sac_policy_fn(
                    d / f"{slug}.zip",
                    eval_city_slug=slug,
                    max_episode_steps=args.episode_len,
                    calendar_year=cy0,
                    weather_start_row=sr0,
                    model_path=args.model_path,
                    device=args.device,
                    vecnorm_path=None,
                    action_box=sac_action_box,
                )
        except Exception as e:
            print(f"Could not load per-city RL from {d}: {e}", file=sys.stderr)
            per_city_sac = None
    else:
        rl_path = args.rl_ckpt
        if not rl_path.is_file() and rl_path.with_suffix(".zip").is_file():
            rl_path = rl_path.with_suffix(".zip")
        if rl_path.is_file():
            try:
                vecnorm_path = args.vecnorm
                if vecnorm_path is None:
                    vecnorm_path = rl_path.with_suffix(".vecnorm.pkl")
                sac_single = load_sb3_sac_policy_fn(
                    rl_path,
                    eval_city_slug=CITY_SLUGS[0],
                    max_episode_steps=args.episode_len,
                    calendar_year=cy0,
                    weather_start_row=sr0,
                    model_path=args.model_path,
                    device=args.device,
                    vecnorm_path=vecnorm_path if vecnorm_path.is_file() else None,
                    action_box=sac_action_box,
                )
            except Exception as e:
                print(f"Could not load RL checkpoint {rl_path}: {e}", file=sys.stderr)
                sac_single = None
        else:
            print(f"No RL checkpoint at {args.rl_ckpt} — SAC column will be N/A.", file=sys.stderr)

    policies_base: list[tuple[str, Any]] = [
        ("baseline", policy_fixed_baseline),
        ("rules", policy_rules_weather),
    ]
    has_sac = per_city_sac is not None or sac_single is not None

    mode = "per-city SAC" if per_city_sac else ("single SAC" if sac_single else "no SAC")
    if args.eval_wide_coverage and wide_meta is not None:
        agg_label = (
            f"mean over {len(scenarios)} scenarios (wide coverage: "
            f"{wide_meta['starts_per_year']} starts/year × {len(wide_meta['years'])} years × {len(wide_meta['seeds'])} seeds)"
        )
    elif args.mean_over_scenarios:
        agg_label = f"mean over {len(scenarios)} scenarios (years × start_rows × seeds)"
    else:
        agg_label = "single slice"
    print(f"Episode length={args.episode_len} h  {agg_label}  ({mode})", flush=True)
    if args.eval_wide_coverage and wide_meta is not None:
        print(f"  weather_dir={args.weather_dir}", flush=True)
        yl = wide_meta["years"]
        print(
            f"  calendar years (intersection of all six cities, n={len(yl)}): "
            f"{yl[0]} … {yl[-1]}",
            flush=True,
        )
        print(
            f"  evenly spaced start_rows per year (globally feasible for all cities); "
            f"seeds={wide_meta['seeds']}",
            flush=True,
        )
    elif args.mean_over_scenarios:
        print(
            f"  years={_parse_csv_ints(args.eval_years)}  "
            f"start_rows={_parse_csv_ints(args.eval_start_rows)}  "
            f"seeds={_parse_csv_ints(args.eval_seeds)}",
            flush=True,
        )
    else:
        print(
            f"  calendar_year={args.calendar_year}  start_row={args.weather_start_row}  seed={args.seed}",
            flush=True,
        )
    if has_sac and sac_action_box == "baseline":
        print("  SAC action_box=baseline (chiller [6.5,8.5], fans [0.55,0.85])", flush=True)
    print("", flush=True)

    costs: dict[str, dict[str, list[float]]] = {
        slug: {name: [] for name, _ in policies_base} for slug in CITY_SLUGS
    }
    if has_sac:
        for slug in CITY_SLUGS:
            costs[slug]["SAC"] = []

    returns: dict[str, dict[str, list[float]]] = {
        slug: {name: [] for name, _ in policies_base} for slug in CITY_SLUGS
    }
    if has_sac:
        for slug in CITY_SLUGS:
            returns[slug]["SAC"] = []

    viols: dict[str, dict[str, list[int]]] = {
        slug: {name: [] for name, _ in policies_base} for slug in CITY_SLUGS
    }
    if has_sac:
        for slug in CITY_SLUGS:
            viols[slug]["SAC"] = []

    for slug in CITY_SLUGS:
        for scenario_idx, (cy, sr, sd) in enumerate(scenarios):
            for name, pol in policies_base:
                env = make_world_model_env(
                    slug,
                    max_episode_steps=args.episode_len,
                    calendar_year=cy,
                    weather_start_row=sr,
                    model_path=args.model_path,
                    device=args.device,
                )
                stats = rollout_episode(env, pol, seed=sd)
                costs[slug][name].append(float(stats["total_cost_rm"]))
                returns[slug][name].append(float(stats["return"]))
                viols[slug][name].append(int(stats["violations"]))
                if args.print_each_scenario:
                    print(
                        f"{slug:18s}  {name:10s}  y={cy} row={sr} sd={sd}  "
                        f"{stats['return']:12.1f}  {stats['total_cost_rm']:12,.0f}  "
                        f"{stats['violations']:5d}  {stats['mean_pue']:9.4f}",
                        flush=True,
                    )

            if has_sac:
                sac_pol = per_city_sac[slug] if per_city_sac is not None else sac_single
                assert sac_pol is not None
                env = make_world_model_env(
                    slug,
                    max_episode_steps=args.episode_len,
                    calendar_year=cy,
                    weather_start_row=sr,
                    model_path=args.model_path,
                    device=args.device,
                )
                stats = rollout_episode(env, sac_pol, seed=sd)
                costs[slug]["SAC"].append(float(stats["total_cost_rm"]))
                returns[slug]["SAC"].append(float(stats["return"]))
                viols[slug]["SAC"].append(int(stats["violations"]))
                if args.print_each_scenario:
                    print(
                        f"{slug:18s}  {'SAC':10s}  y={cy} row={sr} sd={sd}  "
                        f"{stats['return']:12.1f}  {stats['total_cost_rm']:12,.0f}  "
                        f"{stats['violations']:5d}  {stats['mean_pue']:9.4f}",
                        flush=True,
                    )

            if args.print_each_scenario and scenario_idx < len(scenarios) - 1:
                print("", flush=True)

    n_sc = len(scenarios)
    if has_sac:
        hdr = f"{'city':18s}  {'n':>4s}  {'baseline_RM':>26s}  {'rules_RM':>26s}  {'SAC_RM':>26s}"
    else:
        hdr = f"{'city':18s}  {'n':>4s}  {'baseline_RM':>26s}  {'rules_RM':>26s}"
    print("Per-city statistics (cost_RM = episode total predicted cooling cost):", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    per_city_mean_base: dict[str, float] = {}
    per_city_mean_sac: dict[str, float] = {}

    def _fmt_mean_std(vals: list[float]) -> str:
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=0))
        return f"{m:12,.0f} ± {s:,.0f}"

    for slug in CITY_SLUGS:
        b_line = _fmt_mean_std(costs[slug]["baseline"])
        r_line = _fmt_mean_std(costs[slug]["rules"])
        if has_sac:
            s_line = _fmt_mean_std(costs[slug]["SAC"])
            print(
                f"{slug:18s}  {n_sc:4d}  {b_line:>26s}  {r_line:>26s}  {s_line:>26s}",
                flush=True,
            )
            per_city_mean_base[slug] = float(np.mean(costs[slug]["baseline"]))
            per_city_mean_sac[slug] = float(np.mean(costs[slug]["SAC"]))
        else:
            print(
                f"{slug:18s}  {n_sc:4d}  {b_line:>26s}  {r_line:>26s}",
                flush=True,
            )
            per_city_mean_base[slug] = float(np.mean(costs[slug]["baseline"]))

    def _print_city_aggregate(label: str, slugs: list[str]) -> None:
        print(f"Mean across {len(slugs)} cities — {label} (of per-city mean cost_RM):", flush=True)
        for name in ("baseline", "rules", "SAC"):
            if name == "SAC" and not has_sac:
                continue
            city_means = [float(np.mean(costs[sl][name])) for sl in slugs]
            print(
                f"  {name:10s}  cost_RM={np.mean(city_means):12,.0f}  "
                f"return={np.mean([float(np.mean(returns[sl][name])) for sl in slugs]):12.1f}  "
                f"viol/scen={np.mean([float(np.mean(viols[sl][name])) for sl in slugs]):.3f}",
                flush=True,
            )

    # Aggregate across cities (mean of per-city means — same as mean of all city×scenario cells / n_sc if balanced)
    print("", flush=True)
    _print_city_aggregate("all cities", list(CITY_SLUGS))
    if excluded_set and validated_slugs:
        print("", flush=True)
        excl_label = ", ".join(excluded_slugs)
        _print_city_aggregate(f"validated portfolio (excludes {excl_label})", validated_slugs)

    if has_sac:
        base_all = float(
            np.mean([float(np.mean(costs[sl]["baseline"])) for sl in CITY_SLUGS])
        )
        sac_all = float(np.mean([float(np.mean(costs[sl]["SAC"])) for sl in CITY_SLUGS]))
        y_all = _mean_y_pct(costs, list(CITY_SLUGS))
        print("", flush=True)
        print(
            f"Predicted cooling cost reduction (all {len(CITY_SLUGS)} cities): "
            f"Y = {y_all:.2f}%  (baseline {base_all:,.0f} RM vs SAC {sac_all:,.0f} RM)",
            flush=True,
        )
        if excluded_set and validated_slugs:
            base_val = float(
                np.mean([float(np.mean(costs[sl]["baseline"])) for sl in validated_slugs])
            )
            sac_val = float(np.mean([float(np.mean(costs[sl]["SAC"])) for sl in validated_slugs]))
            y_val = _mean_y_pct(costs, validated_slugs)
            excl_label = ", ".join(excluded_slugs)
            print(
                f"Predicted cooling cost reduction (validated portfolio, "
                f"{len(validated_slugs)} cities, excludes {excl_label}): "
                f"Y = {y_val:.2f}%  (baseline {base_val:,.0f} RM vs SAC {sac_val:,.0f} RM)",
                flush=True,
            )

    exit_code = 0
    if args.require_sac_beats_baseline_all:
        if not has_sac:
            print("ERROR: --require-sac-beats-baseline-all needs a loaded SAC checkpoint.", file=sys.stderr)
            return 2
        strict_slugs = validated_slugs if validated_slugs else list(CITY_SLUGS)
        if excluded_set:
            print(
                f"Strict gate (--require-sac-beats-baseline-all) applies to "
                f"{len(strict_slugs)} validated cities; excluded from gate: {', '.join(excluded_slugs)}",
                flush=True,
            )
        failures: list[tuple[str, float, float]] = []
        for slug in strict_slugs:
            b = per_city_mean_base.get(slug)
            s = per_city_mean_sac.get(slug)
            if b is None or s is None:
                print(f"ERROR: missing mean costs for city {slug}", file=sys.stderr)
                return 2
            if not (s < b):
                failures.append((slug, b, s))
        if failures:
            label = "mean cost_RM" if (args.mean_over_scenarios or args.eval_wide_coverage) else "cost_RM"
            scope = f"validated {len(strict_slugs)} cities" if excluded_set else "all cities"
            print("", flush=True)
            print(
                f"SAC did not beat baseline on all {scope} "
                f"(strict SAC {label} < baseline {label}):",
                flush=True,
            )
            for slug, b, s in failures:
                print(f"  {slug}: baseline {b:,.0f} RM  SAC {s:,.0f} RM", flush=True)
            exit_code = 1
        elif excluded_set:
            print(
                f"SAC beat baseline on all {len(strict_slugs)} validated cities "
                f"({', '.join(excluded_slugs)} excluded from strict gate).",
                flush=True,
            )

    print("", flush=True)
    print(
        "Higher return is better (reward = −predicted cost − 1000 on rack > 27 °C). "
        "Lower cost and fewer violations are better.",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
