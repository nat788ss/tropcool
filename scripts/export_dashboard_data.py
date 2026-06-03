#!/usr/bin/env python3
"""Export all precomputed JSON assets for the TropCool Streamlit dashboard.

Outputs:
  dashboard/climate_results.json   — city × facility × cooling × rack × horizon
  dashboard/traces/{slug}_24h.json — hourly baseline vs SAC animation
  dashboard/results.json           — strict / pilot RL cost comparison
  dashboard/model_metrics.json     — world model test R² / MAE
  dashboard/regional_impact.json   — SEA aggregate headlines

Run from repo root::

    python3 scripts/export_dashboard_data.py
    python3 scripts/export_dashboard_data.py --fast   # shorter rollouts
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DASH = ROOT / "dashboard"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fetch_weather_historical import LOCATIONS
from generate_data import CITY_SLUGS

HORIZONS: dict[str, float] = {
    "2025": 0.0,
    "2030": 0.5,
    "2040": 1.2,
    "2050": 2.0,
}
# Legacy period keys in projection CSV filenames
HORIZON_TO_PERIOD = {"2025": "current", "2030": "2030", "2040": "2040", "2050": "2050"}

FACILITY_MW_GRID = (5, 10, 20, 30, 50)
COOLING_TYPES = ("air_cooled", "water_cooled", "hybrid")
RACK_DENSITY_KEYS = ("0.3", "0.5", "0.65", "0.8", "1.0")
BASE_MW = 10.0
BASE_RACK = 0.65
BASE_COOLING = "water_cooled"

COOLING_SCALE = {
    "water_cooled": {"pue_delta": 0.0, "water_mult": 1.0, "cost_mult": 1.0},
    "air_cooled": {"pue_delta": 0.06, "water_mult": 0.22, "cost_mult": 1.04},
    "hybrid": {"pue_delta": 0.03, "water_mult": 0.55, "cost_mult": 1.02},
}

CO2_KG_PER_MWH = 450.0


def rack_density_to_kw(rack_norm: float) -> float:
    """Map normalized rack density 0.3–1.0 → 5–25 kW/rack (linear)."""
    r = float(np.clip(rack_norm, 0.3, 1.0))
    return 5.0 + (r - 0.3) * (20.0 / 0.7)


def _annualize(cost_window: float, hours: int) -> float:
    if hours <= 0:
        return float(cost_window)
    return float(cost_window) * (8760.0 / hours)


def _scale_policy_metrics(
    baseline: dict,
    sac: dict | None,
    *,
    facility_mw: float,
    cooling_type: str,
    rack_key: str,
) -> tuple[dict, dict | None, float]:
    mw_f = facility_mw / BASE_MW
    rack = float(rack_key)
    rack_pue_mult = 1.0 + 0.12 * (rack - BASE_RACK)
    cs = COOLING_SCALE.get(cooling_type, COOLING_SCALE[BASE_COOLING])

    def _one(cell: dict) -> dict:
        pue = float(cell.get("mean_pue", 1.65)) * rack_pue_mult + cs["pue_delta"]
        cost = float(cell.get("cost_RM_annualized") or cell.get("cost_RM", 0)) * mw_f * cs["cost_mult"]
        water = float(cell.get("water_l", cell.get("water_liters_annual", 0)))
        if water <= 0:
            # Estimate from cost if missing (placeholder L/MWh cooling)
            water = cost * 8.0
        water = water * mw_f * cs["water_mult"]
        return {"pue": round(pue, 4), "cost_rm": round(cost, 0), "water_l": round(water, 0)}

    b = _one(baseline)
    s_out = _one(sac) if sac else None
    y = 0.0
    if s_out and b["cost_rm"] > 0:
        y = round(100.0 * (1.0 - s_out["cost_rm"] / b["cost_rm"]), 2)
    return b, s_out, y


def _attach_climate_headlines(payload: dict, base_cities: dict) -> None:
    """Merge legacy + v2 headline blocks from rollout-period city tree."""
    from run_climate_extended_results import _headlines, _headlines_v2

    payload["headlines"] = _headlines(base_cities)
    payload["headlines_v2"] = _headlines_v2(base_cities)
    payload["headlines_v2_note"] = (
        "X/Y = cost growth vs current; savings_at_weather_pct = SAC vs baseline on same future weather. "
        "Stress periods 2040_stress (+2.5°C), 2050_stress (+3.0°C). "
        "validated_portfolio_mean excludes Jakarta."
    )


def _expand_climate_grid(base_cities: dict, episode_hours: int) -> dict:
    """Expand per-city per-period rollouts into full config grid."""
    out: dict = {
        "meta": {
            "facility_mw_grid": list(FACILITY_MW_GRID),
            "cooling_types": list(COOLING_TYPES),
            "rack_density_keys": list(RACK_DENSITY_KEYS),
            "rack_kw_mapping": "rack_kw = 5 + (density - 0.3) * (20 / 0.7) for density in [0.3, 1.0]",
            "base_reference": {
                "facility_mw": BASE_MW,
                "cooling_type": BASE_COOLING,
                "rack_density": BASE_RACK,
            },
            "horizons": HORIZONS,
            "episode_hours": episode_hours,
        },
        "cities": {},
    }
    for slug, periods in base_cities.items():
        out["cities"][slug] = {}
        for fm in FACILITY_MW_GRID:
            out["cities"][slug][str(fm)] = {}
            for ct in COOLING_TYPES:
                out["cities"][slug][str(fm)][ct] = {}
                for rk in RACK_DENSITY_KEYS:
                    out["cities"][slug][str(fm)][ct][rk] = {}
                    for hz, warm in HORIZONS.items():
                        period = HORIZON_TO_PERIOD[hz]
                        cell = periods.get(period, {})
                        b_raw = cell.get("baseline", {})
                        s_raw = cell.get("sac")
                        if not b_raw:
                            continue
                        b_raw = dict(b_raw)
                        if s_raw:
                            s_raw = dict(s_raw)
                            s_raw.setdefault("cost_RM", s_raw.get("cost_RM_annualized"))
                        b_raw.setdefault("water_l", b_raw.get("water_liters_annual", 0))
                        b, s, y = _scale_policy_metrics(
                            b_raw, s_raw, facility_mw=fm, cooling_type=ct, rack_key=rk
                        )
                        out["cities"][slug][str(fm)][ct][rk][hz] = {
                            "baseline": b,
                            "sac": s or b,
                            "y_pct": y,
                            "warming_c": warm,
                        }
    return out


def _run_base_climate_rollouts(
    *,
    projections_dir: Path,
    rl_dir: Path,
    episode_hours: int,
    model_path: Path,
    device: str,
    action_box: str,
) -> dict[str, dict]:
    from run_climate_extended_results import _run_cell, projection_path

    cities_out: dict[str, dict] = {}
    for slug in CITY_SLUGS:
        cities_out[slug] = {}
        sac_zip = rl_dir / f"{slug}.zip"
        for period in ("current", "2030", "2040", "2050", "2040_stress", "2050_stress"):
            wcsv = projection_path(projections_dir, slug, period)
            if not wcsv.is_file():
                raise FileNotFoundError(f"Missing projection weather: {wcsv}")
            print(f"  rollout {slug} / {period} ({episode_hours}h)…", flush=True)
            base_metrics = _run_cell(
                slug,
                wcsv,
                episode_hours=episode_hours,
                calendar_year=2019,
                weather_start_row=0,
                seed=42,
                model_path=str(model_path),
                device=device,
                sac_zip=None,
                action_box=action_box if action_box != "full" else None,
            )
            cell = {"baseline": {k: v for k, v in base_metrics.items() if not k.startswith("sac_")}}
            if sac_zip.is_file():
                full = _run_cell(
                    slug,
                    wcsv,
                    episode_hours=episode_hours,
                    calendar_year=2019,
                    weather_start_row=0,
                    seed=42,
                    model_path=str(model_path),
                    device=device,
                    sac_zip=sac_zip,
                    action_box=action_box if action_box != "full" else None,
                )
                cell["sac"] = {
                    "cost_RM": full.get("sac_cost_RM"),
                    "cost_RM_annualized": full.get("sac_cost_RM_annualized"),
                    "mean_pue": full.get("sac_mean_pue"),
                    "violations": full.get("sac_violations"),
                }
                ann = cell["sac"].get("cost_RM_annualized") or cell["sac"].get("cost_RM", 0)
                b_ann = cell["baseline"].get("cost_RM_annualized") or cell["baseline"].get("cost_RM", 0)
                cell["sac"]["water_l"] = ann * 6.5
                cell["baseline"]["water_l"] = b_ann * 6.5
            cities_out[slug][period] = cell
    return cities_out


def export_model_metrics(out: Path) -> None:
    ck = ROOT / "data/training/checkpoints/kaggle_finetune_metrics.json"
    if ck.is_file():
        raw = json.loads(ck.read_text())
        after = raw.get("after", raw)
    else:
        after = {}
    targets = {
        "rack_inlet_temp": "rack_temp_c",
        "cooling_power": "cooling_power_kw",
        "water_consumption": "water_l",
        "pue": "pue",
        "cost": "cost_rm",
    }
    metrics = {}
    for src, label in targets.items():
        m = after.get(src, {})
        metrics[label] = {
            "r2": m.get("r2"),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(ck.relative_to(ROOT)) if ck.is_file() else "synthetic",
        "split": "kaggle_finetune_holdout",
        "targets": metrics,
        "note": "Post fine-tune metrics on real DC operational overlap (kaggle_finetune).",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")


def parse_eval_log(text: str) -> dict:
    cities: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = re.match(
            r"^(\w+)\s+\d+\s+([\d,]+)\s*±\s*[\d,]+\s+[\d,]+\s*±\s*[\d,]+\s+([\d,]+)",
            line.strip(),
        )
        if m:
            city, base, sac = m.group(1), m.group(2), m.group(3)
            cities[city] = {
                "baseline_RM": float(base.replace(",", "")),
                "SAC_RM": float(sac.replace(",", "")),
            }
    y = re.search(r"Y\s*=\s*([\d.]+)%", text)
    failures = []
    for line in text.splitlines():
        fm = re.search(
            r"^\s+(\w+):\s+baseline\s+([\d,]+)\s+RM\s+SAC\s+([\d,]+)\s+RM",
            line,
        )
        if fm:
            b_cost, s_cost = float(fm.group(2).replace(",", "")), float(fm.group(3).replace(",", ""))
            if s_cost >= b_cost:
                failures.append(fm.group(1))
    strict_pass = len(failures) == 0 and "SAC did not beat baseline" not in text
    return {
        "cities": cities,
        "mean_Y_pct": float(y.group(1)) if y else None,
        "strict_sac_beats_baseline_all": strict_pass,
        "strict_failures": failures,
    }


def export_results(out: Path) -> None:
    for log in (
        ROOT / "data/rl/per_city_v2_strict_eval.log",
        ROOT / "data/rl/pilot/singapore_ab_eval.log",
    ):
        if log.is_file() and "Per-city statistics" in log.read_text():
            data = parse_eval_log(log.read_text())
            data["source"] = str(log.relative_to(ROOT))
            data["generated_at"] = datetime.now(timezone.utc).isoformat()
            out.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Wrote {out} from {log.name}")
            return
    print(f"WARN: no eval log found; keeping existing {out}", file=sys.stderr)


def export_regional(climate_grid: dict, results: dict, out: Path) -> None:
    """Aggregate regional impact from scaled 2025 / 10MW / water / 0.65 grid."""
    total_saved_rm = 0.0
    total_water_saved_l = 0.0
    total_mw_equiv = 0.0
    per_city = []
    for slug in CITY_SLUGS:
        try:
            cell = climate_grid["cities"][slug]["10"]["water_cooled"]["0.65"]["2025"]
        except (KeyError, TypeError):
            continue
        b = cell["baseline"]
        s = cell.get("sac", b)
        saved = max(0.0, b["cost_rm"] - s["cost_rm"])
        water_saved = max(0.0, b["water_l"] - s["water_l"])
        y = cell.get("y_pct", 0.0)
        per_city.append(
            {
                "city": slug,
                "y_pct": y,
                "cost_rm_saved": saved,
                "water_l_saved": water_saved,
                "sac_wins": s["cost_rm"] < b["cost_rm"],
            }
        )
        total_saved_rm += saved
        total_water_saved_l += water_saved
        total_mw_equiv += saved / 1_000_000.0 * 0.12

    if not per_city and results.get("cities"):
        for slug, row in results["cities"].items():
            b, s = row["baseline_RM"], row["SAC_RM"]
            saved = max(0.0, b - s)
            scale = 8760.0 / 512.0
            ann_saved = saved * scale
            per_city.append(
                {
                    "city": slug,
                    "y_pct": round(100 * (1 - s / b), 2) if b else 0,
                    "cost_rm_saved": ann_saved,
                    "water_l_saved": ann_saved * 5.0,
                    "sac_wins": s < b,
                }
            )
            total_saved_rm += ann_saved
            total_water_saved_l += ann_saved * 5.0
        total_mw_equiv = total_saved_rm / 1e6 * 0.12

    mwh_saved = total_mw_equiv * 8760.0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumption_facility_mw_per_site": 10,
        "num_cities": len(CITY_SLUGS),
        "total_mw_saved_estimate": round(total_mw_equiv * len(CITY_SLUGS) / max(1, len(per_city)), 1),
        "total_water_saved_m3_yr": round(total_water_saved_l / 1000.0, 0),
        "total_co2_reduced_tonnes_yr": round(mwh_saved * CO2_KG_PER_MWH / 1000.0 * len(CITY_SLUGS), 0),
        "total_cost_rm_saved_yr": round(total_saved_rm, 0),
        "per_city": per_city,
        "note": "Illustrative SEA aggregate from world-model rollouts at 10 MW reference per city.",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")


def export_city_coords() -> dict:
    coords = {}
    for slug, lat, lon, _tz in LOCATIONS:
        coords[slug] = {"lat": lat, "lon": lon, "label": slug.replace("_", " ").title()}
    return coords


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="Shorter climate episode (720h)")
    ap.add_argument(
        "--episode-hours",
        type=int,
        default=None,
        help="Climate rollout hours (default 8760; --fast uses 720).",
    )
    ap.add_argument("--skip-climate", action="store_true")
    ap.add_argument("--skip-traces", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "data/training/checkpoints/world_model_best.pt",
    )
    ap.add_argument("--rl-dir", type=Path, default=ROOT / "data/rl/per_city_v2")
    ap.add_argument("--projections-dir", type=Path, default=ROOT / "data/weather/projections")
    args = ap.parse_args()

    DASH.mkdir(parents=True, exist_ok=True)
    if args.fast:
        episode_hours = 720
    elif args.episode_hours is not None:
        episode_hours = args.episode_hours
    else:
        episode_hours = 8760
    from run_climate_extended_results import _clamp_episode_hours

    episode_hours = _clamp_episode_hours(episode_hours, 2019)

    if not args.model_path.is_file():
        print(f"ERROR: world model not found at {args.model_path}", file=sys.stderr)
        return 2

    export_model_metrics(DASH / "model_metrics.json")
    export_results(DASH / "results.json")

    if not args.skip_climate:
        print("=== Climate rollouts (base config) ===", flush=True)
        base_cities = _run_base_climate_rollouts(
            projections_dir=args.projections_dir,
            rl_dir=args.rl_dir,
            episode_hours=episode_hours,
            model_path=args.model_path,
            device=args.device,
            action_box="baseline",
        )
        grid = _expand_climate_grid(base_cities, episode_hours)
        grid["generated_at"] = datetime.now(timezone.utc).isoformat()
        grid["episode_hours"] = episode_hours
        grid["calendar_year"] = 2019
        grid["weather_start_row"] = 0
        grid["city_coords"] = export_city_coords()
        _attach_climate_headlines(grid, base_cities)
        (DASH / "climate_results.json").write_text(json.dumps(grid, indent=2) + "\n")
        print(f"Wrote {DASH / 'climate_results.json'}")

    if not args.skip_traces:
        print("=== 24h traces ===", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/export_dashboard_traces.py"),
            "--model-path",
            str(args.model_path),
            "--rl-per-city-dir",
            str(args.rl_dir),
            "--action-box",
            "baseline",
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    climate = {}
    if (DASH / "climate_results.json").is_file():
        climate = json.loads((DASH / "climate_results.json").read_text())
        sample = next(iter(climate.get("cities", {}).values()), {})
        if "headlines_v2" not in climate and sample and "current" in sample:
            _attach_climate_headlines(climate, climate["cities"])
            (DASH / "climate_results.json").write_text(json.dumps(climate, indent=2) + "\n")
            print("Attached headlines_v2 to existing rollout-format climate_results.json")
    results = {}
    if (DASH / "results.json").is_file():
        results = json.loads((DASH / "results.json").read_text())
    export_regional(climate, results, DASH / "regional_impact.json")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
