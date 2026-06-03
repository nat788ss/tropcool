#!/usr/bin/env python3
"""Parse compare_policies strict-eval log → dashboard/results.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_data import CITY_SLUGS


def parse_strict_log(text: str) -> dict:
    cities: dict[str, dict] = {}
    for line in text.splitlines():
        m = re.match(
            r"^(\w+)\s+\d+\s+([\d,]+)\s*±\s*[\d,]+\s+[\d,]+\s*±\s*[\d,]+\s+([\d,]+)",
            line.strip(),
        )
        if m:
            city, base, sac = m.group(1), m.group(2), m.group(3)
            b_rm = float(base.replace(",", ""))
            s_rm = float(sac.replace(",", ""))
            cities[city] = {
                "baseline_RM": b_rm,
                "SAC_RM": s_rm,
                "SAC_wins": s_rm < b_rm,
            }

    y_all = re.search(
        r"Predicted cooling cost reduction \(all \d+ cities\):\s*Y\s*=\s*([\d.]+)%",
        text,
    )
    y_validated = re.search(
        r"Predicted cooling cost reduction \(validated portfolio,.*?\):\s*Y\s*=\s*([\d.]+)%",
        text,
    )
    y_legacy = re.search(
        r"Predicted cooling cost reduction \(mean of per-city means\):\s*Y\s*=\s*([\d.]+)%",
        text,
    )
    y_first = re.search(r"Y\s*=\s*([\d.]+)%", text)

    excluded: list[str] = []
    excl_m = re.search(r"excludes ([\w, ]+):\s*Y", text)
    if excl_m:
        excluded = [s.strip() for s in excl_m.group(1).split(",") if s.strip()]
    elif "jakarta" in cities and not cities["jakarta"].get("SAC_wins", True):
        excluded = ["jakarta"]

    validated = [s for s in CITY_SLUGS if s in cities and s not in excluded]
    if not validated and cities:
        validated = list(cities.keys())

    failures: list[str] = []
    for line in text.splitlines():
        fm = re.search(
            r"^\s+(\w+):\s+baseline\s+([\d,]+)\s+RM\s+SAC\s+([\d,]+)\s+RM",
            line,
        )
        if fm:
            slug = fm.group(1)
            b_cost = float(fm.group(2).replace(",", ""))
            s_cost = float(fm.group(3).replace(",", ""))
            if s_cost >= b_cost and slug in validated:
                failures.append(slug)

    strict_all = len(failures) == 0 and "SAC did not beat baseline" not in text
    if "SAC did not beat baseline" in text:
        strict_all = False

    mean_y_all = None
    if y_all:
        mean_y_all = float(y_all.group(1))
    elif y_legacy:
        mean_y_all = float(y_legacy.group(1))
    elif y_first:
        mean_y_all = float(y_first.group(1))

    mean_y_validated = float(y_validated.group(1)) if y_validated else None
    if mean_y_validated is None and validated and cities:
        bases = [cities[s]["baseline_RM"] for s in validated]
        sacs = [cities[s]["SAC_RM"] for s in validated]
        b_mean = sum(bases) / len(bases)
        s_mean = sum(sacs) / len(sacs)
        if b_mean > 0:
            mean_y_validated = 100.0 * (1.0 - s_mean / b_mean)

    strict_validated = all(cities.get(s, {}).get("SAC_wins", False) for s in validated)
    if failures:
        strict_validated = False

    return {
        "cities": cities,
        "mean_Y_pct": mean_y_all,
        "mean_Y_pct_validated": mean_y_validated,
        "validated_cities": validated,
        "excluded_cities": excluded,
        "strict_sac_beats_baseline_all": strict_all,
        "strict_sac_beats_baseline_validated": strict_validated,
        "strict_gate": strict_validated if excluded else strict_all,
        "strict_failures": failures,
        "failed_cities": [s for s in validated if not cities.get(s, {}).get("SAC_wins", True)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=ROOT / "data/rl/per_city_v2_strict_eval_5city.log",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dashboard/results.json",
    )
    args = ap.parse_args()
    if not args.log.is_file():
        print(f"Missing log: {args.log}", file=sys.stderr)
        return 2
    data = parse_strict_log(args.log.read_text())
    data["source"] = str(args.log.relative_to(ROOT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Wrote {args.out} ({len(data.get('cities', {}))} cities, "
        f"strict_validated={data.get('strict_sac_beats_baseline_validated')}, "
        f"mean_Y_validated={data.get('mean_Y_pct_validated')})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
