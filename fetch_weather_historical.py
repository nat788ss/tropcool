#!/usr/bin/env python3
"""Fetch hourly weather from the Open-Meteo Historical Weather API and save one CSV per city.

**Cities (lat, lon) and local timezones:** Cyberjaya, Singapore, Jakarta, Bangkok,
Johor Bahru, Ho Chi Minh City — see ``LOCATIONS``.

**Period:** default **2004-01-01** through **2024-12-31** (20 years of hourly data).

**Variables:** ``temperature_2m``, ``relative_humidity_2m``, and dew point. The API returns
``dew_point_2m``; it is written as **dewpoint_temperature_2m** in the CSV to match your naming.

**Output:** ``data/weather/{slug}_hourly_{start_year}_{end_year}.csv`` with columns
``time``, the three variables above, and optionally ``wet_bulb_temperature_2m`` if
``--include-wet-bulb`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from weather_utils import wet_bulb_temperature_celsius

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# slug, lat, lon, IANA timezone for hourly timestamps
LOCATIONS: tuple[tuple[str, float, float, str], ...] = (
    ("cyberjaya", 2.9264, 101.6964, "Asia/Kuala_Lumpur"),
    ("singapore", 1.3521, 103.8198, "Asia/Singapore"),
    ("jakarta", -6.2088, 106.8456, "Asia/Jakarta"),
    ("bangkok", 13.7563, 100.5018, "Asia/Bangkok"),
    ("johor_bahru", 1.4927, 103.7414, "Asia/Kuala_Lumpur"),
    ("ho_chi_minh_city", 10.8231, 106.6297, "Asia/Ho_Chi_Minh"),
)

# Open-Meteo hourly keys; dew point is dew_point_2m in JSON — we export as dewpoint_temperature_2m.
API_HOURLY = ("temperature_2m", "relative_humidity_2m", "dew_point_2m")


def _fetch_chunk(
    lat: float, lon: float, timezone: str, start: date, end: date
) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(API_HOURLY),
        "timezone": timezone,
    }
    url = f"{ARCHIVE_URL}?{urlencode(params)}"
    with urlopen(url, timeout=300, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())


def fetch_hourly_range(
    lat: float,
    lon: float,
    timezone: str,
    start: date,
    end: date,
    *,
    chunk_days: int = 366,
    pause_sec: float = 0.35,
    include_wet_bulb: bool = False,
) -> list[dict[str, float | str | None]]:
    """Merge hourly rows across [start, end] using chunked API calls."""
    rows: list[dict[str, float | str | None]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        data = _fetch_chunk(lat, lon, timezone, cur, chunk_end)
        if data.get("error"):
            raise RuntimeError(str(data.get("reason", data)))
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        t2 = hourly.get("temperature_2m") or []
        rh = hourly.get("relative_humidity_2m") or []
        dp = hourly.get("dew_point_2m") or []
        for i, ts in enumerate(times):
            ta = t2[i] if i < len(t2) else None
            rhp = rh[i] if i < len(rh) else None
            dpt = dp[i] if i < len(dp) else None
            row: dict[str, float | str | None] = {
                "time": ts,
                "temperature_2m": ta,
                "relative_humidity_2m": rhp,
                "dewpoint_temperature_2m": dpt,
            }
            if include_wet_bulb:
                wb = None
                if ta is not None and rhp is not None:
                    try:
                        wb = round(
                            wet_bulb_temperature_celsius(float(ta), float(rhp)), 4
                        )
                    except (TypeError, ValueError):
                        wb = None
                row["wet_bulb_temperature_2m"] = wb
            rows.append(row)
        cur = chunk_end + timedelta(days=1)
        if cur <= end and pause_sec > 0:
            time.sleep(pause_sec)
    return rows


def write_city_csv(
    out_dir: str,
    slug: str,
    rows: list[dict[str, float | str | None]],
    start: date,
    end: date,
    *,
    include_wet_bulb: bool,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"{slug}_hourly_{start.year}_{end.year}.csv"
    )
    fieldnames: tuple[str, ...] = (
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_temperature_2m",
    )
    if include_wet_bulb:
        fieldnames = fieldnames + ("wet_bulb_temperature_2m",)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default=os.path.join("data", "weather"),
        help="Directory for CSV files (default: data/weather)",
    )
    p.add_argument("--start", type=date.fromisoformat, default=date(2004, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument(
        "--city",
        action="append",
        help="Slug(s) only, e.g. cyberjaya,singapore. Default: all six cities.",
    )
    p.add_argument("--chunk-days", type=int, default=366, help="Days per API request.")
    p.add_argument(
        "--pause",
        type=float,
        default=0.35,
        help="Seconds between chunk requests (rate courtesy).",
    )
    p.add_argument(
        "--include-wet-bulb",
        action="store_true",
        help="Add computed wet_bulb_temperature_2m column (Stull 2011).",
    )
    args = p.parse_args()

    if args.start > args.end:
        print("error: --start must be on or before --end", file=sys.stderr)
        return 1

    want = None
    if args.city:
        want = set()
        for part in args.city:
            for name in part.split(","):
                n = name.strip().lower().replace("-", "_")
                if n:
                    want.add(n)

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    locs = LOCATIONS
    if want is not None:
        locs = tuple(loc for loc in LOCATIONS if loc[0] in want)
        missing = want - {loc[0] for loc in locs}
        if missing:
            known = ", ".join(s[0] for s in LOCATIONS)
            print(f"unknown city slug(s): {missing}. Known: {known}", file=sys.stderr)
            return 1

    for slug, lat, lon, tz in locs:
        print(f"Fetching {slug} ({lat}, {lon}) {args.start} .. {args.end} …")
        try:
            rows = fetch_hourly_range(
                lat,
                lon,
                tz,
                args.start,
                args.end,
                chunk_days=args.chunk_days,
                pause_sec=args.pause,
                include_wet_bulb=args.include_wet_bulb,
            )
        except (HTTPError, URLError, OSError, json.JSONDecodeError, RuntimeError) as e:
            print(f"{slug}: failed: {e}", file=sys.stderr)
            return 1
        path = write_city_csv(
            out_dir,
            slug,
            rows,
            args.start,
            args.end,
            include_wet_bulb=args.include_wet_bulb,
        )
        print(f"  wrote {len(rows):,} rows -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
