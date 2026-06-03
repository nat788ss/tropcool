#!/usr/bin/env python3
"""TropCool dashboard — precomputed JSON, dark UI, map + live 24h simulation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
DASH = Path(__file__).parent

CITY_LABELS = {
    "cyberjaya": "Cyberjaya",
    "singapore": "Singapore",
    "jakarta": "Jakarta",
    "bangkok": "Bangkok",
    "johor_bahru": "Johor Bahru",
    "ho_chi_minh_city": "Ho Chi Minh City",
}

DEFAULT_COORDS = {
    "cyberjaya": (2.9264, 101.6964),
    "singapore": (1.3521, 103.8198),
    "jakarta": (-6.2088, 106.8456),
    "bangkok": (13.7563, 100.5018),
    "johor_bahru": (1.4927, 103.7414),
    "ho_chi_minh_city": (10.8231, 106.6297),
}

HORIZON_LABELS = {
    "2025": "2025 (current)",
    "2030": "2030",
    "2040": "2040",
    "2050": "2050",
}

COOLING_LABELS = {
    "air_cooled": "Air-cooled",
    "water_cooled": "Water-cooled",
    "hybrid": "Hybrid",
}

LIVE_SIM_DEFAULT_CITY = "ho_chi_minh_city"

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
    }
    .stApp {
        background: linear-gradient(165deg, #F5F0E8 0%, #EDE6DB 40%, #e8dfd2 100%);
        color: #3D3229;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem !important;
        background: linear-gradient(90deg, #3D3229, #C4714A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3, h4 {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
  /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #161b22;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #58a6ff !important;
    }
  /* Panels */
    .panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    .card {
        background: linear-gradient(145deg, #1a2332 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .card-label {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 0.35rem;
    }
    .metric-big {
        font-size: 1.85rem;
        font-weight: 700;
        color: #3fb950;
        line-height: 1.2;
    }
    .headline-card {
        background: linear-gradient(135deg, #1a2a3a 0%, #161b22 100%);
        border-left: 4px solid #58a6ff;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .headline-card strong { color: #e6edf3; }
    .headline-card-stress {
        background: linear-gradient(135deg, #2a1f14 0%, #1c1917 100%);
        border-left: 4px solid #d29922;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1rem 0;
    }
    .headline-card-stress strong { color: #f0c674; }
    .headline-card-stress .metric-big { color: #d29922; }
    .sim-panel-standard {
        background: #1c1917;
        border: 1px solid #3d3330;
        border-radius: 12px;
        padding: 1rem;
    }
    .sim-panel-tropcool {
        background: #0f1f17;
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 1rem;
    }
    .warn-sac { color: #f85149; font-weight: 600; margin: 0; }
    .ok-sac { color: #3fb950; font-weight: 600; margin: 0; }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }
    .stSlider label, .stSelectbox label { color: #c9d1d9 !important; }
    [data-testid="stVerticalBlock"] > div:has(iframe) {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #30363d;
    }
    .stAlert { border-radius: 10px; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    hr.divider { border: none; border-top: 1px solid #30363d; margin: 1.5rem 0; }
</style>
"""

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161b22",
    font=dict(family="DM Sans, system-ui, sans-serif", color="#e6edf3", size=12),
)

PLOTLY_CHART_LAYOUT = {
    **PLOTLY_THEME,
    "margin": dict(l=48, r=24, t=48, b=40),
    "xaxis": dict(gridcolor="#30363d", linecolor="#484f58"),
    "yaxis": dict(gridcolor="#30363d", linecolor="#484f58"),
}


def _section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def _panel_start(title: str) -> None:
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)


def _panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _label(slug: str) -> str:
    return CITY_LABELS.get(slug, slug.replace("_", " ").title())


def _excluded_sac_cities(results: dict) -> frozenset[str]:
    return frozenset(results.get("excluded_cities") or [])


def _sac_recommended(city: str, results: dict, sac_wins: bool | None = None) -> bool:
    if city in _excluded_sac_cities(results):
        return False
    if sac_wins is not None:
        return sac_wins
    row = results.get("cities", {}).get(city, {})
    return bool(row.get("SAC_wins", row.get("sac_wins", True)))


@st.cache_data
def load_json(name: str, _mtime_ns: int = 0) -> dict:
    """Load dashboard JSON; ``_mtime_ns`` busts cache when the file changes on disk."""
    p = DASH / name
    if p.is_file():
        return json.loads(p.read_text())
    return {}


def _json_mtime_ns(name: str) -> int:
    p = DASH / name
    return p.stat().st_mtime_ns if p.is_file() else 0


@st.cache_data
def load_trace(city: str) -> dict | None:
    p = DASH / "traces" / f"{city}_24h.json"
    if p.is_file():
        return json.loads(p.read_text())
    return None


def _interp_scalar(x: float, grid: list[float], values: list[float]) -> float:
    if not grid:
        return values[0] if values else 0.0
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
        return values[-1]
    for i in range(len(grid) - 1):
        if grid[i] <= x <= grid[i + 1]:
            t = (x - grid[i]) / (grid[i + 1] - grid[i])
            return values[i] * (1 - t) + values[i + 1] * t
    return values[-1]


def _nearest_key(value: float, keys: list[str]) -> str:
    floats = [float(k) for k in keys]
    idx = min(range(len(floats)), key=lambda i: abs(floats[i] - value))
    return keys[idx]


HORIZON_TO_HEADLINE_PERIOD = {
    "2025": "current",
    "2030": "2030",
    "2040": "2040",
    "2050": "2050",
}

STRESS_PERIOD_FOR_HORIZON = {
    "2040": "2040_stress",
    "2050": "2050_stress",
}

WARMING_IPCC_C = {"2030": 0.5, "2040": 1.2, "2050": 2.0}
WARMING_STRESS_C = {"2040": 2.5, "2050": 3.0}


def lookup_headline_v2(climate: dict, city: str, horizon: str, *, period: str | None = None) -> dict | None:
    """Climate cost-growth headline for 2030/2040/2050 (world-model rollouts)."""
    key = period or HORIZON_TO_HEADLINE_PERIOD.get(horizon)
    if not key or key == "current":
        return None
    block = climate.get("headlines_v2", {}).get(city)
    if not block:
        return None
    return block.get(key)


def _format_headline_html(
    *,
    title: str,
    headline: dict,
    css_class: str = "headline-card",
) -> str:
    x = headline.get("X_baseline_pct")
    y = headline.get("Y_sac_pct")
    mit = headline.get("mitigation_pp") or headline.get("climate_mitigation_pp")
    sav = headline.get("savings_at_weather_pct")
    y_txt = f"{y:.1f}%" if y is not None else "—"
    mit_txt = f"{mit:.2f} pp" if mit is not None else "—"
    x_txt = f"{x:.1f}%" if x is not None else "—"
    sav_txt = f"{sav:.1f}%" if sav is not None else "—"
    return (
        f'<div class="{css_class}">'
        f"<strong>{title}</strong><br><br>"
        f"Baseline cost rise: <span class='metric-big'>{x_txt}</span> &nbsp;·&nbsp; "
        f"SAC rise: <span class='metric-big'>{y_txt}</span> &nbsp;·&nbsp; "
        f"Mitigation: <span class='metric-big'>{mit_txt}</span><br>"
        f"SAC savings at that weather: <span class='metric-big'>{sav_txt}</span>"
        f"</div>"
    )


def _lookup_climate_legacy_flat(
    climate: dict, city: str, horizon: str, facility_mw: float, cooling: str
) -> dict | None:
    """Old flat ``cities[slug][period]`` JSON (pre-grid export)."""
    period = HORIZON_TO_HEADLINE_PERIOD.get(horizon, horizon)
    if period == "current":
        period = "current"
    branch = climate.get("cities", {}).get(city, {})
    cell = branch.get(period) or branch.get(horizon)
    if not cell or "baseline" not in cell:
        return None
    mw_f = facility_mw / 10.0
    cs = {"water_cooled": 1.0, "air_cooled": 1.04, "hybrid": 1.02}.get(cooling, 1.0)

    def _scale(raw: dict) -> dict:
        cost = float(raw.get("cost_RM_annualized") or raw.get("cost_RM", 0)) * mw_f * cs
        water = float(raw.get("water_l", 0)) * mw_f
        if water <= 0:
            water = cost * 8.0
        return {
            "pue": float(raw.get("mean_pue", raw.get("pue", 1.65))),
            "cost_rm": cost,
            "water_l": water,
        }

    b = _scale(cell["baseline"])
    s = _scale(cell["sac"]) if cell.get("sac") else b
    y = 100.0 * (1.0 - s["cost_rm"] / b["cost_rm"]) if b["cost_rm"] > 0 else 0.0
    return {"baseline": b, "sac": s, "y_pct": y}


def lookup_climate(
    climate: dict,
    city: str,
    facility_mw: float,
    cooling: str,
    rack_density: float,
    horizon: str,
) -> dict | None:
    """Bilinear-style interpolation over precomputed MW × rack grids."""
    cities = climate.get("cities", {})
    if city not in cities:
        return None
    meta = climate.get("meta", {})
    # Legacy flat file: ``cities[slug][period]`` instead of ``cities[slug][mw][cooling]...``.
    sample = cities[city]
    if sample and not meta.get("facility_mw_grid"):
        first_key = next(iter(sample))
        if first_key in ("current", "2030", "2040", "2050", "2025"):
            return _lookup_climate_legacy_flat(climate, city, horizon, facility_mw, cooling)
    mw_grid = sorted(float(x) for x in meta.get("facility_mw_grid", [5, 10, 20, 30, 50]))
    rack_keys = sorted(float(x) for x in meta.get("rack_density_keys", [0.3, 0.5, 0.65, 0.8, 1.0]))
    mw_keys = [str(int(m)) if m == int(m) else str(m) for m in mw_grid]

    def _cell(mw_s: str, rk_s: str) -> dict | None:
        branch = cities[city].get(mw_s, {}).get(cooling, {}).get(rk_s, {})
        return branch.get(horizon)

    def _at_mw_rack(mw: float, rk: float) -> dict | None:
        mw_s = _nearest_key(mw, mw_keys)
        rk_s = _nearest_key(rk, [str(k) for k in rack_keys])
        return _cell(mw_s, rk_s)

    # Interpolate along rack at low/high MW, then along MW
    mw_lo = max(k for k in mw_grid if k <= facility_mw) if facility_mw >= mw_grid[0] else mw_grid[0]
    mw_hi = min(k for k in mw_grid if k >= facility_mw) if facility_mw <= mw_grid[-1] else mw_grid[-1]
    rk_lo = max(k for k in rack_keys if k <= rack_density)
    rk_hi = min(k for k in rack_keys if k >= rack_density)

    corners = []
    for mw in (mw_lo, mw_hi):
        row_rk = []
        for rk in (rk_lo, rk_hi):
            c = _at_mw_rack(mw, rk)
            if c:
                row_rk.append(c)
        if len(row_rk) == 2:
            t_rk = (rack_density - rk_lo) / (rk_hi - rk_lo) if rk_hi > rk_lo else 0.0
            merged = {}
            for pol in ("baseline", "sac"):
                merged[pol] = {
                    k: row_rk[0][pol][k] * (1 - t_rk) + row_rk[1][pol][k] * t_rk
                    for k in ("pue", "cost_rm", "water_l")
                }
            merged["y_pct"] = row_rk[0].get("y_pct", 0) * (1 - t_rk) + row_rk[1].get("y_pct", 0) * t_rk
            corners.append((mw, merged))
        elif len(row_rk) == 1:
            corners.append((mw, row_rk[0]))

    if not corners:
        return _at_mw_rack(facility_mw, rack_density)

    if len(corners) == 1:
        return corners[0][1]

    t_mw = (facility_mw - corners[0][0]) / (corners[1][0] - corners[0][0]) if corners[1][0] > corners[0][0] else 0.0
    c0, c1 = corners[0][1], corners[1][1]
    out = {
        "baseline": {
            k: c0["baseline"][k] * (1 - t_mw) + c1["baseline"][k] * t_mw for k in ("pue", "cost_rm", "water_l")
        },
        "sac": {k: c0["sac"][k] * (1 - t_mw) + c1["sac"][k] * t_mw for k in ("pue", "cost_rm", "water_l")},
        "y_pct": c0.get("y_pct", 0) * (1 - t_mw) + c1.get("y_pct", 0) * t_mw,
    }
    return out


def build_map(coords: dict, selected: str) -> folium.Map:
    fmap = folium.Map(
        location=[8.0, 108.0],
        zoom_start=5,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="CartoDB · OpenStreetMap",
    )
    for slug, (lat, lon) in coords.items():
        is_sel = slug == selected
        color = "#C4714A" if is_sel else "#6B8F71"
        folium.CircleMarker(
            location=[lat, lon],
            radius=12 if is_sel else 8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(_label(slug), max_width=200),
            tooltip=_label(slug),
        ).add_to(fmap)
    return fmap


def pue_gauge(fig_title: str, pue: float) -> go.Figure:
    pue = float(max(1.0, min(2.2, pue)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pue,
            number={"suffix": "", "font": {"color": "#e6edf3"}},
            title={"text": fig_title, "font": {"color": "#8b949e"}},
            gauge={
                "axis": {"range": [1.0, 2.2], "tickcolor": "#8b949e"},
                "bar": {"color": "#3fb950" if pue < 1.55 else "#d29922"},
                "bgcolor": "#1a1f2e",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [1.0, 1.5], "color": "#21262d"},
                    {"range": [1.5, 2.2], "color": "#30363d"},
                ],
            },
        )
    )
    fig.update_layout(
        **PLOTLY_THEME,
        height=200,
        margin=dict(l=20, r=20, t=36, b=8),
    )
    return fig


def tab_explore(climate: dict, results: dict) -> None:
    _section("Explore", "Select a site, configure the facility, and compare baseline vs TropCool SAC.")
    coords = climate.get("city_coords") or {
        s: {"lat": lat, "lon": lon} for s, (lat, lon) in DEFAULT_COORDS.items()
    }
    city_list = list(coords.keys()) if coords else list(DEFAULT_COORDS.keys())

    if "selected_city" not in st.session_state:
        preferred = LIVE_SIM_DEFAULT_CITY
        st.session_state.selected_city = preferred if preferred in city_list else city_list[0]

    col_map, col_cfg = st.columns([1.4, 1], gap="large")

    with col_map:
        _panel_start("Southeast Asia — data center sites")
        coord_pairs = {
            s: (coords[s]["lat"], coords[s]["lon"])
            if isinstance(coords[s], dict)
            else DEFAULT_COORDS[s]
            for s in city_list
        }
        fmap = build_map(coord_pairs, st.session_state.selected_city)
        picked = st_folium(
            fmap,
            width=None,
            height=420,
            returned_objects=["last_object_clicked"],
            key="sea_map",
        )
        clicked = picked.get("last_object_clicked") if picked else None
        if isinstance(clicked, dict) and "lat" in clicked and "lon" in clicked:
            lat, lon = float(clicked["lat"]), float(clicked["lon"])
            for slug, (clat, clon) in coord_pairs.items():
                if abs(clat - lat) < 0.35 and abs(clon - lon) < 0.35:
                    st.session_state.selected_city = slug
                    break

        st.selectbox(
            "City",
            city_list,
            format_func=_label,
            key="selected_city",
        )
        _panel_end()

    with col_cfg:
        _panel_start("Facility configuration")
        facility_mw = st.slider("Facility IT load (MW)", 5, 50, 20)
        cooling = st.selectbox(
            "Cooling type",
            list(COOLING_LABELS.keys()),
            format_func=lambda k: COOLING_LABELS[k],
            index=1,
        )
        rack_density = st.slider(
            "Rack density (normalized)",
            0.3,
            1.0,
            0.65,
            0.05,
            help="Maps to ~5–25 kW/rack: kW = 5 + (density − 0.3) × (20/0.7)",
        )
        rack_kw = 5.0 + (rack_density - 0.3) * (20.0 / 0.7)
        st.caption(f"≈ {rack_kw:.1f} kW/rack equivalent heat density")
        horizon = st.selectbox(
            "Time horizon",
            list(HORIZON_LABELS.keys()),
            format_func=lambda h: HORIZON_LABELS[h],
        )
        _panel_end()

    headline_city = st.session_state.selected_city
    headline = lookup_headline_v2(climate, headline_city, horizon)
    if headline and horizon in ("2030", "2040", "2050"):
        warm_h = climate.get("meta", {}).get("horizons", {}).get(horizon) or WARMING_IPCC_C.get(horizon)
        ipcc_label = f"IPCC-style (+{warm_h}°C)" if warm_h is not None else "IPCC central"
        st.markdown(
            _format_headline_html(
                title=f"{_label(headline_city)} · {horizon} · {ipcc_label}",
                headline=headline,
            ),
            unsafe_allow_html=True,
        )
        pitch = headline.get("pitch", "")
        if pitch:
            st.markdown(f"*{pitch}*")
        sav_pitch = headline.get("savings_pitch")
        if sav_pitch:
            st.markdown(f"*{sav_pitch}*")
        stress_key = STRESS_PERIOD_FOR_HORIZON.get(horizon)
        if stress_key:
            stress_hl = lookup_headline_v2(climate, headline_city, horizon, period=stress_key)
            if stress_hl:
                stress_w = WARMING_STRESS_C.get(horizon) or stress_hl.get("warming_c")
                stress_label = (
                    f"Stress scenario (+{stress_w}°C)" if stress_w is not None else "High-stress scenario"
                )
                st.markdown(
                    _format_headline_html(
                        title=f"{_label(headline_city)} · {horizon} · {stress_label}",
                        headline=stress_hl,
                        css_class="headline-card-stress",
                    ),
                    unsafe_allow_html=True,
                )
        st.caption(
            "Predicted annualized cooling cost from world-model rollouts on synthetic warming weather "
            "(not metered operations). "
            "**SAC savings at that weather** = baseline vs SAC on the same future CSV. "
            "Five-city validated SAC portfolio mean excludes Jakarta."
        )
        port_period = HORIZON_TO_HEADLINE_PERIOD.get(horizon, horizon)
        port = climate.get("headlines_v2", {}).get("validated_portfolio_mean", {}).get(port_period)
        if port and headline_city != "validated_portfolio_mean":
            px, py, pm = port.get("X_baseline_pct"), port.get("Y_sac_pct"), port.get("mitigation_pp")
            psav = port.get("savings_at_weather_pct")
            st.caption(
                f"Validated 5-city portfolio (excl. Jakarta) for {horizon} IPCC central: "
                f"baseline +{px:.1f}%, SAC +{py:.1f}%, mitigation {pm:.2f} pp, "
                f"SAC savings at weather {psav:.1f}%."
                if psav is not None
                else f"Validated 5-city portfolio (excl. Jakarta) for {horizon}: "
                f"baseline +{px:.1f}%, SAC +{py:.1f}%, mitigation {pm:.2f} pp."
            )
    elif horizon == "2025" and results.get("cities"):
        slug = headline_city
        if slug in results["cities"]:
            r = results["cities"][slug]
            b, s = r["baseline_RM"], r["SAC_RM"]
            y_now = 100 * (1 - s / b) if b else 0
            st.markdown(
                f'<div class="headline-card">'
                f"<strong>{_label(slug)} · 2025 (current weather)</strong><br><br>"
                f"SAC savings at current weather: <span class='metric-big'>{y_now:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("From strict eval window (`results.json`); not a climate growth scenario.")

    cell = lookup_climate(
        climate,
        st.session_state.selected_city,
        facility_mw,
        cooling,
        rack_density,
        horizon,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    _section("Results", f"{_label(st.session_state.selected_city)} · {HORIZON_LABELS.get(horizon, horizon)}")
    if not cell:
        st.warning(
            "No climate results for this selection. Regenerate data, then refresh the app "
            "(Streamlit caches JSON — use **⋮ → Rerun** or restart after export):\n\n"
            "`python3 scripts/export_dashboard_data.py`"
        )
        if results.get("cities"):
            slug = st.session_state.selected_city
            if slug in results["cities"]:
                r = results["cities"][slug]
                b, s = r["baseline_RM"], r["SAC_RM"]
                y = 100 * (1 - s / b) if b else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Baseline cost (eval window)", f"{b:,.0f} RM")
                c2.metric("SAC cost", f"{s:,.0f} RM")
                c3.metric("Savings Y", f"{y:.1f}%")
                c4.metric("SAC wins", "Yes" if s < b else "No")
        return

    b, s = cell["baseline"], cell["sac"]
    y = cell.get("y_pct", 0)
    sac_wins = s["cost_rm"] < b["cost_rm"]
    city = st.session_state.selected_city
    sac_ok = _sac_recommended(city, results, sac_wins)

    if not sac_ok:
        st.warning(
            "Baseline recommended — SAC underperforms on eval contract "
            "(excluded from validated 5-city SAC portfolio)."
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Baseline PUE", f"{b['pue']:.3f}")
    if sac_ok:
        m2.metric("TropCool PUE", f"{s['pue']:.3f}")
        m3.metric("Annual cost", f"{s['cost_rm']:,.0f} RM", delta=f"{y:.1f}% vs baseline", delta_color="normal")
        m4.metric("Water / yr", f"{s['water_l']:,.0f} L", delta=f"{(s['water_l']-b['water_l'])/max(b['water_l'],1)*100:.1f}%")
        m5.metric("Savings Y", f"{y:.1f}%")
    else:
        m2.metric("TropCool PUE", "—")
        m3.metric("Annual cost", f"{b['cost_rm']:,.0f} RM")
        m4.metric("Water / yr", f"{b['water_l']:,.0f} L")
        m5.metric("Savings Y", "—")
    cls = "ok-sac" if sac_ok else "warn-sac"
    st.markdown(
        f'<p class="{cls}">'
        f'{"✓ SAC recommended on strict eval" if sac_ok else "⚠ Baseline recommended — SAC excluded from 5-city portfolio"}'
        f"</p>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=["PUE"],
            y=[b["pue"]],
            marker_color="#8b949e",
        )
    )
    if sac_ok:
        fig.add_trace(
            go.Bar(
                name="TropCool SAC",
                x=["PUE"],
                y=[s["pue"]],
                marker_color="#3fb950",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                name="TropCool SAC (not recommended)",
                x=["PUE"],
                y=[s["pue"]],
                marker_color="#484f58",
                opacity=0.45,
            )
        )
    fig.update_layout(
        barmode="group",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **PLOTLY_CHART_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    warm = climate.get("meta", {}).get("horizons", {}).get(horizon)
    if warm and float(warm) > 0:
        st.caption(f"Climate scenario: +{warm}°C dry-bulb vs 2025 baseline (synthetic projection).")


def _render_sim_frame(
    base_rows: list,
    sac_rows: list,
    temps: list,
    h: int,
    n: int,
) -> None:
    br, sr = base_rows[h], sac_rows[h]
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<div class="sim-panel-standard"><p style="margin:0 0 0.5rem;font-weight:600;">Standard DC</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(pue_gauge("PUE", float(br.get("pue", 1.7))), width="stretch", key="bpue_live")
        c1, c2 = st.columns(2)
        c1.metric("Cost (cum.)", f"{float(br.get('cumulative_cost_rm', 0)):,.0f} RM")
        c2.metric("Water (cum.)", f"{float(br.get('cumulative_water_l', 0)):,.0f} L")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="sim-panel-tropcool"><p style="margin:0 0 0.5rem;font-weight:600;">TropCool DC (SAC)</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(pue_gauge("PUE", float(sr.get("pue", 1.7))), width="stretch", key="spue_live")
        c1, c2 = st.columns(2)
        c1.metric("Cost (cum.)", f"{float(sr.get('cumulative_cost_rm', 0)):,.0f} RM")
        c2.metric("Water (cum.)", f"{float(sr.get('cumulative_water_l', 0)):,.0f} L")
        st.markdown("</div>", unsafe_allow_html=True)
    temp = temps[h] if h < len(temps) else None
    if temp is not None:
        st.caption(f"Hour {h + 1} / {n} — outdoor {float(temp):.1f}°C")
    st.progress((h + 1) / n, text=f"Simulated hour {h + 1} of {n}")


def tab_live_sim() -> None:
    _section(
        "Live simulation",
        "Hottest summer day (24 h) — side-by-side baseline vs SAC.",
    )
    st.caption(
        "One **stress day** (max dry-bulb in 2019) inside the **world model**, not the "
        "27-scenario strict eval. Hourly SAC can look worse while **end-of-day cumulative** "
        "cost is better — compare the final summary. Prefer **Ho Chi Minh, Singapore, or Bangkok** "
        "for a positive demo; Cyberjaya/Jakarta often lose on this day."
    )
    for key, default in (("sim_running", False), ("sim_hour", 0), ("sim_done", False)):
        if key not in st.session_state:
            st.session_state[key] = default

    cities = [c for c in CITY_LABELS if load_trace(c)]
    if not cities:
        st.warning("Export traces: `python3 scripts/export_dashboard_data.py`")
        return

    default_idx = cities.index(LIVE_SIM_DEFAULT_CITY) if LIVE_SIM_DEFAULT_CITY in cities else 0
    city = st.selectbox(
        "City",
        cities,
        index=default_idx,
        format_func=_label,
        key="sim_city",
    )
    if city == "jakarta":
        st.caption(
            "Jakarta: baseline recommended on strict multi-scenario eval — "
            "SAC trace shown for illustration only."
        )
    trace = load_trace(city)
    if not trace:
        return

    base_rows = trace.get("baseline", [])
    sac_rows = trace.get("sac", []) or base_rows
    n = min(len(base_rows), len(sac_rows), 24)
    if n == 0:
        st.error("Trace has no hourly rows. Re-run `python3 scripts/export_dashboard_traces.py`.")
        return
    temps = trace.get("outdoor_temp_c", [None] * n)

    auto = st.checkbox("Auto-advance (1 h/sec)", value=False, key="sim_auto")
    c_play, c_reset = st.columns(2)
    with c_play:
        play = st.button("▶ Play 24-hour simulation", type="primary", key="sim_play")
    with c_reset:
        if st.button("Reset", key="sim_reset"):
            st.session_state.sim_running = False
            st.session_state.sim_done = False
            st.session_state.sim_hour = 0
            st.rerun()

    if play:
        st.session_state.sim_running = True
        st.session_state.sim_done = False
        st.session_state.sim_hour = 0
        st.rerun()

    if st.session_state.sim_done:
        savings = float(trace.get("daily_savings_pct") or 0.0)
        b_end = float(base_rows[-1].get("cumulative_cost_rm", 0))
        s_end = float(sac_rows[-1].get("cumulative_cost_rm", 0))
        sac_wins = savings > 0 and s_end < b_end
        msg = (
            f"**End of day (cumulative)** — SAC total **{s_end:,.0f} RM** vs baseline "
            f"**{b_end:,.0f} RM** → **{savings:.1f}%** savings"
            if sac_wins
            else f"**End of day** — SAC cumulative cost not below baseline ({savings:.1f}%). "
            f"Try another city or the Explore tab (multi-scenario eval)."
        )
        if sac_wins:
            st.success(msg)
        else:
            st.warning(msg)
        return

    if not st.session_state.sim_running:
        st.info("Press **Play** to start the 24-hour side-by-side run (1 simulated hour per second).")
        return

    h = int(st.session_state.sim_hour)
    if h >= n:
        st.session_state.sim_running = False
        st.session_state.sim_done = True
        st.session_state.sim_hour = 0
        st.rerun()

    _render_sim_frame(base_rows, sac_rows, temps, h, n)

    if auto:
        time.sleep(1.0)
        st.session_state.sim_hour = h + 1
        st.rerun()
    elif st.button("Next hour →", key="sim_next"):
        st.session_state.sim_hour = h + 1
        st.rerun()


def tab_regional(regional: dict) -> None:
    _section("Regional impact", "Illustrative Southeast Asia aggregate (world-model rollouts).")
    if not regional:
        st.warning("Run export to generate `dashboard/regional_impact.json`.")
        return

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, val, lbl in (
        (c1, regional.get("total_mw_saved_estimate", 0), "MW grid capacity recovered (est.)"),
        (c2, regional.get("total_water_saved_m3_yr", 0), "m³ water saved / year (est.)"),
        (c3, regional.get("total_co2_reduced_tonnes_yr", 0), "tonnes CO₂ avoided / year (est.)"),
    ):
        col.markdown(
            f'<div class="card"><div class="metric-big">{val:,.0f}</div>'
            f'<div class="card-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.metric(
        "Annual cost savings (10 MW reference per site)",
        f"{regional.get('total_cost_rm_saved_yr', 0):,.0f} RM",
    )
    st.caption(
        (regional.get("note", "") or "")
        + " Headline SAC portfolio: five validated cities (Cyberjaya, Singapore, Bangkok, "
        "Johor Bahru, Ho Chi Minh City); Jakarta shown baseline-only on strict eval."
    )

    per = regional.get("per_city", [])
    if per:
        df = pd.DataFrame(per)
        df["city"] = df["city"].map(_label)
        df = df.rename(
            columns={
                "y_pct": "Savings Y (%)",
                "cost_rm_saved": "Cost saved (RM/yr)",
                "sac_wins": "SAC wins",
            }
        )
        st.dataframe(df, width="stretch", hide_index=True)


def tab_how_it_works() -> None:
    _section("How it works", "Data pipeline and model architecture.")
    st.markdown(
        """
#### Pipeline

1. **Real weather** — 20 years of hourly Open-Meteo data for six tropical cities (`data/weather/`).
2. **Physics simulator** — calibrated chiller / tower / CRAH model generates millions of training hours.
3. **Transformer world model** — predicts next-hour rack temperature, cooling power, water, PUE, and cost.
4. **RL agent (SAC)** — trains inside the world model to minimize cost while keeping rack inlet ≤ 27°C.
5. **Evaluate** — `compare_policies.py` runs baseline vs per-city SAC on **27 scenarios** (3 years × 3 start rows × 3 seeds, 512 h); see `data/rl/per_city_v2_strict_eval_5city.log`.

#### Climate projections (synthetic future weather)

*Not CMIP6 — warmed copies of historical CSVs, then scored in the world model.*

- `scripts/generate_climate_weather.py` → `data/weather/projections/` (e.g. `singapore_2040.csv` +1.2°C, `singapore_2040_stress.csv` +2.5°C)
- `scripts/run_climate_extended_results.py` → `dashboard/climate_results.json`
- **Explore** tab horizons 2030 / 2040 / 2050 use `headlines_v2` (see climate headline cards)

6. **Dashboard** — precomputed JSON powers Explore, Live Simulation traces, and regional rollups.

### Architecture
"""
    )
    st.markdown(
        """
```mermaid
flowchart LR
  W[Open-Meteo weather] --> S[DC simulator]
  S --> T[Training parquet]
  T --> WM[Transformer world model]
  WM --> ENV[WorldModelEnv]
  ENV --> SAC[SAC per_city_v2]
  SAC --> E[compare_policies]
  E --> R[results.json]
  W --> G[generate_climate_weather]
  G --> P[projections CSVs]
  P --> C[climate rollouts]
  C --> CR[climate_results.json]
  R --> D[Dashboard]
  CR --> D
  WM --> D
```
"""
    )
    st.caption(
        "Explore climate cards use headlines_v2 from climate_results.json. "
        "Precomputed rollouts — no live backend at demo time."
    )


def tab_model_accuracy(metrics: dict) -> None:
    _section("Model accuracy", "World model test / fine-tune holdout metrics.")
    targets = metrics.get("targets", {})
    if not targets:
        st.warning("Generate `dashboard/model_metrics.json` via export script.")
        return
    rows = []
    for name, m in targets.items():
        rows.append(
            {
                "target": name,
                "R²": m.get("r2"),
                "MAE": m.get("mae"),
                "RMSE": m.get("rmse"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(metrics.get("note", metrics.get("source", "")))

    fig = go.Figure(
        go.Bar(
            x=df["target"],
            y=df["R²"],
            marker_color=["#3fb950" if (v or 0) > 0.85 else "#d29922" for v in df["R²"]],
        )
    )
    fig.update_layout(title="R² by target", height=360, **PLOTLY_CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="TropCool",
        layout="wide",
        page_icon="🌴",
        initial_sidebar_state="collapsed",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.title("TropCool")
    st.markdown(
        '<p class="hero-sub">Tropical data-center cooling · transformer world model · Soft Actor-Critic (SAC)</p>',
        unsafe_allow_html=True,
    )

    climate = load_json("climate_results.json", _mtime_ns=_json_mtime_ns("climate_results.json"))
    results = load_json("results.json", _mtime_ns=_json_mtime_ns("results.json"))
    regional = load_json("regional_impact.json", _mtime_ns=_json_mtime_ns("regional_impact.json"))
    metrics = load_json("model_metrics.json", _mtime_ns=_json_mtime_ns("model_metrics.json"))

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Explore",
            "Live Simulation",
            "Regional Impact",
            "How It Works",
            "Model accuracy",
        ]
    )
    with t1:
        tab_explore(climate, results)
    with t2:
        tab_live_sim()
    with t3:
        tab_regional(regional)
    with t4:
        tab_how_it_works()
    with t5:
        tab_model_accuracy(metrics)


if __name__ == "__main__":
    main()
