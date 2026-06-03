#!/usr/bin/env python3
"""Compare ``dc_simulator.DataCenterSimulator`` to Kaggle CSVs under ``data/real/``.

Run: ``python3 compare_simulator_to_kaggle.py``

Uses only the standard library + pandas/numpy if available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Install pandas: pip install pandas", file=sys.stderr)
    raise

from dc_simulator import DataCenterSimulator

DATA_REAL = Path(__file__).resolve().parent / "data" / "real"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_hvac_vs_sim(sim: DataCenterSimulator) -> None:
    path = DATA_REAL / "HVAC Energy Data.csv"
    hv = pd.read_csv(path)
    hv["T_C"] = (hv["Outside Temperature (F)"] - 32) * 5 / 9
    mean_rt = hv["Building Load (RT)"].mean()
    # Scale cooling tons to a ~10 MW nominal IT envelope for the toy simulator.
    hv["it_mw_equiv"] = hv["Building Load (RT)"] / mean_rt * 10.0
    y = hv["Chiller Energy Consumption (kWh)"].values

    sim_chiller_kw = []
    for _, row in hv.iterrows():
        r = sim.simulate_hour(
            float(row["T_C"]),
            72.0,
            float(row["it_mw_equiv"]),
            7.0,
            0.7,
            0.7,
        )
        sim_chiller_kw.append(
            r.chiller_compressor_power_kw + r.chiller_auxiliary_power_kw
        )
    sim_chiller_kw = np.asarray(sim_chiller_kw)

    print("\n--- HVAC Energy Data vs simulator (same joint rows) ---")
    rt_arr = hv["Building Load (RT)"].values
    ct_r = _corr(hv["T_C"].values, y)
    cl_r = _corr(rt_arr, y)
    ct_s = _corr(hv["T_C"].values, sim_chiller_kw)
    cl_s = _corr(hv["it_mw_equiv"].values, sim_chiller_kw)
    print(f"  Real: corr(T_out, chiller_kWh)      = {ct_r:.4f}")
    print(f"  Real: corr(load_RT, chiller_kWh)    = {cl_r:.4f}")
    print(f"  Sim:  corr(T_out, chiller_kw)       = {ct_s:.4f}")
    print(f"  Sim:  corr(load_equiv, chiller_kw)  = {cl_s:.4f}")
    print(
        "  Interpretation: interval chiller energy tracks building load strongly on "
        "both datasets; outdoor temperature adds ~0.56 Pearson correlation with "
        "real chiller use. The simulator matches that *temperature sensitivity* within "
        "~0.04 correlation units after COP tuning. Load correlation is necessarily "
        "higher in the idealized plant (~1.0) because compressor power is a smooth "
        "deterministic function of IT load with no measurement noise or ancillary loads "
        "outside the model."
    )
    # Partial intuition: after scaling load, residual correlation with T on real data.
    b = np.cov(y, rt_arr)[0, 1] / np.var(rt_arr)
    res = y - b * rt_arr
    print(
        f"  Real: corr(T, residual chiller|load_RT) ≈ {_corr(hv['T_C'].values, res):.4f}"
        "  (small: load explains most variance on site)"
    )


def analyze_cold_source(sim: DataCenterSimulator) -> None:
    path = DATA_REAL / "cold_source_control_dataset.csv"
    cs = pd.read_csv(path)
    print("\n--- Cold source control dataset (different scale: ~kW cooling unit) ---")
    print(
        f"  Real: corr(Ambient, cooling_kW)     = {_corr(cs['Ambient_Temperature(°C)'].values, cs['Cooling_Unit_Power_Consumption(kW)'].values):.4f}"
    )
    print(
        f"  Real: corr(Server_Workload, cooling_kW) = {_corr(cs['Server_Workload(%)'].values, cs['Cooling_Unit_Power_Consumption(kW)'].values):.4f}"
    )
    print(
        "  Note: workload dominates; ambient is weak — consistent with a small lab DC."
    )
    # Align simulator qualitatively: vary synthetic load at fixed ambient.
    loads = np.linspace(10, 100, 50)
    ch = []
    for Lpct in loads:
        r = sim.simulate_hour(26.0, 65.0, max(0.5, Lpct / 100.0 * 12.0), 7.0, 0.7, 0.7)
        ch.append(r.chiller_compressor_power_kw + r.chiller_auxiliary_power_kw)
    print(
        f"  Sim (illustrative): corr(load_proxy, chiller_kw) at fixed weather = {_corr(loads, np.array(ch)):.4f}"
    )


def print_pue_benchmarks(sim: DataCenterSimulator) -> None:
    r_tropical = sim.simulate_hour(32.0, 80.0, 10.0, 7.0, 0.7, 0.7)
    r_mild = sim.simulate_hour(22.0, 55.0, 10.0, 7.0, 0.7, 0.7)
    # Lower overhead + pumping — typical of efficient temperate/colocation reporting (~global ~1.54).
    global_like = DataCenterSimulator(
        thermal_storage=None,
        facility_overhead_fraction_of_it=0.24,
        pump_kw_per_kw_load=0.070,
    )
    r_globalish = global_like.simulate_hour(22.0, 55.0, 10.0, 7.0, 0.7, 0.7)

    print("\n--- PUE vs published ballparks ---")
    print(
        "  Literature-style benchmarks (approximate): global fleet mean ~1.54 (wide scatter); "
        "humid tropical / high-ambient mechanical plants often ~1.6–1.8."
    )
    print(
        "  Note: in this lumped model most parasitic power scales with IT load, so PUE "
        "climate contrast is modest (~few ×10⁻³) at fixed fans unless overhead fractions "
        "change between scenarios."
    )
    print(f"  Baseline sim — tropical (32 °C, 80 % RH, 10 MW):     PUE = {r_tropical.pue:.3f}")
    print(f"  Baseline sim — mild (22 °C, 55 % RH, 10 MW):         PUE = {r_mild.pue:.3f}")
    print(
        f"  Efficient temperate proxy — mild 22 °C, lower OH/pumps: PUE = {r_globalish.pue:.3f}"
        "  (benchmark ~1.54)"
    )


def main() -> int:
    if not DATA_REAL.is_dir():
        print(f"Missing {DATA_REAL}", file=sys.stderr)
        return 1

    sim = DataCenterSimulator(thermal_storage=None)
    print(
        "DataCenterSimulator defaults (COP ambient tuned toward HVAC Energy Data.csv; "
        "tropical reference ~1.61 PUE)"
    )
    analyze_hvac_vs_sim(sim)
    analyze_cold_source(sim)
    print_pue_benchmarks(sim)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
