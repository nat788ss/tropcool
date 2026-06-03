#!/usr/bin/env python3
"""Least-squares fit of COP ambient parameters to HVAC Kaggle joint statistics.

Fits ``outdoor_cop_penalty_per_k`` and ``design_outdoor_temp_c`` so that, when the
simulator is evaluated on the same (T, load) rows as ``HVAC Energy Data.csv``,
Pearson correlations of simulated chiller power vs T and vs load match the file's
correlations of interval chiller energy vs T and vs building load.

Also constrains the tropical reference case PUE (32 °C, 80 %% RH, 10 MW, 7 °C / 0.7 / 0.7).

Usage::

    python3 calibrate_simulator_to_hvac.py
    python3 calibrate_simulator_to_hvac.py --coarse 24 12 --refine
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pip install pandas", file=sys.stderr)
    raise

from dc_simulator import DataCenterSimulator

DATA_REAL = Path(__file__).resolve().parent / "data" / "real"
HVAC_PATH = DATA_REAL / "HVAC Energy Data.csv"


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _simulate_chiller_series(
    sim: DataCenterSimulator,
    t_c: np.ndarray,
    it_mw: np.ndarray,
    rh_pct: float,
) -> np.ndarray:
    out = np.empty(len(t_c), dtype=np.float64)
    for i in range(len(t_c)):
        r = sim.simulate_hour(
            float(t_c[i]),
            rh_pct,
            float(it_mw[i]),
            7.0,
            0.7,
            0.7,
        )
        out[i] = r.chiller_compressor_power_kw + r.chiller_auxiliary_power_kw
    return out


def load_hvac_arrays(path: Path, max_rows: int | None) -> tuple[np.ndarray, ...]:
    hv = pd.read_csv(path)
    if max_rows is not None and max_rows > 0:
        hv = hv.iloc[:max_rows].copy()
    t_c = ((hv["Outside Temperature (F)"].astype(float) - 32) * 5 / 9).values
    rt = hv["Building Load (RT)"].astype(float).values
    y = hv["Chiller Energy Consumption (kWh)"].astype(float).values
    mean_rt = float(np.mean(rt))
    it_mw = rt / mean_rt * 10.0
    tgt_t = _corr(t_c, y)
    tgt_l = _corr(rt, y)
    return t_c, it_mw, y, rt, tgt_t, tgt_l


def loss_fn(
    k_out: float,
    t_design: float,
    *,
    t_c: np.ndarray,
    it_mw: np.ndarray,
    tgt_t: float,
    tgt_l: float,
    rh_fit: float,
    w_t: float,
    w_l: float,
    pue_lo: float,
    pue_hi: float,
    pue_penalty: float,
) -> tuple[float, float, float, float]:
    sim = DataCenterSimulator(
        outdoor_cop_penalty_per_k=k_out,
        design_outdoor_temp_c=t_design,
        thermal_storage=None,
    )
    sim_kw = _simulate_chiller_series(sim, t_c, it_mw, rh_fit)
    ct = _corr(t_c, sim_kw)
    cl = _corr(it_mw, sim_kw)

    err = w_t * (ct - tgt_t) ** 2 + w_l * (cl - tgt_l) ** 2

    r0 = sim.simulate_hour(32.0, 80.0, 10.0, 7.0, 0.7, 0.7)
    pue = float(r0.pue)
    if pue < pue_lo:
        err += pue_penalty * (pue_lo - pue) ** 2
    elif pue > pue_hi:
        err += pue_penalty * (pue - pue_hi) ** 2

    return err, ct, cl, pue


def grid_search(
    t_c: np.ndarray,
    it_mw: np.ndarray,
    y_real: np.ndarray,
    rt: np.ndarray,
    tgt_t: float,
    tgt_l: float,
    *,
    n_k: int,
    n_td: int,
    k_lo: float,
    k_hi: float,
    td_lo: float,
    td_hi: float,
    rh_fit: float,
    w_t: float,
    w_l: float,
    pue_lo: float,
    pue_hi: float,
    pue_penalty: float,
) -> tuple[float, float, float, float, float, float]:
    best = (math.inf, 0.02, 25.0, float("nan"), float("nan"), float("nan"))
    ks = np.linspace(k_lo, k_hi, n_k)
    tds = np.linspace(td_lo, td_hi, n_td)
    for k_out in ks:
        for t_design in tds:
            err, ct, cl, pue = loss_fn(
                float(k_out),
                float(t_design),
                t_c=t_c,
                it_mw=it_mw,
                tgt_t=tgt_t,
                tgt_l=tgt_l,
                rh_fit=rh_fit,
                w_t=w_t,
                w_l=w_l,
                pue_lo=pue_lo,
                pue_hi=pue_hi,
                pue_penalty=pue_penalty,
            )
            if err < best[0]:
                best = (err, float(k_out), float(t_design), ct, cl, pue)
    return best


def refine_scipy(
    t_c: np.ndarray,
    it_mw: np.ndarray,
    y_real: np.ndarray,
    rt: np.ndarray,
    tgt_t: float,
    tgt_l: float,
    k0: float,
    td0: float,
    **kw,
) -> tuple[float, float, float, float, float, float]:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return grid_search(
            t_c,
            it_mw,
            y_real,
            rt,
            tgt_t,
            tgt_l,
            n_k=int(kw.get("n_k", 18)),
            n_td=int(kw.get("n_td", 11)),
            k_lo=float(kw["k_lo"]),
            k_hi=float(kw["k_hi"]),
            td_lo=float(kw["td_lo"]),
            td_hi=float(kw["td_hi"]),
            rh_fit=float(kw["rh_fit"]),
            w_t=float(kw["w_t"]),
            w_l=float(kw["w_l"]),
            pue_lo=float(kw["pue_lo"]),
            pue_hi=float(kw["pue_hi"]),
            pue_penalty=float(kw["pue_penalty"]),
        )

    def objective(v: np.ndarray) -> float:
        e, _, _, _ = loss_fn(
            float(v[0]),
            float(v[1]),
            t_c=t_c,
            it_mw=it_mw,
            tgt_t=tgt_t,
            tgt_l=tgt_l,
            rh_fit=kw["rh_fit"],
            w_t=kw["w_t"],
            w_l=kw["w_l"],
            pue_lo=kw["pue_lo"],
            pue_hi=kw["pue_hi"],
            pue_penalty=kw["pue_penalty"],
        )
        return e

    res = minimize(
        objective,
        x0=np.array([k0, td0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(kw["k_lo"], kw["k_hi"]), (kw["td_lo"], kw["td_hi"])],
        options={"maxiter": 120, "ftol": 1e-11},
    )
    err = float(res.fun)
    k_out, t_design = float(res.x[0]), float(res.x[1])
    _, ct, cl, pue = loss_fn(
        k_out,
        t_design,
        t_c=t_c,
        it_mw=it_mw,
        tgt_t=tgt_t,
        tgt_l=tgt_l,
        rh_fit=kw["rh_fit"],
        w_t=kw["w_t"],
        w_l=kw["w_l"],
        pue_lo=kw["pue_lo"],
        pue_hi=kw["pue_hi"],
        pue_penalty=kw["pue_penalty"],
    )
    return err, k_out, t_design, ct, cl, pue


def scipy_only_fit(
    t_c: np.ndarray,
    it_mw: np.ndarray,
    y_real: np.ndarray,
    rt: np.ndarray,
    tgt_t: float,
    tgt_l: float,
    k0: float,
    td0: float,
    **kw,
) -> tuple[float, float, float, float, float, float]:
    """Optimize with SciPy only (no coarse grid). Falls back to ImportError message."""
    return refine_scipy(t_c, it_mw, y_real, rt, tgt_t, tgt_l, k0, td0, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use first N rows only (0 = all). Faster iteration.",
    )
    ap.add_argument("--coarse", nargs=2, type=int, default=[28, 15], metavar=("N_K", "N_TD"))
    ap.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=[0.003, 0.045, 21.0, 29.0],
        metavar=("K_LO", "K_HI", "TD_LO", "TD_HI"),
    )
    ap.add_argument(
        "--rh-fit",
        type=float,
        default=72.0,
        help="RH %% used when replaying HVAC rows (file has no RH).",
    )
    ap.add_argument("--w-t", type=float, default=1.0, help="Weight on (corr T)² error.")
    ap.add_argument("--w-l", type=float, default=0.35, help="Weight on (corr load)² error.")
    ap.add_argument("--pue-lo", type=float, default=1.58)
    ap.add_argument("--pue-hi", type=float, default=1.78)
    ap.add_argument("--pue-penalty", type=float, default=25.0)
    ap.add_argument(
        "--refine",
        action="store_true",
        help="After grid search, refine with SciPy L-BFGS-B if available.",
    )
    ap.add_argument(
        "--scipy-only",
        action="store_true",
        help="Skip grid; optimize from --init-k / --init-td with SciPy only (fast on large CSV).",
    )
    ap.add_argument("--init-k", type=float, default=0.022)
    ap.add_argument("--init-td", type=float, default=25.0)
    ap.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=None,
        help="Write best parameters to JSON.",
    )
    args = ap.parse_args()

    if not HVAC_PATH.is_file():
        print(f"Missing {HVAC_PATH}", file=sys.stderr)
        return 1

    max_rows = args.max_rows if args.max_rows > 0 else None
    t_c, it_mw, y_real, rt, tgt_t, tgt_l = load_hvac_arrays(HVAC_PATH, max_rows)

    print(f"HVAC rows used: {len(t_c):,}")
    print(f"Target corr(T, chiller_kWh)      = {tgt_t:.6f}")
    print(f"Target corr(load_RT, chiller_kWh)= {tgt_l:.6f}")

    k_lo, k_hi, td_lo, td_hi = args.bounds
    n_k, n_td = args.coarse

    kw = dict(
        rh_fit=args.rh_fit,
        w_t=args.w_t,
        w_l=args.w_l,
        pue_lo=args.pue_lo,
        pue_hi=args.pue_hi,
        pue_penalty=args.pue_penalty,
        k_lo=k_lo,
        k_hi=k_hi,
        td_lo=td_lo,
        td_hi=td_hi,
        n_k=max(5, n_k),
        n_td=max(5, n_td),
    )

    if args.scipy_only:
        err, k_best, td_best, ct, cl, pue = scipy_only_fit(
            t_c,
            it_mw,
            y_real,
            rt,
            tgt_t,
            tgt_l,
            args.init_k,
            args.init_td,
            **kw,
        )
    else:
        err, k_best, td_best, ct, cl, pue = grid_search(
            t_c,
            it_mw,
            y_real,
            rt,
            tgt_t,
            tgt_l,
            n_k=n_k,
            n_td=n_td,
            k_lo=k_lo,
            k_hi=k_hi,
            td_lo=td_lo,
            td_hi=td_hi,
            rh_fit=args.rh_fit,
            w_t=args.w_t,
            w_l=args.w_l,
            pue_lo=args.pue_lo,
            pue_hi=args.pue_hi,
            pue_penalty=args.pue_penalty,
        )

        if args.refine:
            err, k_best, td_best, ct, cl, pue = refine_scipy(
                t_c,
                it_mw,
                y_real,
                rt,
                tgt_t,
                tgt_l,
                k_best,
                td_best,
                **kw,
            )

    print("\n--- Best fit ---")
    print(f"  weighted_loss        = {err:.6f}")
    print(f"  outdoor_cop_penalty_per_k = {k_best:.6f}")
    print(f"  design_outdoor_temp_c     = {td_best:.4f}")
    print(f"  sim corr(T, chiller_kw)    = {ct:.6f}  (target {tgt_t:.6f})")
    print(f"  sim corr(load, chiller_kw)= {cl:.6f}  (target {tgt_l:.6f})")
    print(f"  tropical reference PUE      = {pue:.4f}  (band [{args.pue_lo}, {args.pue_hi}])")
    print("\nApply to dc_simulator.py defaults:")
    print(f"        outdoor_cop_penalty_per_k: float = {k_best:.6f},")
    print(f"        design_outdoor_temp_c: float = {td_best:.4f},")

    if args.output_json:
        payload = {
            "outdoor_cop_penalty_per_k": k_best,
            "design_outdoor_temp_c": td_best,
            "weighted_loss": err,
            "sim_corr_T": ct,
            "sim_corr_load": cl,
            "target_corr_T": tgt_t,
            "target_corr_load": tgt_l,
            "tropical_pue_reference": pue,
            "hvac_rows": len(t_c),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
