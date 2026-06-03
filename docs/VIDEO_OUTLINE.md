# TropCool demo video outline (~3 min)

1. **Hook (15s)** — Tropical DC cooling is expensive; six ASEAN cities, real weather, learned surrogate.
2. **Data flywheel (30s)** — Weather CSVs → `dc_simulator` → transformer world model (`world_model.py`).
3. **RL (45s)** — SAC in `WorldModelEnv`; A+B recipe (512 h, baseline action box); show `compare_policies` table: 5/6 cities beat baseline, Jakarta exception.
4. **Singapore slice (45s)** — One week 2019: baseline vs SAC cost on surrogate; mention simulator spot-check caveat.
5. **Dashboard (20s)** — `streamlit run dashboard/app.py` city dropdown.
6. **Close (15s)** — Strict gate, next step Jakarta, climate projections.
