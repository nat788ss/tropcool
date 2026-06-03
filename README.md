TropCool

Transformer world model + per-city SAC reinforcement learning for tropical data-center cooling across six Southeast Asian cities (Cyberjaya, Singapore, Jakarta, Bangkok, Johor Bahru, Ho Chi Minh City).

Live dashboard: https://web-zeta-two-66.vercel.app
Source: https://github.com/nat788ss/tropcool

What this is





A 24-hour-context transformer predicts next-hour rack temperature, cooling power, water, PUE, and cost from weather, plant config, and setpoints.



Policies train in a Gymnasium world-model environment (fast surrogate) instead of stepping full physics every RL trial.



Synthetic climate projections (IPCC-style central warming + optional stress bands) are not CMIP6—they adjust historical Open-Meteo archives for planning what-ifs.

Results (honest reporting)







Claim



Detail





Validated SAC portfolio



Five cities pass strict eval (Cyberjaya, Singapore, Bangkok, Johor Bahru, Ho Chi Minh): SAC mean cost below baseline on the frozen compare_policies.py contract (~8% mean savings on headline Y when Jakarta is excluded).





Jakarta



Still shown on the map and in tables; baseline recommended—SAC can regress vs baseline under the same contract.





World model



Planning surrogate for RL and dashboards; held-out synthetic eval PUE R² ~0.92 (re-run evaluate_world_model.py for your checkpoint).





Physics spot-check



Singapore: SAC can win on the world model but flip vs ordering on dc_simulator—see scripts/simulator_spotcheck.py and data/rl/simulator_spotcheck.log.

Full architecture and benchmark narrative: docs/World_Model_and_RL_Brief.md. Video outline: docs/VIDEO_SCRIPT_AND_WALKTHROUGH.md.

Quick start

git clone https://github.com/nat788ss/tropcool.git && cd tropcool
python3 -m venv .venv && source .venv/bin/activate
pip install gymnasium stable-baselines3 numpy pandas torch scikit-learn joblib matplotlib requests
pip install -r dashboard/requirements.txt   # Streamlit dashboard only

Artifacts not in git (obtain locally):





World model weights: world_model_best.pt or data/training/checkpoints/world_model_best.pt (~9 MB)—train/finetune with project scripts, or copy from your training run. Scalers: data/training/checkpoints/world_model_scalers.json (committed) / .joblib.



Weather CSVs: data/weather/ — run fetch_weather_historical.py (large; gitignored).



SAC checkpoints: data/rl/per_city_v2/*.zip — train with train_sac_tropcool.py and ./scripts/assemble_per_city_v2.sh (zips gitignored).



Large Kaggle CSV: data/real/final_dataset_std.csv (optional fine-tune; download separately). Smaller real CSVs may be added locally for calibration.

Reproduce key numbers

Strict policy comparison (5-city portfolio gate):

./scripts/assemble_per_city_v2.sh
python3 compare_policies.py \
  --rl-per-city-dir data/rl/per_city_v2 \
  --action-box baseline \
  --episode-len 512 \
  --mean-over-scenarios \
  --exclude-cities jakarta \
  --require-sac-beats-baseline-all

Climate headlines + dashboard JSON:

python3 scripts/generate_climate_weather.py --periods current,2030,2040,2050,2040_stress,2050_stress
python3 scripts/run_climate_extended_results.py --episode-hours 8760   # or --fast (720h)
python3 scripts/export_dashboard_data.py --skip-traces --episode-hours 8760
# Full bundle: ./scripts/run_dashboard_assets.sh

Streamlit (local): streamlit run dashboard/app.py
Static site (Vercel): ./scripts/prepare_vercel_assets.sh then cd web && npm install && npm run build

Train SAC (A+B recipe — Singapore pilot)

python3 train_sac_tropcool.py --city singapore \
  --save data/rl/pilot/singapore_ab \
  --timesteps 500000 --episode-len 512 --n-envs 4 \
  --vecnormalize --no-norm-reward \
  --calendar-year-choices 2018,2019,2020 \
  --random-start-row-max 8760 \
  --eval-aligned-fraction 0.4 \
  --action-box baseline --seed 42

Sequential retrain for failing cities: ./scripts/run_ab_train_sequential.sh

Repo structure







Path



Purpose





data/weather/



Hourly Open-Meteo archives (gitignored; fetch locally)





data/training/



Checkpoints, eval plots, processed parquets (mostly gitignored)





data/rl/



SAC .zip / VecNormalize (gitignored); training logs optional





data/real/



HVAC calibration JSON + optional Kaggle CSVs





dashboard/



Streamlit app + precomputed *.json





web/



Vite/React public dashboard (web/public/data/ JSON)





scripts/



Climate generation, dashboard export, Vercel prep, eval parsers





docs/



World model brief, video script





Root



train_sac_tropcool.py, compare_policies.py, WorldModelEnv, simulator

Citation

If you use this work, please link the repository:

TropCool — Transformer world model + RL for tropical DC cooling (SEA).
https://github.com/nat788ss/tropcool

License

See repository defaults; weather data via Open-Meteo; public DC/HVAC datasets as cited in docs/World_Model_and_RL_Brief.md.
