#!/usr/bin/env bash
# Train A+B recipe for cities that failed strict gate (sequential).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/rl/pilot

COMMON=(
  --timesteps 500000
  --episode-len 512
  --n-envs 4
  --vecnormalize
  --no-norm-reward
  --calendar-year-choices 2018,2019,2020
  --random-start-row-max 8760
  --eval-aligned-fraction 0.4
  --action-box baseline
  --seed 42
)

for city in cyberjaya jakarta ho_chi_minh_city; do
  log="data/rl/pilot/${city}_ab_train.log"
  echo "=== $(date -Iseconds) START ${city} ===" | tee -a "$log"
  python3 train_sac_tropcool.py --city "$city" \
    --save "data/rl/pilot/${city}_ab" \
    "${COMMON[@]}" 2>&1 | tee -a "$log"
  echo "=== $(date -Iseconds) DONE ${city} ===" | tee -a "$log"
done

echo "=== All A+B pilot trains finished $(date -Iseconds) ==="
