#!/usr/bin/env bash
# Generate climate projections, extended results, and 24h traces for the dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Climate weather CSVs ==="
python3 scripts/generate_climate_weather.py

echo "=== Assemble per_city_v2 (if needed) ==="
if [[ ! -d data/rl/per_city_v2 ]] || [[ $(ls -1 data/rl/per_city_v2/*.zip 2>/dev/null | wc -l) -lt 6 ]]; then
  ./scripts/assemble_per_city_v2.sh
fi

echo "=== Export all dashboard JSON (climate grid + traces + metrics) ==="
python3 scripts/export_dashboard_data.py "$@"

echo "Done. Launch: streamlit run dashboard/app.py"
