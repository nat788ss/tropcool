#!/usr/bin/env bash
# Assemble data/rl/per_city_v2/ from pilot A+B and legacy per_city checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/rl/per_city_v2
mkdir -p "$OUT"

link_or_copy() {
  local src="$1" dest="$2"
  if [[ -f "$src" ]]; then
    rm -f "$dest"
    ln -sf "$(cd "$(dirname "$src")" && pwd)/$(basename "$src")" "$dest" 2>/dev/null || cp -f "$src" "$dest"
  else
    echo "MISSING: $src" >&2
    return 1
  fi
}

for city in cyberjaya jakarta ho_chi_minh_city singapore; do
  base="data/rl/pilot/${city}_ab"
  if [[ ! -f "${base}.zip" && "${city}" == "ho_chi_minh_city" ]]; then
    echo "Pilot ${city}_ab not ready — using data/rl/per_city/${city}" >&2
    base="data/rl/per_city/${city}"
  fi
  link_or_copy "${base}.zip" "${OUT}/${city}.zip"
  link_or_copy "${base}.vecnorm.pkl" "${OUT}/${city}.vecnorm.pkl"
done

for city in bangkok johor_bahru; do
  link_or_copy "data/rl/per_city/${city}.zip" "${OUT}/${city}.zip"
  link_or_copy "data/rl/per_city/${city}.vecnorm.pkl" "${OUT}/${city}.vecnorm.pkl"
done

echo "Assembled $OUT:"
ls -la "$OUT"
