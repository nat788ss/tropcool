#!/usr/bin/env bash
# Copy precomputed dashboard JSON into web/public/data for Vercel static deploy.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="web/public/data"
mkdir -p "$DEST/traces"

for f in climate_results.json results.json regional_impact.json model_metrics.json; do
  if [[ ! -f "dashboard/$f" ]]; then
    echo "Missing dashboard/$f — run ./scripts/run_dashboard_assets.sh first" >&2
    exit 1
  fi
  cp "dashboard/$f" "$DEST/$f"
done

if [[ ! -d dashboard/traces ]] || [[ -z "$(ls -A dashboard/traces/*.json 2>/dev/null)" ]]; then
  echo "Missing dashboard/traces/*.json — run export script first" >&2
  exit 1
fi
cp dashboard/traces/*.json "$DEST/traces/"

echo "Copied dashboard JSON to $DEST ($(du -sh "$DEST" | cut -f1))"
