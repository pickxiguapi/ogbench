#!/usr/bin/env bash
set -euo pipefail

DASHBOARD_ROOT="/home/dzb/experiment-dashboard"
RUN_GROUP="EXP018_HIQL_visual_scene_official_s500k"
GROUP_DIR="/data/dzb/ogbench-native-runs/OGBench/${RUN_GROUP}"
RUN_DIR="$(find "${GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'sd000_*' | sort | tail -n 1)"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || { echo "ERROR: missing recorder run ID" >&2; exit 1; }
[[ -s "${RUN_DIR}/params_500000.pkl" ]] || { echo "ERROR: final checkpoint unavailable" >&2; exit 1; }
[[ -s "${RUN_DIR}/eval.csv" ]] || { echo "ERROR: official eval.csv unavailable" >&2; exit 1; }
python3 "${DASHBOARD_ROOT}/scripts/aggregate_evals.py" \
  --database "${DASHBOARD_ROOT}/data/experiments.json" --events "${DASHBOARD_ROOT}/data/run_events.csv" \
  --catalog "${DASHBOARD_ROOT}/data/experiment_catalog.json" --run-id "${EXPERIMENT_RUN_ID}" "${RUN_DIR}/eval.csv"
