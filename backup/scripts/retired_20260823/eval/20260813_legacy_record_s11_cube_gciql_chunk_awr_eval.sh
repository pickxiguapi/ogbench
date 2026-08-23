#!/usr/bin/env bash
set -euo pipefail

DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/yyf/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
RUN_ID="EXP-010-s11-cube-20260811T171935Z"
EXP_NAME="GCIQLChunkAWR_ogbench_lewm_cube_k5_bs256_s100k_seed0_alpha3_expectile09_aug05_s11_r1_wandboffline"
OUTPUT_DIR="/data/yyf/H-LeWM/ogbench-runs/OGBench/lewm-gciql-chunk-awr-k5-bs256-s100k-s11/sd000_20260812_011938/eval_ff/cube_seed_42"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }

payload="{\"checkpoint_dir\":\"/data/yyf/H-LeWM/ogbench-runs/OGBench/lewm-gciql-chunk-awr-k5-bs256-s100k-s11/sd000_20260812_011938\",\"checkpoint_step\":100000,\"evaluation\":{\"task\":\"cube\",\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":25,\"eval_budget\":50}}"

CUDA_VISIBLE_DEVICES=0 \
EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-010 "${EXP_NAME}" "${RUN_ID}" --eval-only \
  bash "${OGBENCH_ROOT}/scripts/eval/20260813_legacy_eval_s11_lewm_cube_gciql_chunk_awr_s100k_seed42.sh"

python3 "${DASHBOARD_ROOT}/scripts/aggregate_evals.py" \
  --database "${DASHBOARD_ROOT}/data/experiments.json" \
  --events "${DASHBOARD_ROOT}/data/run_events.csv" \
  --catalog "${DASHBOARD_ROOT}/data/experiment_catalog.json" \
  --run-id "${RUN_ID}" \
  "${OUTPUT_DIR}"
