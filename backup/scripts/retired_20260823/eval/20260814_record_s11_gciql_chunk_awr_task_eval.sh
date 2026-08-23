#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/yyf/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"

case "${TASK}" in
  tworoom|reacher|pusht) ;;
  *) echo "Usage: bash $0 {tworoom|reacher|pusht}" >&2; exit 2 ;;
esac

RUN_ID="EXP-010-s11-${TASK}-20260813T031705Z"
EXP_NAME="GCIQLChunkAWR_ogbench_lewm_${TASK}_k5_bs256_s100k_seed0_alpha3_expectile09_aug05_s11_r2_separate_output"
CHECKPOINT_DIR="$(find "${RUNS_ROOT}/${TASK}/OGBench/lewm-${TASK}-gciql-chunk-awr-k5-bs256-s100k-s11" \
  -mindepth 1 -maxdepth 1 -type d -name 'sd*' | sort | tail -n 1)"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_ff/${TASK}_seed_42"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_100000.pkl" ]] || { echo "ERROR: checkpoint unavailable" >&2; exit 1; }

payload="{\"checkpoint_dir\":\"${CHECKPOINT_DIR}\",\"checkpoint_step\":100000,\"evaluation\":{\"task\":\"${TASK}\",\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":25,\"eval_budget\":50}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-010 "${EXP_NAME}" "${RUN_ID}" --eval-only \
  bash "${OGBENCH_ROOT}/scripts/eval/20260814_eval_s11_lewm_gciql_chunk_awr_task_s100k_seed42.sh" "${TASK}"

python3 "${DASHBOARD_ROOT}/scripts/aggregate_evals.py" \
  --database "${DASHBOARD_ROOT}/data/experiments.json" \
  --events "${DASHBOARD_ROOT}/data/run_events.csv" \
  --catalog "${DASHBOARD_ROOT}/data/experiment_catalog.json" \
  --run-id "${RUN_ID}" \
  "${OUTPUT_DIR}"

