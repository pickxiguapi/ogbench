#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench-release-audit-20260814"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
CHECKPOINT_DIR="/root/data/yyf/lewm-runs/OGBench/lewm-tworoom-visual-hiql-stable-bs256-s100k/sd000_20260811_175212"
CHECKPOINT_STEP=100000
GPU_ID=1
SEED=42
NUM_EVAL=50
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
EXP_NAME="HIQL_ogbench_builtin_tworoom_s100k_seed42_goaloffset25_releaseaudit"
OUTPUT_DIR="/root/data/yyf/ogbench-release-audit-results/20260814/${EXP_NAME}"
PYTHON_BIN="/root/data/yyf/ogbench/.venv/bin/python"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || { echo "ERROR: launch through recorded_run.sh --eval-only" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
[[ -e "${DATA_ROOT}/tworoom.h5" ]] || { echo "ERROR: dataset not found: tworoom.h5" >&2; exit 1; }
[[ -e "${DATA_ROOT}/tworoom.lance" ]] || { echo "ERROR: dataset not found: tworoom.lance" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${PYTHON_BIN}" eval_ogbench_agent_lewm_envs.py \
  --task tworoom \
  --method hiql \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --num-eval "${NUM_EVAL}" \
  --seed "${SEED}" \
  --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
  --eval-budget "${EVAL_BUDGET}" \
  --output "${OUTPUT_DIR}/results.json" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${OUTPUT_DIR}/exit_status.txt"
exit "${status}"
