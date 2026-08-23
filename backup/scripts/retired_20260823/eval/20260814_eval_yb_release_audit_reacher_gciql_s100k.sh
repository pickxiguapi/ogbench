#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench-release-audit-20260814"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
CHECKPOINT_DIR="/root/data/yyf/ogbench/checkpoints/lewm_gciql_s100k/reacher"
CHECKPOINT_STEP=100000
GPU_ID=3
SEED=42
NUM_EVAL=50
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
EXP_NAME="GCIQL_ogbench_builtin_reacher_s100k_seed42_goaloffset25_releaseaudit"
OUTPUT_DIR="/root/data/yyf/ogbench-release-audit-results/20260814/${EXP_NAME}"
PYTHON_BIN="/root/data/yyf/ogbench/.venv/bin/python"
EGL_LIB_DIR="/root/data/yyf/ogbench/.runtime/libegl1/usr/lib/x86_64-linux-gnu"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || { echo "ERROR: launch through recorded_run.sh --eval-only" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
[[ -s "${EGL_LIB_DIR}/libEGL.so.1" ]] || { echo "ERROR: user-level libEGL not found" >&2; exit 1; }
[[ -e "${DATA_ROOT}/reacher.h5" ]] || { echo "ERROR: dataset not found: reacher.h5" >&2; exit 1; }
[[ -e "${DATA_ROOT}/reacher.lance" ]] || { echo "ERROR: dataset not found: reacher.lance" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${PYTHON_BIN}" eval_ogbench_agent_lewm_envs.py \
  --task reacher \
  --method gciql \
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
