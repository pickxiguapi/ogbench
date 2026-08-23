#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OGBENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
CHECKPOINT_DIR="/root/data/yyf/ogbench/checkpoints/lewm_gciql_s100k/reacher"
CHECKPOINT_STEP=100000
GPU_ID=7
SEED=42
NUM_EVAL=50
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_ff/seed_${SEED}"
EGL_LIB_DIR="${OGBENCH_ROOT}/.runtime/libegl1/usr/lib/x86_64-linux-gnu"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || {
  exit 1
}
[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || {
  echo "ERROR: OGBench Python not found: ${OGBENCH_ROOT}/.venv/bin/python" >&2
  exit 1
}
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || {
  echo "ERROR: checkpoint not found: ${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" >&2
  exit 1
}
[[ -s "${EGL_LIB_DIR}/libEGL.so.1" ]] || {
  echo "ERROR: user-level libEGL not found: ${EGL_LIB_DIR}/libEGL.so.1" >&2
  exit 1
}
for dataset in reacher.h5 reacher.lance; do
  [[ -e "${DATA_ROOT}/${dataset}" ]] || {
    echo "ERROR: dataset not found: ${DATA_ROOT}/${dataset}" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
  "${OGBENCH_ROOT}/.venv/bin/python" eval_ogbench_agent_lewm_envs.py \
    --task reacher \
    --method gciql \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-root="${DATA_ROOT}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --num-eval "${NUM_EVAL}" \
    --seed "${SEED}" \
    --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
    --eval-budget "${EVAL_BUDGET}" \
    --video-dir "${OUTPUT_DIR}/videos" \
    --output "${OUTPUT_DIR}/reacher_gciql_step${CHECKPOINT_STEP}.json" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${OUTPUT_DIR}/exit_status.txt"
exit "${status}"
