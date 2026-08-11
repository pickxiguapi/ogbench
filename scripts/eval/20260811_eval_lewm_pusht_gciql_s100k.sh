#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OGBENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STABLEWM_ROOT="/root/data/yyf/stable-worldmodel"
CHECKPOINT_DIR="/root/data/yyf/ogbench/checkpoints/lewm_gciql_s100k/pusht"
CHECKPOINT_STEP=100000
GPU_ID=6
SEED=42
NUM_EVAL=50
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_ff/seed_${SEED}"

[[ -x "${STABLEWM_ROOT}/.venv/bin/python" ]] || { echo "ERROR: StableWM Python not found" >&2; exit 1; }
[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: PushT checkpoint not found" >&2; exit 1; }
for dataset in pusht_expert_train.h5 pusht_expert_train.lance; do
  [[ -e "${STABLEWM_ROOT}/datasets/${dataset}" ]] || { echo "ERROR: PushT dataset not found: ${dataset}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  PYTHONPATH="${STABLEWM_ROOT}:${OGBENCH_ROOT}/impls" \
  "${STABLEWM_ROOT}/.venv/bin/python" eval_lewm.py \
    --task pusht \
    --method gciql \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --stable-wm-root "${STABLEWM_ROOT}" \
    --ogbench-root "${OGBENCH_ROOT}" \
    --num-eval "${NUM_EVAL}" \
    --seed "${SEED}" \
    --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
    --eval-budget "${EVAL_BUDGET}" \
    --video-dir "${OUTPUT_DIR}/videos" \
    --output "${OUTPUT_DIR}/pusht_gciql_step${CHECKPOINT_STEP}.json" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${OUTPUT_DIR}/exit_status.txt"
exit "${status}"
