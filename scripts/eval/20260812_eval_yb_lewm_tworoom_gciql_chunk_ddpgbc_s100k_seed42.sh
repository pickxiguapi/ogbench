#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
CHECKPOINT_DIR="/root/data/yyf/lewm-runs/OGBench/lewm-tworoom-gciql-chunk-ddpgbc-k5-bs256-s100k/sd000_20260811_181019"
CHECKPOINT_STEP=100000
GPU_ID=0
SEED=42
NUM_EVAL=50
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_ff/seed_${SEED}"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
for dataset in tworoom.h5 tworoom.lance; do
  [[ -e "${DATA_ROOT}/${dataset}" ]] || { echo "ERROR: TwoRoom dataset not found: ${dataset}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_ROOT}/.venv/bin/python" eval_ogbench_agent_lewm_envs.py \
  --task tworoom \
  --method gciql_chunk \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-root="${DATA_ROOT}" \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --num-eval "${NUM_EVAL}" \
  --seed "${SEED}" \
  --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
  --eval-budget "${EVAL_BUDGET}" \
  --video-dir "${OUTPUT_DIR}/videos" \
  --output "${OUTPUT_DIR}/tworoom_gciql_chunk_step${CHECKPOINT_STEP}.json" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${OUTPUT_DIR}/exit_status.txt"
exit "${status}"
