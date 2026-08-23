#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
OGBENCH_PYTHON="${OGBENCH_PYTHON:-/data/yyf/H-LeWM/envs/ogbench/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/yyf/H-LeWM/ogbench-runs/OGBench/lewm-gciql-chunk-awr-k5-bs256-s100k-s11/sd000_20260812_011938}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/eval_ff/cube_seed_${SEED}}"
OUTPUT_JSON="${OUTPUT_DIR}/cube_gciql_chunk_awr_step${CHECKPOINT_STEP}.json"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh --eval-only" >&2
  exit 2
}
[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
[[ -e "${DATA_ROOT}/cube_single_expert.h5" ]] || { echo "ERROR: Cube HDF5 not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/cube_single_expert.lance/.conversion_complete" ]] || { echo "ERROR: Cube Lance dataset incomplete" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_PYTHON}" eval_ogbench_agent_lewm_envs.py \
  --task cube \
  --method gciql_chunk \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-root="${DATA_ROOT}" \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --num-eval "${NUM_EVAL}" \
  --seed "${SEED}" \
  --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
  --eval-budget "${EVAL_BUDGET}" \
  --video-dir "${OUTPUT_DIR}/videos" \
  --output "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
