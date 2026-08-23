#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
OGBENCH_PYTHON="${OGBENCH_PYTHON:-/data/yyf/H-LeWM/envs/ogbench/bin/python}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
SEED="${SEED:-42}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"

case "${TASK}" in
  tworoom)
    DATASET_STEM=tworoom
    ;;
  reacher)
    DATASET_STEM=reacher
    ;;
  pusht)
    DATASET_STEM=pusht_expert_train
    ;;
  *)
    echo "Usage: bash $0 {tworoom|reacher|pusht}" >&2
    exit 2
    ;;
esac

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh --eval-only" >&2
  exit 2
}

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_DIR="$(find "${RUNS_ROOT}/${TASK}/OGBench/lewm-${TASK}-gciql-chunk-awr-k5-bs256-s100k-s11" \
    -mindepth 1 -maxdepth 1 -type d -name 'sd*' | sort | tail -n 1)"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/eval_ff/${TASK}_seed_${SEED}}"
OUTPUT_JSON="${OUTPUT_DIR}/${TASK}_gciql_chunk_awr_step${CHECKPOINT_STEP}.json"

[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
[[ -e "${DATA_ROOT}/${DATASET_STEM}.h5" ]] || { echo "ERROR: HDF5 dataset not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/${DATASET_STEM}.lance/.conversion_complete" ]] || { echo "ERROR: Lance dataset incomplete" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_PYTHON}" eval_ogbench_agent_lewm_envs.py \
  --task "${TASK}" \
  --method gciql_chunk \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --num-eval "${NUM_EVAL}" \
  --seed "${SEED}" \
  --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
  --eval-budget "${EVAL_BUDGET}" \
  --video-dir "${OUTPUT_DIR}/videos" \
  --output "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

