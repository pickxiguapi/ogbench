#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 {tworoom|reacher|pusht|cube}" >&2
  exit 2
fi

TASK="$1"
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
  tworoom) DATASET_STEM=tworoom ;;
  reacher) DATASET_STEM=reacher ;;
  pusht) DATASET_STEM=pusht_expert_train ;;
  cube) DATASET_STEM=cube_single_expert ;;
  *) echo "ERROR: unknown task ${TASK}" >&2; exit 2 ;;
esac

RUN_GROUP="hiql-chunk-share-v-lewm-${TASK}-k5-sg10-bs256-s100k"
RUN_GROUP_DIR="${RUNS_ROOT}/${TASK}/OGBench/${RUN_GROUP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(find "${RUN_GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'sd000_*' | sort | tail -n 1)}"
EXP_NAME="${EXP_NAME:-HIQLChunkShareV_ogbench_lewm_${TASK}_k5_sg10_s100k_direct_g25_b50_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/eval_direct/${EXP_NAME}}"
OUTPUT_JSON="${OUTPUT_DIR}/${TASK}_hiql_chunk_share_v_step${CHECKPOINT_STEP}.json"

[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found: ${OGBENCH_PYTHON}" >&2; exit 1; }
[[ -n "${CHECKPOINT_DIR}" ]] || { echo "ERROR: run not found under ${RUN_GROUP_DIR}" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found: ${CHECKPOINT_DIR}/flags.json" >&2; exit 1; }
[[ -e "${DATA_ROOT}/${DATASET_STEM}.h5" ]] || { echo "ERROR: HDF5 dataset not found: ${DATA_ROOT}/${DATASET_STEM}.h5" >&2; exit 1; }
[[ -f "${DATA_ROOT}/${DATASET_STEM}.lance/.conversion_complete" ]] || { echo "ERROR: Lance dataset incomplete: ${DATA_ROOT}/${DATASET_STEM}.lance" >&2; exit 1; }
[[ ! -e "${OUTPUT_JSON}" ]] || { echo "ERROR: result already exists: ${OUTPUT_JSON}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_PYTHON}" eval_ogbench_agent_lewm_envs.py \
  --task "${TASK}" \
  --method hiql_chunk_share_v \
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
