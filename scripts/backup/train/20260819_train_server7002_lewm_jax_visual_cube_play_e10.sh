#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train/20260819_train_server7002_lewm_jax_visual_cube_play_e10.sh {single|double|triple}" >&2
  exit 2
fi

TASK="$1"
case "${TASK}" in
  single|double|triple) ;;
  *) echo "ERROR: unknown task: ${TASK}" >&2; exit 2 ;;
esac

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/mnt/18T/yyf/ogbench-data}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/18T/yyf/lewm-jax-runs}"
PYTHON_BIN="${PYTHON_BIN:-${OGBENCH_ROOT}/.venv/bin/python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by the launcher}"
ENV_NAME="visual-cube-${TASK}-play-v0"
EXP_NAME="LeWMJAX_ogbench_visual_cube_${TASK}_play_impalasmall_bs128_e10_seed3072_fs1_h3_bf16_7002"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
TRAIN_DATA="${DATA_ROOT}/${ENV_NAME}.npz"
VAL_DATA="${DATA_ROOT}/${ENV_NAME}-val.npz"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/train_lewm_jax.py" ]] || { echo "ERROR: LeWM JAX trainer not found" >&2; exit 1; }
[[ -s "${TRAIN_DATA}" ]] || { echo "ERROR: training dataset not found: ${TRAIN_DATA}" >&2; exit 1; }
[[ -s "${VAL_DATA}" ]] || { echo "ERROR: validation dataset not found: ${VAL_DATA}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: run directory already exists: ${RUN_DIR}" >&2; exit 1; }

mkdir -p "${RUN_DIR}" "${RUNS_ROOT}/tmp"
cd "${OGBENCH_ROOT}/impls"

echo "Starting ${EXP_NAME} on physical GPU ${GPU_ID}"
echo "Dataset environment: ${ENV_NAME}; training budget: 10 epochs; one atomic action per model step"

set +e
TMPDIR="${RUNS_ROOT}/tmp" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${PYTHON_BIN}" train_lewm_jax.py \
  --dataset_path="${TRAIN_DATA}" \
  --validation_dataset_path="${VAL_DATA}" \
  --dataset_format=npz \
  --save_dir="${RUN_DIR}" \
  --exp_name="${EXP_NAME}" \
  --seed=3072 \
  --epochs=10 \
  --batch_size=128 \
  --frameskip=1 \
  --image_size=64 \
  --learning_rate=5e-5 \
  --weight_decay=1e-3 \
  --sigreg_weight=0.09 \
  --sigreg_knots=17 \
  --sigreg_num_proj=1024 \
  --decode_workers=1 \
  2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${RUN_DIR}/exit_status.txt"
exit "${status}"
