#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench-new"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
RUNS_ROOT="/root/data/yyf/lewm-jax-runs"
GPU_ID=2
EXP_NAME="LeWMJAX_ogbench_cube_single_impala_bs128_e10_seed3072_fs5_h3_sigreg009_bf16_yb"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}"
LOG_PATH="${RUN_DIR}/train.log"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch this script through experiment-dashboard/scripts/recorded_run.sh" >&2
  exit 1
}
[[ "${EXPERIMENT_EXP_NAME:-}" == "${EXP_NAME}" ]] || {
  echo "ERROR: recorded exp_name must be ${EXP_NAME}" >&2
  exit 1
}
[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/train_lewm_jax.py" ]] || { echo "ERROR: train_lewm_jax.py not found" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cube_single_expert.lance" ]] || { echo "ERROR: Cube Lance dataset not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/cube_single_expert.h5" ]] || { echo "ERROR: Cube source HDF5 not found" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: run directory already exists: ${RUN_DIR}" >&2; exit 1; }

mkdir -p "${RUN_DIR}"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
"${OGBENCH_ROOT}/.venv/bin/python" train_lewm_jax.py \
  --dataset_path="${DATA_ROOT}/cube_single_expert.lance" \
  --save_dir="${RUN_DIR}" \
  --exp_name="${EXP_NAME}" \
  --seed=3072 \
  --epochs=10 \
  --batch_size=128 \
  --frameskip=5 \
  --learning_rate=5e-5 \
  --weight_decay=1e-3 \
  --sigreg_weight=0.09 \
  --sigreg_knots=17 \
  --sigreg_num_proj=1024 \
  --decode_workers=8 \
  2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${RUN_DIR}/exit_status.txt"
exit "${status}"
