#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
RUNS_ROOT="/root/data/yyf/lewm-runs"
GPU_ID=5
EXP_NAME="LeWMJAX_lance_tworoom_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}"
LOG_PATH="${RUN_DIR}/train.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -d "${DATA_ROOT}/tworoom.lance" ]] || { echo "ERROR: TwoRoom Lance dataset not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/tworoom.h5" ]] || { echo "ERROR: TwoRoom source HDF5 is required for exact action statistics" >&2; exit 1; }
mkdir -p "${RUN_DIR}"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
"${OGBENCH_ROOT}/.venv/bin/python" train_lewm_jax.py \
  --dataset_path="${DATA_ROOT}/tworoom.lance" \
  --save_dir="${RUN_DIR}" \
  --exp_name="${EXP_NAME}" \
  --seed=3072 \
  --epochs=100 \
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
