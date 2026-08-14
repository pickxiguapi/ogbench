#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench"
DATA_ROOT="/data/dzb/stablewm-data/datasets"
EXP_NAME="LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
RUN_DIR="/data/dzb/stablewm-data/lewm-jax-runs/${EXP_NAME}"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -d "${DATA_ROOT}/pusht_expert_train.lance" ]] || { echo "ERROR: PushT Lance dataset not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/pusht_expert_train.h5" ]] || { echo "ERROR: PushT source HDF5 not found" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: run directory already exists: ${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
"${OGBENCH_ROOT}/.venv/bin/python" train_lewm_jax.py \
  --dataset_path="${DATA_ROOT}/pusht_expert_train.lance" \
  --save_dir="${RUN_DIR}" --exp_name="${EXP_NAME}" \
  --epochs=10 --batch_size=128 --seed=3072 \
  --frameskip=5 --learning_rate=5e-5 --weight_decay=1e-3 \
  --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 \
  --decode_workers=6 2>&1 | tee "${RUN_DIR}/train.log"
