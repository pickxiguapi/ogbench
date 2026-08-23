#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench-lewm-envs-test"
OGBENCH_PYTHON="/home/dzb/ogbench/.venv/bin/python"
DATA_ROOT="/data/dzb/stablewm-data/datasets"
RUN_DIR="/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
CHECKPOINT="${RUN_DIR}/weights_epoch_10.msgpack"
OUTPUT_DIR="/data/dzb/stablewm-data/ogbench-lewm-envs-smoke/20260814_jax_cem_cube"
OUTPUT_JSON="${OUTPUT_DIR}/cube.json"
GPU_ID=2

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh" >&2
  exit 2
}
[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT}" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${DATA_ROOT}/cube_single_expert.h5" ]] || { echo "ERROR: Cube HDF5 not found" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_PYTHON}" eval_lewm_jax_cem.py \
  --task=cube \
  --checkpoint="${CHECKPOINT}" \
  --data-root="${DATA_ROOT}" \
  --num-eval=1 \
  --seed=42 \
  --goal-offset-steps=25 \
  --eval-budget=1 \
  --cem-horizon=5 \
  --cem-receding-horizon=5 \
  --action-block=5 \
  --cem-num-samples=4 \
  --cem-steps=1 \
  --cem-topk=2 \
  --cem-var-scale=1.0 \
  --output="${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
