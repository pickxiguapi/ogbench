#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench"
STABLEWM_ROOT="/home/dzb/stable-worldmodel"
EXP_NAME="LeWMJAX_ref66_vit_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
RUN_DIR="/data/dzb/stablewm-data/lewm-jax-runs/${EXP_NAME}"
OUTPUT_DIR="${RUN_DIR}/eval_cem300x30_seed42"

[[ -s "${RUN_DIR}/weights_epoch_10.msgpack" ]] || { echo "ERROR: epoch-10 checkpoint not found" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="${STABLEWM_ROOT}:${OGBENCH_ROOT}/impls" \
"${STABLEWM_ROOT}/.venv/bin/python" eval_lewm_jax.py \
  --task=cube --checkpoint="${RUN_DIR}/weights_epoch_10.msgpack" \
  --stable-wm-root="${STABLEWM_ROOT}" --ogbench-root="${OGBENCH_ROOT}" \
  --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
  --video-dir="${OUTPUT_DIR}/videos" --output="${OUTPUT_DIR}/cube.json" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" \
  --run-id "${EXPERIMENT_RUN_ID}" \
  --database "${EXPERIMENT_RECORDER_ROOT}/data/experiments.json" \
  --events "${EXPERIMENT_RECORDER_ROOT}/data/run_events.csv" \
  --catalog "${EXPERIMENT_RECORDER_ROOT}/data/experiment_catalog.json" \
  "${OUTPUT_DIR}/cube.json"
