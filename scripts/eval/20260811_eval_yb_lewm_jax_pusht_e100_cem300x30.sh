#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
STABLEWM_ROOT="/root/data/yyf/stable-worldmodel"
EXP_NAME="LeWMJAX_lance_pusht_expert_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30"
RUN_DIR="/root/data/yyf/lewm-runs/${EXP_NAME}"
CHECKPOINT="${RUN_DIR}/weights_epoch_100.msgpack"
OUTPUT_DIR="${RUN_DIR}/eval_cem300x30_seed42"
OUTPUT_JSON="${OUTPUT_DIR}/pusht.json"
GPU_ID=1
EGL_LIB_DIR="${OGBENCH_ROOT}/.runtime/libegl1/usr/lib/x86_64-linux-gnu"

[[ -x "${STABLEWM_ROOT}/.venv/bin/python" ]] || { echo "ERROR: StableWM Python not found" >&2; exit 1; }
[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT}" ]] || { echo "ERROR: LeWM epoch-100 checkpoint not found" >&2; exit 1; }
[[ -f "${STABLEWM_ROOT}/datasets/pusht_expert_train.h5" ]] || { echo "ERROR: PushT HDF5 not found" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${STABLEWM_ROOT}:${OGBENCH_ROOT}/impls" \
"${STABLEWM_ROOT}/.venv/bin/python" eval_lewm_jax.py \
  --task=pusht \
  --checkpoint="${CHECKPOINT}" \
  --stable-wm-root="${STABLEWM_ROOT}" \
  --ogbench-root="${OGBENCH_ROOT}" \
  --num-eval=50 \
  --seed=42 \
  --goal-offset-steps=25 \
  --eval-budget=50 \
  --cem-horizon=5 \
  --cem-receding-horizon=5 \
  --action-block=5 \
  --cem-num-samples=300 \
  --cem-steps=30 \
  --cem-topk=30 \
  --video-dir="${OUTPUT_DIR}/videos" \
  --output="${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${status}" >"${OUTPUT_DIR}/exit_status.txt"
if [[ "${status}" -eq 0 && -n "${EXPERIMENT_RUN_ID:-}" && -n "${EXPERIMENT_RECORDER_ROOT:-}" ]]; then
  python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" --run-id "${EXPERIMENT_RUN_ID}" "${OUTPUT_JSON}"
fi
exit "${status}"
