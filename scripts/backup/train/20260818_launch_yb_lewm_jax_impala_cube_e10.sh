#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench-new"
DASHBOARD_ROOT="/root/data/yyf/experiment-dashboard"
TRAIN_SCRIPT="${OGBENCH_ROOT}/scripts/train/20260818_train_yb_lewm_jax_impala_cube_e10.sh"
EXP_ID="EXP-030"
RUN_ID="EXP-030-YB-CUBE-R01"
EXP_NAME="LeWMJAX_ogbench_cube_single_impala_bs128_e10_seed3072_fs5_h3_sigreg009_bf16_yb"
GPU_ID=2
TMUX_SESSION="exp030-yb-lewm-jax-cube-e10"
EXTRA_PAYLOAD='{"model_backend":"lewm_jax","encoder":"impala_small","dataset_backend":"lance","environment":"cube_single","batch_size":128,"epochs":10,"seed":3072,"frameskip":5,"history_size":3,"precision":"bf16"}'

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not found" >&2; exit 1; }
[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: experiment recorder not found" >&2; exit 1; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: training Bash not found" >&2; exit 1; }
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${TMUX_SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${TMUX_SESSION}" -c "${OGBENCH_ROOT}" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    EXPERIMENT_EXTRA_PAYLOAD_JSON="${EXTRA_PAYLOAD}" \
    bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
      "${EXP_ID}" "${EXP_NAME}" "${RUN_ID}" \
      -- bash "${TRAIN_SCRIPT}"

echo "started ${TMUX_SESSION}: GPU ${GPU_ID}, ${EXP_NAME}, run=${RUN_ID}"
echo "attach with: tmux attach -t ${TMUX_SESSION}"
