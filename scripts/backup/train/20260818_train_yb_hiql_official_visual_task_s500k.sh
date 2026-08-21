#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train/20260818_train_yb_hiql_official_visual_task_s500k.sh {cube-single|cube-double|cube-triple|cube-quadruple|scene}" >&2
  exit 2
fi

TASK="$1"
case "${TASK}" in
  cube-single) ENV_NAME="visual-cube-single-play-v0" ;;
  cube-double) ENV_NAME="visual-cube-double-play-v0" ;;
  cube-triple) ENV_NAME="visual-cube-triple-play-v0" ;;
  cube-quadruple) ENV_NAME="visual-cube-quadruple-play-v0" ;;
  scene) ENV_NAME="visual-scene-play-v0" ;;
  *) echo "ERROR: unknown task ${TASK}" >&2; exit 2 ;;
esac

OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench-new}"
RUNS_ROOT="${RUNS_ROOT:-/root/data/yyf/ogbench-hiql-official-runs}"
EGL_LIB_DIR="${EGL_LIB_DIR:-/root/data/yyf/egl-runtime/root/usr/lib/x86_64-linux-gnu}"
GPU_ID="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by the recorded launcher}"
EXP_NAME="${EXPERIMENT_EXP_NAME:?EXPERIMENT_EXP_NAME must be set by recorded_run.sh}"
RUN_ID="${EXPERIMENT_RUN_ID:?EXPERIMENT_RUN_ID must be set by recorded_run.sh}"
RUN_GROUP="EXP022_HIQL_${TASK}"
LOG_PATH="${RUNS_ROOT}/logs/${RUN_ID}.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/agents/hiql.py" ]] || { echo "ERROR: original HIQL agent not found" >&2; exit 1; }
[[ -f "${EGL_LIB_DIR}/libEGL.so.1" ]] || { echo "ERROR: user EGL runtime not found" >&2; exit 1; }
[[ ! -e "${RUNS_ROOT}/OGBench/${RUN_GROUP}" ]] || { echo "ERROR: run group already exists: ${RUN_GROUP}" >&2; exit 1; }

mkdir -p "${RUNS_ROOT}/logs" "${RUNS_ROOT}/tmp" "${RUNS_ROOT}/wandb"
cd "${OGBENCH_ROOT}/impls"

MUJOCO_GL=egl \
LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
TMPDIR="${RUNS_ROOT}/tmp" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --agent=agents/hiql.py \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.lr=3e-4 \
  --agent.discount=0.99 \
  --agent.expectile=0.7 \
  --agent.tau=0.005 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --agent.low_actor_rep_grad=True \
  --agent.p_aug=0.5 \
  --agent.subgoal_steps=10 \
  --train_steps=500000 \
  --save_dir="${RUNS_ROOT}" \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=100000 \
  --run_group="${RUN_GROUP}" \
  --wandb_mode=offline \
  --seed=0 \
  --eval_episodes=50 \
  --eval_on_cpu=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
