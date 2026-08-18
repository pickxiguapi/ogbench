#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/eval/20260818_eval_yb_hiql_chunk_two_v_visual_s500k_seed42.sh {cube-single|cube-double|cube-triple|cube-quadruple|scene}" >&2
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
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/root/data/yyf/experiment-dashboard}"
TRAIN_RUNS_ROOT="${TRAIN_RUNS_ROOT:-/root/data/yyf/ogbench-hiql-chunk-two-v-runs}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-/root/data/yyf/ogbench-hiql-chunk-two-v-evals}"
EGL_LIB_DIR="${EGL_LIB_DIR:-/root/data/yyf/egl-runtime/root/usr/lib/x86_64-linux-gnu}"
GPU_ID="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by the recorded launcher}"
RUN_ID="${EXPERIMENT_RUN_ID:?EXPERIMENT_RUN_ID must be set by recorded_run.sh}"
TRAIN_GROUP="EXP021_HIQLChunk2V_${TASK}"
EVAL_GROUP="EXP021_EVAL_HIQLChunk2V_${TASK}_s500k_seed42"
TRAIN_GROUP_DIR="${TRAIN_RUNS_ROOT}/OGBench/${TRAIN_GROUP}"
RESTORE_DIR="$(find "${TRAIN_GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'sd000_*' | sort | tail -n 1)"
LOG_PATH="${EVAL_RUNS_ROOT}/logs/${RUN_ID}.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -n "${RESTORE_DIR}" ]] || { echo "ERROR: training run not found under ${TRAIN_GROUP_DIR}" >&2; exit 1; }
[[ -s "${RESTORE_DIR}/params_500000.pkl" ]] || { echo "ERROR: final checkpoint unavailable: ${RESTORE_DIR}/params_500000.pkl" >&2; exit 1; }
[[ -f "${EGL_LIB_DIR}/libEGL.so.1" ]] || { echo "ERROR: user EGL runtime not found" >&2; exit 1; }
[[ ! -e "${EVAL_RUNS_ROOT}/OGBench/${EVAL_GROUP}" ]] || { echo "ERROR: evaluation group already exists: ${EVAL_GROUP}" >&2; exit 1; }

mkdir -p "${EVAL_RUNS_ROOT}/logs" "${EVAL_RUNS_ROOT}/tmp" "${EVAL_RUNS_ROOT}/wandb"
cd "${OGBENCH_ROOT}/impls"

MUJOCO_GL=egl \
LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
TMPDIR="${EVAL_RUNS_ROOT}/tmp" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${EVAL_RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --agent=agents/hiql_chunk.py \
  --agent.batch_size=256 \
  --agent.chunk_size=5 \
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
  --restore_path="${RESTORE_DIR}" \
  --restore_epoch=500000 \
  --eval_only=True \
  --save_dir="${EVAL_RUNS_ROOT}" \
  --run_group="${EVAL_GROUP}" \
  --wandb_mode=offline \
  --seed=42 \
  --eval_episodes=50 \
  --eval_on_cpu=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"

EVAL_DIR="$(find "${EVAL_RUNS_ROOT}/OGBench/${EVAL_GROUP}" -mindepth 1 -maxdepth 1 -type d -name 'sd042_*' | sort | tail -n 1)"
[[ -s "${EVAL_DIR}/eval.csv" ]] || { echo "ERROR: evaluation CSV unavailable" >&2; exit 1; }
python3 "${DASHBOARD_ROOT}/scripts/aggregate_evals.py" \
  --database "${DASHBOARD_ROOT}/data/experiments.json" \
  --events "${DASHBOARD_ROOT}/data/run_events.csv" \
  --catalog "${DASHBOARD_ROOT}/data/experiment_catalog.json" \
  --run-id "${RUN_ID}" "${EVAL_DIR}/eval.csv"

