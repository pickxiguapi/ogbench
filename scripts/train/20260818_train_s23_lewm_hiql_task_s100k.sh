#!/usr/bin/env bash
set -euo pipefail

# Server 23 only: consolidated LeWM visual HIQL trainer.
# Launch through experiment-dashboard/scripts/recorded_run.sh.

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train/20260818_train_s23_lewm_hiql_task_s100k.sh {cube|pusht|reacher|tworoom}" >&2
  exit 2
fi

TASK="$1"
case "${TASK}" in
  cube)
    ENV_NAME="visual-lewm-cube-single-expert-v0"
    DATASET_NAME="cube_single_expert.lance"
    DEFAULT_GPU=1
    ;;
  pusht)
    ENV_NAME="visual-lewm-pusht-expert-train-v0"
    DATASET_NAME="pusht_expert_train.lance"
    DEFAULT_GPU=3
    ;;
  reacher)
    ENV_NAME="visual-lewm-reacher-v0"
    DATASET_NAME="reacher.lance"
    DEFAULT_GPU=7
    ;;
  tworoom)
    ENV_NAME="visual-lewm-tworoom-v0"
    DATASET_NAME="tworoom.lance"
    DEFAULT_GPU=5
    ;;
  *)
    echo "ERROR: unsupported task: ${TASK}" >&2
    exit 2
    ;;
esac

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/dzb/lewm-runs}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPU}}"
EXP_NAME="${EXPERIMENT_EXP_NAME:?launch through recorded_run.sh}"
RUN_ID="${EXPERIMENT_RUN_ID:?launch through recorded_run.sh}"
DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}"
LOG_PATH="${RUNS_ROOT}/logs/${RUN_ID}.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: Server 23 OGBench Python not found" >&2; exit 1; }
[[ -e "${DATASET_PATH}" ]] || { echo "ERROR: Server 23 dataset not found: ${DATASET_PATH}" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --dataset_path="${DATASET_PATH}" \
  --agent=agents/hiql.py \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.high_alpha=3.0 \
  --agent.low_actor_rep_grad=True \
  --agent.low_alpha=3.0 \
  --agent.p_aug=0.5 \
  --agent.subgoal_steps=10 \
  --train_steps=100000 \
  --save_dir="${RUNS_ROOT}" \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group="${EXP_NAME}" \
  --wandb_mode=offline \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
