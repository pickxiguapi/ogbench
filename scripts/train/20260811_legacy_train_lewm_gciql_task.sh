#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train/20260811_legacy_train_lewm_gciql_task.sh {tworoom|reacher|pusht|cube}" >&2
  exit 2
fi

TASK="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${OGBENCH_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMPLS_DIR="${IMPLS_DIR:-${REPO_ROOT}/impls}"
DATASETS_DIR="${DATASETS_DIR:-/root/data/yyf/stable-worldmodel/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/root/data/yyf/lewm-runs}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
if (( TRAIN_STEPS % 1000 == 0 )); then
  TRAIN_LABEL="$((TRAIN_STEPS / 1000))k"
else
  TRAIN_LABEL="${TRAIN_STEPS}"
fi

case "${TASK}" in
  tworoom)
    ENV_NAME="visual-lewm-tworoom-v0"
    DATASET_NAME="tworoom.lance"
    ;;
  reacher)
    ENV_NAME="visual-lewm-reacher-v0"
    DATASET_NAME="reacher.lance"
    ;;
  pusht)
    ENV_NAME="visual-lewm-pusht-expert-train-v0"
    DATASET_NAME="pusht_expert_train.lance"
    ;;
  cube)
    ENV_NAME="visual-lewm-cube-single-expert-v0"
    DATASET_NAME="cube_single_expert.lance"
    ;;
  *)
    echo "ERROR: unsupported task: ${TASK}" >&2
    exit 2
    ;;
esac

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -d "${IMPLS_DIR}" ]] || {
  echo "ERROR: OGBench impls directory not found: ${IMPLS_DIR}" >&2
  exit 1
}
[[ -e "${DATASETS_DIR}/${DATASET_NAME}" ]] || {
  echo "ERROR: dataset not found: ${DATASETS_DIR}/${DATASET_NAME}" >&2
  exit 1
}

mkdir -p "${RUNS_ROOT}/wandb"
cd "${IMPLS_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${RUNS_ROOT}/wandb"

exec "${PYTHON_BIN}" main.py \
  "--env_name=${ENV_NAME}" \
  "--dataset_path=${DATASETS_DIR}/${DATASET_NAME}" \
  --agent=agents/gciql.py \
  --agent.alpha=1.0 \
  "--agent.batch_size=${BATCH_SIZE}" \
  --agent.encoder=impala_small \
  --agent.p_aug=0.5 \
  "--train_steps=${TRAIN_STEPS}" \
  "--seed=${SEED}" \
  "--save_dir=${RUNS_ROOT}" \
  --log_interval=5000 \
  "--save_interval=${TRAIN_STEPS}" \
  "--run_group=lewm-${TASK}-visual-gciql-bs${BATCH_SIZE}-${TRAIN_LABEL}" \
  "--wandb_mode=${WANDB_MODE}" \
  --eval_episodes=0 \
  --video_episodes=0
