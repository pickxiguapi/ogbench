#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
VENV_DIR="${VENV_DIR:-/data/yyf/H-LeWM/envs/ogbench}"
WANDB_MODE="${WANDB_MODE:-offline}"

case "${TASK}" in
  tworoom)
    ENV_NAME=visual-lewm-tworoom-v0
    DATASET_NAME=tworoom.lance
    ;;
  reacher)
    ENV_NAME=visual-lewm-reacher-v0
    DATASET_NAME=reacher.lance
    ;;
  pusht)
    ENV_NAME=visual-lewm-pusht-expert-train-v0
    DATASET_NAME=pusht_expert_train.lance
    ;;
  cube)
    ENV_NAME=visual-lewm-cube-single-expert-v0
    DATASET_NAME=cube_single_expert.lance
    ;;
  *)
    echo "Usage: bash $0 {tworoom|reacher|pusht|cube}" >&2
    exit 2
    ;;
esac

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh" >&2
  exit 2
}
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${DATA_ROOT}/${DATASET_NAME}/.conversion_complete" ]] || {
  echo "ERROR: complete Lance dataset not found: ${DATA_ROOT}/${DATASET_NAME}" >&2
  exit 1
}

EXP_NAME="${EXPERIMENT_EXP_NAME:-GCIQLChunkAWR_ogbench_lewm_${TASK}_k5_bs256_s100k_seed0_alpha3_expectile09_aug05_s11}"
LOG_DIR="${RUNS_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.log"
mkdir -p "${RUNS_ROOT}/wandb" "${LOG_DIR}"
cd "${OGBENCH_ROOT}/impls"

export PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls${PYTHONPATH:+:${PYTHONPATH}}"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${VENV_DIR}/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --dataset_path="${DATA_ROOT}/${DATASET_NAME}" \
  --agent=agents/gciql_chunk.py \
  --agent.actor_loss=awr \
  --agent.alpha=3.0 \
  --agent.chunk_size=5 \
  --agent.batch_size=256 \
  --agent.lr=3e-4 \
  --agent.discount=0.99 \
  --agent.expectile=0.9 \
  --agent.tau=0.005 \
  --agent.encoder=impala_small \
  --agent.p_aug=0.5 \
  --train_steps=100000 \
  --save_dir="${RUNS_ROOT}" \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-gciql-chunk-awr-k5-bs256-s100k-s11 \
  --wandb_mode="${WANDB_MODE}" \
  --seed=0 \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
