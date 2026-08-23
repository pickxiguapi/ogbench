#!/usr/bin/env bash
set -euo pipefail

# Server 11：训练指定 LeWM 单任务的 HIQL-Chunk-Share-V；共享高低层 reachability value，s100k、k5、sg10、bs256、seed0。
TASK="${1:-}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
VENV_DIR="${VENV_DIR:-/data/yyf/H-LeWM/envs/ogbench}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"

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

[[ -x "${VENV_DIR}/bin/python" ]] || { echo "ERROR: OGBench Python not found: ${VENV_DIR}" >&2; exit 1; }
[[ -f "${DATA_ROOT}/${DATASET_NAME}/.conversion_complete" ]] || {
  echo "ERROR: complete Lance dataset not found: ${DATA_ROOT}/${DATASET_NAME}" >&2
  exit 1
}
[[ -f "${OGBENCH_ROOT}/impls/agents/hiql_chunk_share_v.py" ]] || {
  echo "ERROR: HIQL-Chunk-Share-V agent not found under ${OGBENCH_ROOT}" >&2
  exit 1
}

RUN_GROUP="hiql-chunk-share-v-lewm-${TASK}-k5-sg10-bs256-s100k"
EXP_NAME="${EXP_NAME:-HIQLChunkShareV_ogbench_lewm_${TASK}_k5_sg10_bs256_s100k_seed0_aH3_aL3_exp07_aug05_${RUN_STAMP}}"
TASK_RUNS_ROOT="${RUNS_ROOT}/${TASK}"
LOG_DIR="${TASK_RUNS_ROOT}/logs/${RUN_GROUP}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.log"
mkdir -p "${TASK_RUNS_ROOT}/wandb" "${LOG_DIR}"
cd "${OGBENCH_ROOT}/impls"

export PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls${PYTHONPATH:+:${PYTHONPATH}}"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${TASK_RUNS_ROOT}/wandb" \
"${VENV_DIR}/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --dataset_path="${DATA_ROOT}/${DATASET_NAME}" \
  --agent=agents/hiql_chunk_share_v.py \
  --agent.chunk_size=5 \
  --agent.subgoal_steps=10 \
  --agent.batch_size=256 \
  --agent.lr=3e-4 \
  --agent.discount=0.99 \
  --agent.expectile=0.7 \
  --agent.low_alpha=3.0 \
  --agent.high_alpha=3.0 \
  --agent.tau=0.005 \
  --agent.encoder=impala_small \
  --agent.low_actor_rep_grad=True \
  --agent.p_aug=0.5 \
  --train_steps="${TRAIN_STEPS}" \
  --save_dir="${TASK_RUNS_ROOT}" \
  --log_interval="${LOG_INTERVAL}" \
  --save_interval="${SAVE_INTERVAL}" \
  --run_group="${RUN_GROUP}" \
  --wandb_mode="${WANDB_MODE}" \
  --seed=0 \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
