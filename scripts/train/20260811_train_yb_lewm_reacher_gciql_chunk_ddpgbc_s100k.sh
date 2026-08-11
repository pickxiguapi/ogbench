#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
STABLEWM_ROOT="/root/data/yyf/stable-worldmodel"
RUNS_ROOT="/root/data/yyf/lewm-runs"
GPU_ID=1
EXP_NAME="GCIQLChunkDDPGBC_ogbench_reacher_k5_bs256_s100k_seed0_alpha1_expectile09_aug05"
LOG_PATH="${RUNS_ROOT}/logs/${EXP_NAME}.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -d "${STABLEWM_ROOT}/datasets/reacher.lance" ]] || { echo "ERROR: Reacher Lance dataset not found" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name=visual-lewm-reacher-v0 \
  --dataset_path="${STABLEWM_ROOT}/datasets/reacher.lance" \
  --agent=agents/gciql_chunk.py \
  --agent.actor_loss=ddpgbc \
  --agent.alpha=1.0 \
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
  --run_group=lewm-reacher-gciql-chunk-ddpgbc-k5-bs256-s100k \
  --wandb_mode=online \
  --seed=0 \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
