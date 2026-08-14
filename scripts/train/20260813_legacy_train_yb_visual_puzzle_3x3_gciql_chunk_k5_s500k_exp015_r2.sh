#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
RUNS_ROOT="/root/data/yyf/ogbench-native-runs"
GPU_ID=3
EXP_NAME="GCIQLChunk_ogbench_visual_puzzle_3x3_play_k5_bs256_s500k_seed0_alpha1_expectile09_aug05_ddpgbc"
LOG_PATH="${RUNS_ROOT}/logs/${EXP_NAME}_v2.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/agents/gciql_chunk.py" ]] || { echo "ERROR: GCIQL-Chunk agent not found" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gciql_chunk.py \
  --agent.actor_loss=ddpgbc --agent.alpha=1.0 --agent.chunk_size=5 --agent.batch_size=256 \
  --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.9 --agent.tau=0.005 \
  --agent.encoder=impala_small --agent.p_aug=0.5 --train_steps=500000 \
  --save_dir="${RUNS_ROOT}" --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
  --run_group=EXP015_GCIQLChunk_puzzle3x3_k5 --wandb_mode=online --seed=0 \
  --eval_episodes=50 --eval_on_cpu=0 --video_episodes=0 2>&1 | tee "${LOG_PATH}"
