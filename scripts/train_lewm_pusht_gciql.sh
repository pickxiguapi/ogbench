#!/usr/bin/env bash
set -euo pipefail

cd /root/data/yyf/ogbench/impls
mkdir -p /root/data/yyf/lewm-runs/wandb

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_CACHE_DIR=/root/data/yyf/.cache/uv
export WANDB_DIR=/root/data/yyf/lewm-runs/wandb

/home/yyf/.local/bin/uv run --project /root/data/yyf/ogbench --extra train python main.py \
  --env_name=visual-lewm-pusht-expert-train-v0 \
  --dataset_path=/root/data/yyf/stable-worldmodel/datasets/pusht_expert_train.lance \
  --agent=agents/gciql.py \
  --agent.alpha=1.0 \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.p_aug=0.5 \
  --train_steps=100000 \
  --save_dir=/root/data/yyf/lewm-runs \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-pusht-visual-gciql-bs256-100k \
  --wandb_mode=offline \
  --eval_episodes=0 \
  --video_episodes=0
