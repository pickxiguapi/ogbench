#!/usr/bin/env bash
set -euo pipefail

cd /home/dzb/ogbench/impls
mkdir -p /data/dzb/lewm-runs/wandb

export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/dzb/lewm-runs/wandb

/home/dzb/ogbench/.venv/bin/python main.py \
  --env_name=visual-lewm-reacher-v0 \
  --dataset_path=/data/dzb/stablewm-data/datasets/reacher.lance \
  --agent=agents/gciql.py \
  --agent.alpha=1.0 \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.p_aug=0.5 \
  --train_steps=100000 \
  --save_dir=/data/dzb/lewm-runs \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-reacher-visual-gciql-bs256-100k \
  --wandb_mode=offline \
  --eval_episodes=0 \
  --video_episodes=0
