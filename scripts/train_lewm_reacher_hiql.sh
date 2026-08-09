#!/usr/bin/env bash
set -euo pipefail

cd /home/dzb/ogbench/impls
mkdir -p /data/dzb/lewm-runs/wandb

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/dzb/lewm-runs/wandb

/home/dzb/ogbench/.venv/bin/python main.py \
  --env_name=visual-lewm-reacher-v0 \
  --dataset_path=/data/dzb/stablewm-data/datasets/reacher.lance \
  --agent=agents/hiql.py \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.high_alpha=3.0 \
  --agent.low_actor_rep_grad=True \
  --agent.low_alpha=3.0 \
  --agent.p_aug=0.5 \
  --agent.subgoal_steps=10 \
  --train_steps=100000 \
  --save_dir=/data/dzb/lewm-runs \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-reacher-visual-hiql-bs256-100k \
  --wandb_mode=offline \
  --eval_episodes=0 \
  --video_episodes=0
