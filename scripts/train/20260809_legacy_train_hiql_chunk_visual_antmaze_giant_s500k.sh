#!/usr/bin/env bash
set -euo pipefail

mkdir -p /data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-giant-navigate /home/dzb/ogbench/logs/hiql_chunk/visual-antmaze-giant-navigate

tmux new-session -d -s hiql-chunk-visual-antmaze-giant -c /home/dzb/ogbench/impls \
  "CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  /home/dzb/ogbench/.venv/bin/python /home/dzb/ogbench/impls/main.py \
  --env_name=visual-antmaze-giant-navigate-v0 \
  --agent=/home/dzb/ogbench/impls/agents/hiql_chunk.py \
  --agent.batch_size=512 --agent.encoder=impala_small \
  --agent.lr=0.0003 --agent.tau=0.005 --agent.discount=0.995 \
  --agent.subgoal_steps=25 --agent.chunk_size=2 --agent.expectile=0.5 \
  --agent.high_alpha=3.0 --agent.low_alpha=3.0 \
  --agent.low_actor_rep_grad=True --agent.const_std=True --agent.p_aug=0.0 \
  --train_steps=500000 --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
  --eval_episodes=50 --video_episodes=1 --eval_on_cpu=0 --seed=0 \
  --run_group=hiql-chunk-visual-antmaze-giant-bs512 \
  --save_dir=/data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-giant-navigate \
  2>&1 | tee -a /home/dzb/ogbench/logs/hiql_chunk/visual-antmaze-giant-navigate/train.log"

tmux ls | grep hiql-chunk-visual-antmaze-giant
