#!/usr/bin/env bash
set -euo pipefail

# 英博云：将 seed0 的四个 LeWM-JAX IMPALA e10 训练并行启动到 GPU 0–3。
cd /root/data/yyf/ogbench-new

for task in cube pusht reacher tworoom; do
  session="lewm-jax-e10-s0-${task}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "bash scripts/train/20260821_train_yb_lewm_jax_impala_task_e10_seed0.sh $task"
done
