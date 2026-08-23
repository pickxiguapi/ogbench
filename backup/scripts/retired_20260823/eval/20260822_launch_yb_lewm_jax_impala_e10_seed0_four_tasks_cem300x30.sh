#!/usr/bin/env bash
set -euo pipefail

# 英博云：在 GPU 0–3 并行启动训练 seed0 的四个 LeWM-JAX epoch10 CEM 300×30 评测。
cd /root/data/yyf/ogbench-new
worker=scripts/eval/20260822_eval_yb_lewm_jax_impala_task_e10_seed0_cem300x30.sh

for task in cube pusht reacher tworoom; do
  session="lewm-jax-cem-s0-${task}-20260822"
  tmux new-session -d -s "$session" "bash $worker $task"
done
