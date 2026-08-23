#!/usr/bin/env bash
set -euo pipefail

# Server 23：在 GPU 2/3/4/5 并行启动 Cube、PushT、Reacher、TwoRoom 的 LeWM-JAX epoch-10 CEM 300×30 评测。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
worker="$OGBENCH_ROOT/scripts/eval/20260819_eval_s23_lewm_jax_impala_task_e10_cem300x30.sh"

tasks=(cube pusht reacher tworoom)
gpus=(2 3 4 5)

for i in "${!tasks[@]}"; do
  session="lewm-jax-e10-cem-${tasks[$i]}-20260819"
  tmux new-session -d -s "$session" \
    "CUDA_VISIBLE_DEVICES=${gpus[$i]} bash $worker ${tasks[$i]}"
  echo "Started $session on GPU ${gpus[$i]}"
done
