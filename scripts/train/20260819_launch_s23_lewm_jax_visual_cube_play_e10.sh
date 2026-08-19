#!/usr/bin/env bash
set -euo pipefail

# Server 23：在 GPU 2/3/4 并行启动 Visual Cube single/double/triple 的 LeWM-JAX IMPALA 10-epoch 训练。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
worker="$OGBENCH_ROOT/scripts/train/20260819_train_s23_lewm_jax_visual_cube_play_e10.sh"

tasks=(single double triple)
gpus=(2 3 4)

for i in "${!tasks[@]}"; do
  session="lewm-jax-visual-cube-${tasks[$i]}-play-e10-s23"
  tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" \
    "CUDA_VISIBLE_DEVICES=${gpus[$i]} bash $worker ${tasks[$i]}"
  echo "Started $session on GPU ${gpus[$i]}"
done
