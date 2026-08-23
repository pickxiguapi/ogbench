#!/usr/bin/env bash
set -euo pipefail

# Server 23：在 GPU 2/3 并行评测 LeWM-JAX Visual Cube single/double epoch-10。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
worker="$OGBENCH_ROOT/scripts/eval/20260819_eval_s23_lewm_jax_visual_cube_play_e10_cem300x30.sh"

tasks=(single double)
gpus=(2 3)

for i in "${!tasks[@]}"; do
  session="lewm-jax-visual-cube-${tasks[$i]}-e10-cem-s23"
  tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" \
    "CUDA_VISIBLE_DEVICES=${gpus[$i]} bash $worker ${tasks[$i]}"
  echo "Started $session on GPU ${gpus[$i]}"
done
