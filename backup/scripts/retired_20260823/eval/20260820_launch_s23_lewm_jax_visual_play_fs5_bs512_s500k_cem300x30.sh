#!/usr/bin/env bash
set -euo pipefail

# Server 23：在 GPU 2/3/4/5 并行评测 LeWM-JAX Visual Cube single/double/triple 与 Scene 的 500k checkpoint。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
worker="$OGBENCH_ROOT/scripts/eval/20260820_eval_s23_lewm_jax_visual_play_fs5_bs512_s500k_cem300x30.sh"

tasks=(single double triple scene)
gpus=(2 3 4 5)

for i in "${!tasks[@]}"; do
  session="lewm-jax-visual-${tasks[$i]}-fs5-s500k-cem-s23"
  tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" \
    "CUDA_VISIBLE_DEVICES=${gpus[$i]} bash $worker ${tasks[$i]}"
  echo "Started $session on GPU ${gpus[$i]}"
done
