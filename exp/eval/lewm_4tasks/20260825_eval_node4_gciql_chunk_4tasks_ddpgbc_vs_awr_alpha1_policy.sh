#!/usr/bin/env bash
set -euo pipefail

# node4：8 卡并行评测 independent GCIQL-Chunk actor 对照；GPU 0–3 为 DDPG+BC alpha1，GPU 4–7 为 AWR alpha1，直接 policy-only 各测 50 episodes（seed42、g25/b50）。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom cube pusht reacher tworoom)
tags=(cube pusht reacher tworoom cube pusht reacher tworoom)
gpus=(0 1 2 3 4 5 6 7)
actor_losses=(ddpgbc ddpgbc ddpgbc ddpgbc awr awr awr awr)
policy_root="$CLIENT_ROOT/ogbench-lewm-policy-runs/gciql-chunk-4tasks-actor-ablation"
output_root="$policy_root/evals/2026-08-25_policy_only_ep50_seed42_g25_b50"
pids=()

for i in "${!tasks[@]}"; do
  policy_dir="$policy_root/gc4_${tags[$i]}_ind_${actor_losses[$i]}_alpha1_n100000_b256_a0.5_sd0"
  output_dir="$output_root/${actor_losses[$i]}_${tags[$i]}"
  mkdir -p "$output_dir/tmp"
  TMPDIR="$output_dir/tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="${tasks[$i]}" --mode=policy --data-root="$LEWM_DATA_ROOT" \
    --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step=100000 \
    --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
