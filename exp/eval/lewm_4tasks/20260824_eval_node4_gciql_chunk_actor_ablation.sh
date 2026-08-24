#!/usr/bin/env bash
set -euo pipefail

# node4：等待 8 个 LeWM 四任务 independent GCIQL-Chunk actor 对照 checkpoint 全部完成，再用 8 卡并行评测 DDPG+BC alpha1 与 AWR alpha1；每项默认 50 episodes。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs}
LEWM_DATA_ROOT=${LEWM_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
WAIT_SECONDS=${WAIT_SECONDS:-60}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom cube pusht reacher tworoom)
actors=(ddpgbc ddpgbc ddpgbc ddpgbc awr awr awr awr)
gpus=(0 1 2 3 4 5 6 7)
checkpoints=()
for i in "${!tasks[@]}"; do
  checkpoints+=("$RUNS_ROOT/gciql-chunk-4tasks-actor-ablation/gc4_${tasks[$i]}_ind_${actors[$i]}_alpha1_n100000_b256_a0.5_sd0")
done

while true; do
  missing=0
  for checkpoint in "${checkpoints[@]}"; do
    [[ -f "$checkpoint/params_100000.pkl" ]] || missing=$((missing + 1))
  done
  (( missing == 0 )) && break
  echo "$(date '+%F %T') waiting for $missing/8 checkpoints"
  sleep "$WAIT_SECONDS"
done

output_root=${OUTPUT_ROOT:-$RUNS_ROOT/evals/2026-08-24_node4_actor_ablation_s100k_ep${NUM_EVAL}_seed${EVAL_SEED}_go${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}}
pids=()
for i in "${!tasks[@]}"; do
  output_dir="$output_root/${tasks[$i]}_${actors[$i]}_alpha1"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="${tasks[$i]}" --mode=policy --data-root="$LEWM_DATA_ROOT" \
    --policy-checkpoint-dir="${checkpoints[$i]}" --policy-checkpoint-step=100000 \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
