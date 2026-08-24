#!/usr/bin/env bash
set -euo pipefail

# node4：使用空闲 GPU 评测从 node3 同步的 Reacher/TwoRoom independent GCIQL-Chunk-AWR seed777 100k checkpoint；Cube/PushT 已完成同协议评测。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LEWM_DATA_ROOT=${LEWM_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(reacher tworoom)
gpus=(1 6)
checkpoints=(
  "$RUNS_ROOT/node3-checkpoints/GCAWR_lewm_reacher_s777_100k"
  "$RUNS_ROOT/node3-checkpoints/GCAWR_lewm_tworoom_s777_100k"
)
output_root=${OUTPUT_ROOT:-$RUNS_ROOT/evals/2026-08-24_node3_GCAWR_seed777_remaining_s100k_ep${NUM_EVAL}_seed${EVAL_SEED}_go${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}}
pids=()

for i in "${!tasks[@]}"; do
  output_dir="$output_root/${tasks[$i]}"
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
