#!/usr/bin/env bash
set -euo pipefail

# A800 node3：评测 shared-all GCIQL-Chunk-AWR policy seeds 777/789 的
# final-goal direct-policy 基线；每个 seed 四任务在 GPU1-4 并行，两个 seed
# 顺序运行。配置为 50 episodes、goal/budget=25/50、evaluation seed42。
CLIENT_ID=node3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_IDS=${GPU_IDS:-"1 2 3 4"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_STEPS=100000
POLICY_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks
DATA_ROOT=/data-training/yyf/datasets/lewm
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260901_gciql_chunk_all_seeds777_789_direct_policy_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}}
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/gciql-chunk-all-direct-policy

tasks=(cube pusht reacher tworoom)
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != ${#tasks[@]} )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

cd "$OGBENCH_ROOT/impls"

run_seed() {
  local policy_seed=$1
  local pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${policy_seed}"
    local output_dir="$OUTPUT_ROOT/seed${policy_seed}/$task"
    local task_tmp="$TMP_ROOT/seed${policy_seed}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_4tasks.py \
      --task="$task" --controller=direct_policy --policy-guidance=none \
      --data-root="$DATA_ROOT" \
      --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
      --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
      --goal-offset-steps=25 --eval-budget=50 --action-block=5 \
      --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

run_seed 777
run_seed 789
