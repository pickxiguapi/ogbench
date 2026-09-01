#!/usr/bin/env bash
set -euo pipefail

# A800 node4：评测三个已完成的 OGBench play shared-all GCIQL-Chunk-AWR seed0 500k direct policies；每个内部 task 50 episodes、eval seed42。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_SEED=${POLICY_SEED:-0}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
P_AUG=${P_AUG:-0.0}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_ROOT=${POLICY_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/gciql-chunk-ogbench8}
EVAL_ROOT=${EVAL_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0)
tags=(cs_play cd_play ct_play)
gpus=(0 1 2)
output_root="$EVAL_ROOT/gciql_chunk_all_500k_seed0_direct_policy_ep${NUM_EVAL}_seed${EVAL_SEED}"

for tag in "${tags[@]}"; do
  policy_dir="$POLICY_ROOT/gc8_${tag}_all_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  if [[ ! -s "$policy_dir/flags.json" || ! -s "$policy_dir/params_${POLICY_STEPS}.pkl" ]]; then
    echo "Missing complete policy checkpoint: $policy_dir" >&2
    exit 2
  fi
done

cd "$OGBENCH_ROOT/impls"
pids=()
for i in "${!envs[@]}"; do
  policy_dir="$POLICY_ROOT/gc8_${tags[$i]}_all_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  output_dir="$output_root/${tags[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_ogbench_env_8tasks.py \
    --env-name="${envs[$i]}" \
    --dataset-path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --controller=direct_policy --policy-guidance=none \
    --policy-checkpoint-dir="$policy_dir" \
    --policy-checkpoint-step="$POLICY_STEPS" \
    --policy-action-space=environment \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --output="$output_dir/result.json" \
    >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
