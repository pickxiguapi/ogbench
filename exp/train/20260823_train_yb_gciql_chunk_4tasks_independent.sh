#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行训练纯 GCIQL-Chunk baseline；不加载任何 LeWM 模型或 checkpoint。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
P_AUG=${P_AUG:-0.0}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
POLICY_SEED=${POLICY_SEED:-0}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
pids=()

for i in "${!datasets[@]}"; do
  exp_name="gc4_${tags[$i]}_ind_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  if (( ${#exp_name} >= 64 )); then
    echo "Experiment name must be shorter than 64 characters: $exp_name" >&2
    exit 2
  fi
  run_dir="$GCIQL_RUNS_ROOT/gciql-chunk-4tasks/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$GCIQL_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode=independent \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
