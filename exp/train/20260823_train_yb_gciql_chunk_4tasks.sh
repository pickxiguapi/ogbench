#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行训练 LeWM-4Tasks GCIQL-Chunk；REPRESENTATION_MODE 可设 independent/pi/qv/all，默认正式消融关闭增强。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}
P_AUG=${P_AUG:-0.0}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
POLICY_SEED=${POLICY_SEED:-0}
LEWM_SEED=${LEWM_SEED:-3072}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

case "$REPRESENTATION_MODE" in independent|pi|qv|all) ;; *) echo "REPRESENTATION_MODE must be independent, pi, qv, or all" >&2; exit 2 ;; esac
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
pids=()

for i in "${!datasets[@]}"; do
  exp_name="gciql_chunk_4tasks_${tags[$i]}_${REPRESENTATION_MODE}_s${POLICY_STEPS}_bs${POLICY_BATCH_SIZE}_paug${P_AUG}_s${POLICY_SEED}"
  run_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-4tasks/$exp_name"
  lewm_args=()
  if [[ "$REPRESENTATION_MODE" != independent ]]; then
    lewm_dir="$CLIENT_ROOT/lewm-final/lewm-4tasks/lewm_4tasks_${tags[$i]}_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
    lewm_args=(--lewm_checkpoint="$lewm_dir/weights_epoch_${LEWM_EPOCH}.msgpack")
  fi
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode="$REPRESENTATION_MODE" "${lewm_args[@]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
