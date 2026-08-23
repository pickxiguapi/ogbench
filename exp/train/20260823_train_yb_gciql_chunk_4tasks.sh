#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行训练使用冻结 LeWM 表征的 GCIQL-Chunk；仅允许 pi/qv/all。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REPRESENTATION_MODE=${REPRESENTATION_MODE:?Set REPRESENTATION_MODE to pi, qv, or all}
P_AUG=${P_AUG:-0.0}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
POLICY_SEED=${POLICY_SEED:-0}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

case "$REPRESENTATION_MODE" in
  pi|qv|all)
    MODE_TAG=$REPRESENTATION_MODE
    : "${LEWM_EPOCH:?Set LEWM_EPOCH for shared representation mode $REPRESENTATION_MODE}"
    : "${LEWM_BATCH_SIZE:?Set LEWM_BATCH_SIZE for shared representation mode $REPRESENTATION_MODE}"
    : "${LEWM_SEED:?Set LEWM_SEED for shared representation mode $REPRESENTATION_MODE}"
    ;;
  *)
    echo "REPRESENTATION_MODE must be pi, qv, or all; use 20260823_train_yb_gciql_chunk_4tasks_independent.sh for the independent baseline" >&2
    exit 2
    ;;
esac
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
pids=()
lewm_checkpoints=()
for tag in "${tags[@]}"; do
  lewm_checkpoint="$CLIENT_ROOT/lewm-final/lewm-4tasks/lewm_4tasks_${tag}_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}/weights_epoch_${LEWM_EPOCH}.msgpack"
  if [[ ! -f "$lewm_checkpoint" ]]; then
    echo "Frozen LeWM checkpoint not found: $lewm_checkpoint" >&2
    exit 2
  fi
  lewm_checkpoints+=("$lewm_checkpoint")
done

for i in "${!datasets[@]}"; do
  exp_name="gc4_${tags[$i]}_${MODE_TAG}_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  if (( ${#exp_name} >= 64 )); then
    echo "Experiment name must be shorter than 64 characters: $exp_name" >&2
    exit 2
  fi
  run_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-4tasks/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode="$REPRESENTATION_MODE" \
    --lewm_checkpoint="${lewm_checkpoints[$i]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
