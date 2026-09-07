#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/configs/lewmpp_paths.env"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
DATASETS=(
  "$LEWM_DATA_ROOT/cube_single_expert.lance"
  "$LEWM_DATA_ROOT/pusht_expert_train.lance"
  "$LEWM_DATA_ROOT/reacher.lance"
  "$LEWM_DATA_ROOT/tworoom.lance"
)
LEWM_CHECKPOINTS=(
  "$LEWM_CUBE_CHECKPOINT"
  "$LEWM_PUSHT_CHECKPOINT"
  "$LEWM_REACHER_CHECKPOINT"
  "$LEWM_TWOROOM_CHECKPOINT"
)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/impls"

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  save_dir="$ACTION_PRIOR_RUN_ROOT/$task"
  log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    "$PYTHON_BIN" action-prior-chunk.py \
      --dataset_path="${DATASETS[$index]}" \
      --lewm_checkpoint="${LEWM_CHECKPOINTS[$index]}" \
      --save_dir="$save_dir" \
      --train_steps=100000 \
      --save_interval=100000 \
      --log_interval=5000 \
      --batch_size=256 \
      --seed=777 \
      --lr=3e-4 \
      --discount=0.99 \
      --expectile=0.9 \
      --tau=0.005 \
      --chunk_size=5 \
      --alpha=3.0 \
      --p_aug=0.0 \
      --validation_fraction=0.05 \
      --representation_mode=all \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
