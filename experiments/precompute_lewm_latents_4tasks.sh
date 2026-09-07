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
  output="$LEWM_LATENT_ROOT/$task.h5"
  log_file="$LEWM_LATENT_ROOT/$task.log"
  mkdir -p "$LEWM_LATENT_ROOT"

  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    "$PYTHON_BIN" precompute_lewm_latents.py \
      --task="$task" \
      --lance-path="${DATASETS[$index]}" \
      --checkpoint="${LEWM_CHECKPOINTS[$index]}" \
      --output="$output" \
      --batch-size=512 \
      --decode-workers=12 \
      --output-dtype=float32 \
      --flush-every-batches=20 \
      --log-every-batches=20 \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
