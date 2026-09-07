#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/configs/lewmpp_paths.env"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/impls"

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  save_dir="$EXPERIMENT_ROOT/open-source-retrain/subgoal-generators/general_uniform_future/mlp/$task"
  mkdir -p "$save_dir"
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    "$PYTHON_BIN" train_subgoal_generator.py \
      --latent-dataset="$LEWM_DATA_ROOT/lewm-latents/$task.h5" \
      --save-dir="$save_dir" \
      --exp-name="lewmpp_general_uniform_future_mlp_${task}_seed0" \
      --generator-family=general_uniform_future \
      --goal-sampling=hiql_uniform_future_same_trajectory \
      --generator-type=mlp \
      --seed=0 \
      --split-seed=0 \
      --train-fraction=0.95 \
      --subgoal-steps=10 \
      --action-block=5 \
      --history-size=3 \
      --train-steps=200000 \
      --batch-size=1024 \
      --hidden-dim=512 \
      --hidden-dims=512 512 512 \
      --depth=4 \
      --num-heads=8 \
      --ff-dim=2048 \
      --time-dim=64 \
      --flow-sampling-steps=16 \
      --num-samples=1 \
      --ema-decay=0.9999 \
      --learning-rate=1e-4 \
      --final-learning-rate=1e-5 \
      --warmup-steps=5000 \
      --weight-decay=1e-4 \
      --gradient-clip=1.0 \
      --validation-pairs=10000 \
      --eval-batch-size=1024 \
      --log-interval=1000 \
      --eval-interval=10000 \
      --checkpoint-interval=25000 \
      --resume \
      2>&1 | tee "$save_dir/train.log"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
