#!/usr/bin/env bash
set -euo pipefail

# Shared formal trainer. Call through one of the task/server wrappers beside this file.
: "${CLIENT_ID:?CLIENT_ID is required}"
: "${LATENT_DATASET:?LATENT_DATASET is required}"
: "${SUBGOAL_RUNS_ROOT:?SUBGOAL_RUNS_ROOT is required}"
: "${EXP_NAME:?EXP_NAME is required}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=${OGBENCH_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_ID=${GPU_ID:-0}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
VALIDATION_PAIRS=${VALIDATION_PAIRS:-10000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1024}
LOG_INTERVAL=${LOG_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25000}
run_dir="$SUBGOAL_RUNS_ROOT/$EXP_NAME"

mkdir -p "$run_dir"
cd "$OGBENCH_ROOT/impls"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" train_latent_subgoal_gcbc.py \
  --latent-dataset="$LATENT_DATASET" \
  --save-dir="$run_dir" \
  --exp-name="$EXP_NAME" \
  --architecture=transformer_flow \
  --seed="$TRAIN_SEED" \
  --split-seed=0 \
  --train-fraction=0.95 \
  --subgoal-steps=10 \
  --train-steps="$TRAIN_STEPS" \
  --batch-size="$TRAIN_BATCH_SIZE" \
  --model-dim=384 \
  --num-layers=8 \
  --num-heads=8 \
  --mlp-dim=1536 \
  --flow-sampling-steps=16 \
  --flow-solver=heun \
  --ema-decay=0.9999 \
  --learning-rate=1e-4 \
  --final-learning-rate=1e-5 \
  --warmup-steps="$WARMUP_STEPS" \
  --weight-decay=1e-4 \
  --gradient-clip=1.0 \
  --validation-pairs="$VALIDATION_PAIRS" \
  --eval-batch-size="$EVAL_BATCH_SIZE" \
  --log-interval="$LOG_INTERVAL" \
  --eval-interval="$EVAL_INTERVAL" \
  --checkpoint-interval="$CHECKPOINT_INTERVAL" \
  --resume \
  2>&1 | tee -a "$run_dir/train.log"
