#!/usr/bin/env bash
set -euo pipefail

# 四任务共用正式训练入口：frozen LeWM z192、3 帧历史、HIQL 同轨迹未来 goal、subgoal K10/action block5、仅 CFM loss、LeFlow 风格 Transformer、200k、bs1024、seed0。
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
HISTORY_SIZE=${HISTORY_SIZE:-3}
SUBGOAL_STEPS=${SUBGOAL_STEPS:-10}
ACTION_BLOCK=${ACTION_BLOCK:-5}
GOAL_SAMPLING=${GOAL_SAMPLING:-uniform_future}
NUM_SAMPLES=${NUM_SAMPLES:-8}
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
  --architecture=latent_path_flow \
  --seed="$TRAIN_SEED" \
  --split-seed=0 \
  --train-fraction=0.95 \
  --subgoal-steps="$SUBGOAL_STEPS" \
  --action-block="$ACTION_BLOCK" \
  --goal-sampling="$GOAL_SAMPLING" \
  --history-size="$HISTORY_SIZE" \
  --train-steps="$TRAIN_STEPS" \
  --batch-size="$TRAIN_BATCH_SIZE" \
  --hidden-dim=512 \
  --depth=4 \
  --num-heads=8 \
  --ff-dim=2048 \
  --time-dim=64 \
  --flow-sampling-steps=16 \
  --flow-solver=euler \
  --num-samples="$NUM_SAMPLES" \
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
