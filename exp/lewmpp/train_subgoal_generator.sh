#!/usr/bin/env bash
set -euo pipefail

# Unified subgoal-generator training entrypoint.
# GENERATOR_TYPE switches mlp / endpoint_flow / latent_path_flow without changing data protocol.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${TASK:?Set TASK}"
: "${FAMILY:?Set FAMILY to goalmax25 or general_uniform_future}"
: "${GENERATOR_TYPE:=latent_path_flow}"
: "${LATENT_DATASET:?Set LATENT_DATASET}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
case "$FAMILY" in
  goalmax25|general_uniform_future) ;;
  *) echo "Unsupported FAMILY: $FAMILY" >&2; exit 2 ;;
esac
case "$GENERATOR_TYPE" in
  mlp|endpoint_flow|latent_path_flow) ;;
  *) echo "Unsupported GENERATOR_TYPE: $GENERATOR_TYPE" >&2; exit 2 ;;
esac
require_file "$LATENT_DATASET" "frozen LeWM latent cache"

TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_SEED=${TRAIN_SEED:-0}
GENERATOR_SAMPLES=8
if [[ "$GENERATOR_TYPE" == mlp ]]; then GENERATOR_SAMPLES=1; fi
run_dir="$OUTPUT_ROOT/$FAMILY/$GENERATOR_TYPE/$TASK/seed${TRAIN_SEED}"
mkdir -p "$run_dir"
cd "$OGBENCH_ROOT/impls"
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" train_subgoal_generator.py \
    --latent-dataset="$LATENT_DATASET" \
    --save-dir="$run_dir" \
    --exp-name="lewmpp_${FAMILY}_${GENERATOR_TYPE}_${TASK}_seed${TRAIN_SEED}" \
    --generator-family="$FAMILY" \
    --generator-type="$GENERATOR_TYPE" \
    --seed="$TRAIN_SEED" \
    --split-seed=0 \
    --subgoal-steps=10 \
    --action-block=5 \
    --history-size=3 \
    --train-steps="$TRAIN_STEPS" \
    --batch-size=1024 \
    --hidden-dim=512 \
    --hidden-dims=512 512 512 \
    --depth=4 \
    --num-heads=8 \
    --ff-dim=2048 \
    --time-dim=64 \
    --flow-sampling-steps=16 \
    --num-samples="$GENERATOR_SAMPLES" \
    --ema-decay=0.9999 \
    --learning-rate=1e-4 \
    --final-learning-rate=1e-5 \
    --warmup-steps=5000 \
    --resume
