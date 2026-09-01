#!/usr/bin/env bash
set -euo pipefail

# A800 node3：用 single-play epoch10 检验 LeWM train-batch 与 inference-running BatchNorm 表征差异，并另存 BN 校准 checkpoint。
CLIENT_ID=node3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_ID=${GPU_ID:-1}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_EPOCHS=${LEWM_EPOCHS:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
SAMPLE_BATCHES=${SAMPLE_BATCHES:-8}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-200}
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/lewm-ogbench8}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/diagnostics}

tag=cs_play
env_name=visual-cube-single-play-v0
checkpoint_dir="$LEWM_RUN_ROOT/lewm_ogbench8_${tag}_e${LEWM_EPOCHS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
output_dir="$DIAGNOSTIC_ROOT/lewm_ogbench8_${tag}_epoch${LEWM_EPOCH}_bn_geometry"
mkdir -p "$output_dir"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES="$GPU_ID" XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" diagnose_lewm_bn_geometry.py \
  --checkpoint="$checkpoint_dir/weights_epoch_${LEWM_EPOCH}.msgpack" \
  --dataset-path="$OGBENCH_DATA_DIR/${env_name}.npz" \
  --validation-dataset-path="$OGBENCH_DATA_DIR/${env_name}-val.npz" \
  --output-dir="$output_dir" \
  --batch-size="$LEWM_BATCH_SIZE" \
  --sample-batches="$SAMPLE_BATCHES" \
  --calibration-batches="$CALIBRATION_BATCHES" \
  --seed="$LEWM_SEED" \
  >"$output_dir/diagnose.log" 2>&1
