#!/usr/bin/env bash
set -euo pipefail

# A800 node4：使用当前 mixed frozen LeWM（PushT seed666，其余 seed3072）在固定 episode-level diagnostic split 上测 open-loop latent rollout error；512 trajectories/task、H=5..50，并标出 LeWM++ local horizon k=10。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_ID=${GPU_ID:-0}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-512}
MAX_HORIZON=${MAX_HORIZON:-50}
ACTION_BLOCK=${ACTION_BLOCK:-5}
LOCAL_HORIZON=${LOCAL_HORIZON:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_SEED=${EVAL_SEED:-42}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-/data-training/yyf/ogbench-lewm-policy-runs/diagnostics/20260902_lewm_rollout_error_mixed}
LATENT_ROOT=${LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
SEED666_ROOT=${SEED666_ROOT:-/data-training/yyf/models/lewm-jax-seed666}
SEED3072_ROOT=${SEED3072_ROOT:-/data-training/yyf/models/lewm-jax-seed3072}

mkdir -p "$OUTPUT_DIR"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" analyze_lewm_rollout_error.py \
  --tasks cube pusht reacher tworoom \
  --latent-datasets \
    "$LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5" \
    "$LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5" \
    "$LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5" \
    "$LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5" \
  --checkpoints \
    "$SEED3072_ROOT/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack" \
    "$SEED666_ROOT/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack" \
    "$SEED3072_ROOT/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack" \
    "$SEED3072_ROOT/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack" \
  --output-dir "$OUTPUT_DIR" \
  --max-horizon "$MAX_HORIZON" \
  --action-block "$ACTION_BLOCK" \
  --local-horizon "$LOCAL_HORIZON" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --episode-holdout-fraction 0.1 \
  --batch-size "$BATCH_SIZE" \
  --seed "$EVAL_SEED" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  2>&1 | tee "$OUTPUT_DIR/eval.log"
