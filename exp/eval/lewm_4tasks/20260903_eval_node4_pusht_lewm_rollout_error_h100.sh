#!/usr/bin/env bash
set -euo pipefail

# A800 node4：只评估 PushT seed666 frozen LeWM；固定 episode-level diagnostic split，512 trajectories，按 environment steps 报告 H=5..100（action block=5），并标出 LeWM++ local horizon k=10。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_ID=${GPU_ID:-0}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-512}
MAX_HORIZON=${MAX_HORIZON:-100}
ACTION_BLOCK=${ACTION_BLOCK:-5}
LOCAL_HORIZON=${LOCAL_HORIZON:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_SEED=${EVAL_SEED:-42}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-/data-training/yyf/ogbench-lewm-policy-runs/diagnostics/20260903_pusht_lewm_rollout_error_h100}
LATENT_DATASET=${LATENT_DATASET:-/data-training/yyf/datasets/lewm-latents/pusht_expert_train__lewm_s666_e10_z192.h5}
LEWM_CHECKPOINT=${LEWM_CHECKPOINT:-/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack}

mkdir -p "$OUTPUT_DIR"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" analyze_lewm_rollout_error.py \
  --tasks pusht \
  --latent-datasets "$LATENT_DATASET" \
  --checkpoints "$LEWM_CHECKPOINT" \
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
