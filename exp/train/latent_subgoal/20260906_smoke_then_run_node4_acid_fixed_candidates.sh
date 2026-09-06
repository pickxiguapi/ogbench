#!/usr/bin/env bash
set -euo pipefail

# A800 node4: wait for four free GPUs, validate the complete ACID training and
# fixed-candidate simulator-replay path on a tiny run, then start the paper run.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/20260906_run_node4_acid_subgoal_reachability.sh"

SMOKE_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/smoke/20260906_acid_fixed_candidates
env \
  MODE=driver PIPELINE=fixed_candidates WAIT_FOR_GPUS=1 GPU_IDS="0 1 2 3 4 5 6 7" \
  TRAIN_STEPS=10 IDM_BATCH_SIZE=4 IDM_WARMUP_STEPS=2 \
  IDM_VALIDATION_PAIRS=8 IDM_EVAL_BATCH_SIZE=4 \
  IDM_LOG_INTERVAL=5 IDM_EVAL_INTERVAL=5 IDM_CHECKPOINT_INTERVAL=5 \
  NUM_DIAGNOSTIC_STATES=1 DIAGNOSTIC_CANDIDATES=4 DIAGNOSTIC_TOPK=2 \
  IDM_ROOT="$SMOKE_ROOT/idm" FIXED_ROOT="$SMOKE_ROOT/fixed" \
  EVAL_ROOT="$SMOKE_ROOT/unused_selected" TMP_ROOT="$SMOKE_ROOT/tmp" \
  bash "$RUNNER"

env \
  MODE=driver PIPELINE=fixed_candidates WAIT_FOR_GPUS=1 GPU_IDS="0 1 2 3 4 5 6 7" \
  TRAIN_STEPS=200000 IDM_BATCH_SIZE=256 IDM_WARMUP_STEPS=2000 \
  IDM_VALIDATION_PAIRS=50000 IDM_EVAL_BATCH_SIZE=5000 \
  IDM_LOG_INTERVAL=1000 IDM_EVAL_INTERVAL=5000 IDM_CHECKPOINT_INTERVAL=25000 \
  NUM_DIAGNOSTIC_STATES=200 DIAGNOSTIC_CANDIDATES=300 DIAGNOSTIC_TOPK=30 \
  bash "$RUNNER"
