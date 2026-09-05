#!/usr/bin/env bash
set -euo pipefail

# Decoder-only reconstruction probes for the canonical mixed LeWM mapping:
# PushT seed666; Cube/Reacher/TwoRoom seed3072.  Subgoal models are unrelated
# to this training job and are deliberately not loaded here.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data-training/yyf/envs/latent-geometry/bin/python}
DATA_ROOT=${DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
LATENT_ROOT=${LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
RUN_ROOT=${RUN_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder}
DECODER_TYPE=${DECODER_TYPE:-conv}
RUN_NAME=${RUN_NAME:-20260905_mixed666_3072_official_cnn_image_decoder}
MODE=${MODE:-launch}
EPOCHS=${EPOCHS:-50}
TRAIN_ROWS=${TRAIN_ROWS:-200000}
VAL_ROWS=${VAL_ROWS:-20000}
BATCH_SIZE=${BATCH_SIZE:-128}
DECODE_WORKERS=${DECODE_WORKERS:-12}
mkdir -p "$RUN_ROOT/$RUN_NAME/logs"

run_task() {
  local task=$1 gpu=$2 latent lance
  case "$task" in
    pusht)
      latent="$LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
      lance="$DATA_ROOT/pusht_expert_train.lance"
      ;;
    cube)
      latent="$LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/cube_single_expert.lance"
      ;;
    reacher)
      latent="$LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/reacher.lance"
      ;;
    tworoom)
      latent="$LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/tworoom.lance"
      ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
  test -s "$latent"
  test -d "$lance"
  cd "$OGBENCH_ROOT/impls"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train_lewm_visual_decoder.py \
    --task="$task" --latent-hdf5="$latent" --lance-path="$lance" \
    --output-dir="$RUN_ROOT/$RUN_NAME/$task" --epochs="$EPOCHS" \
    --train-rows="$TRAIN_ROWS" --val-rows="$VAL_ROWS" \
    --batch-size="$BATCH_SIZE" --decode-workers="$DECODE_WORKERS" \
    --decoder-type="$DECODER_TYPE" \
    --seed=3072 ${SMOKE_BATCHES:+--smoke-batches="$SMOKE_BATCHES"} \
    2>&1 | tee -a "$RUN_ROOT/$RUN_NAME/logs/$task.log"
}

case "$MODE" in
  smoke)
    SMOKE_BATCHES=${SMOKE_BATCHES:-2}
    EPOCHS=1 TRAIN_ROWS=512 VAL_ROWS=256 run_task "${TASK:-cube}" "${GPU_ID:-0}"
    ;;
  worker)
    run_task "${TASK:?TASK is required}" "${GPU_ID:?GPU_ID is required}"
    ;;
  launch)
    for spec in tworoom:0 pusht:1 cube:2 reacher:3; do
      task=${spec%%:*}; gpu=${spec##*:}; session="lewm-visdec-$task"
      tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 3; }
      tmux new-session -d -s "$session" \
        "cd '$OGBENCH_ROOT' && MODE=worker TASK='$task' GPU_ID='$gpu' RUN_NAME='$RUN_NAME' bash exp/train/20260904_train_node4_lewm_visual_decoder_4tasks.sh"
    done
    echo "launched $RUN_NAME on GPUs 0,1,2,3"
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep 'lewm-visdec-' || true
    for log in "$RUN_ROOT/$RUN_NAME"/logs/*.log; do
      [[ -f "$log" ]] || continue
      echo "===== $log ====="
      tail -n 5 "$log"
    done
    ;;
  *) echo "MODE must be smoke, worker, launch, or status" >&2; exit 2 ;;
esac
