#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/eval_lewm.sh <cube|pusht> <gciql|hiql> <checkpoint_dir>" >&2
  exit 2
fi

TASK=$1
METHOD=$2
CHECKPOINT_DIR=$3

case "$TASK" in
  cube|pusht) ;;
  *) echo "Unknown task: $TASK (expected cube or pusht)" >&2; exit 2 ;;
esac

case "$METHOD" in
  gciql|hiql) ;;
  *) echo "Unknown method: $METHOD (expected gciql or hiql)" >&2; exit 2 ;;
esac

CHECKPOINT_STEP=100000
OUTPUT_DIR="$CHECKPOINT_DIR/eval_ff"

mkdir -p "$OUTPUT_DIR/videos"
cd /home/dzb/ogbench/impls

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONPATH=/home/dzb/stable-worldmodel:/home/dzb/ogbench/impls

/home/dzb/stable-worldmodel/.venv/bin/python eval_lewm.py \
  --task "$TASK" \
  --method "$METHOD" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-step "$CHECKPOINT_STEP" \
  --stable-wm-root /home/dzb/stable-worldmodel \
  --ogbench-root /home/dzb/ogbench \
  --num-eval 50 \
  --seed 42 \
  --goal-offset-steps 25 \
  --eval-budget 50 \
  --video-dir "$OUTPUT_DIR/videos" \
  --output "$OUTPUT_DIR/${TASK}_${METHOD}_step${CHECKPOINT_STEP}.json"
