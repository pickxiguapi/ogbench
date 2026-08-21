#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
GOAL_OFFSET="${2:-}"
EVAL_BUDGET="${3:-}"

DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/dzb/stablewm-data/gciql-chunk-proposals-s11}"

case "${TASK}" in
  cube) DATASET="cube_single_expert.h5" ;;
  pusht) DATASET="pusht_expert_train.h5" ;;
  reacher) DATASET="reacher.h5" ;;
  tworoom) DATASET="tworoom.h5" ;;
  *) echo "Invalid task: ${TASK}" >&2; exit 2 ;;
esac
case "${GOAL_OFFSET}:${EVAL_BUDGET}" in
  50:100|75:150) ;;
  *) echo "Expected goal/budget pair 50/100 or 75/150, got ${GOAL_OFFSET}/${EVAL_BUDGET}" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${TASK}"
[[ -s "${DATA_ROOT}/${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATA_ROOT}/${DATASET}" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_100000.pkl" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT_DIR}" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags not found: ${CHECKPOINT_DIR}" >&2; exit 1; }

echo "Validated GCIQL-Chunk-AWR policy-only ${TASK}: goal=${GOAL_OFFSET}, budget=${EVAL_BUDGET}, chunk=5, step=100000."
