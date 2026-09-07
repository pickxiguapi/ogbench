#!/usr/bin/env bash
set -euo pipefail

# H75-only general LatentPathFlow evaluation for a caller-selected K. Set
# SUBGOAL_STEPS to 15 or 25; the shared launcher strictly validates the matching
# checkpoint config and evaluates seeds 0/1/42 with 50 episodes each.

: "${SUBGOAL_STEPS:?Set SUBGOAL_STEPS to 15 or 25}"
if [[ "$SUBGOAL_STEPS" != 15 && "$SUBGOAL_STEPS" != 25 ]]; then
  echo "SUBGOAL_STEPS must be 15 or 25 for this ablation." >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

LEWM_DATA_ROOT=/data-training/yyf/datasets/lewm-eval-compact-h75-seeds0-1-42-v1 \
GOAL_OFFSETS=75 \
EVAL_SEEDS="0 1 42" \
NUM_EVAL=50 \
bash "$SCRIPT_DIR/20260907_eval_node3_lewmpp_general_k15_h25_h50_3seeds.sh"
