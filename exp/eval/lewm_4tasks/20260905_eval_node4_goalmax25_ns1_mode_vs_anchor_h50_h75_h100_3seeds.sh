#!/usr/bin/env bash
set -euo pipefail

# Compare policy mode against mode-anchor at longer goal offsets.  Keep the
# selected goalmax25 LatentPathFlow ns=1 stack fixed and use a 2H rollout
# budget for every evaluation horizon.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260905_eval_node4_goalmax25_ns1_policy_combinations_seed42.sh"

GOAL_OFFSETS=${GOAL_OFFSETS:-"50 75 100"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
RUN_VARIANTS=${RUN_VARIANTS:-"policy_mode policy_mode_anchor"}
GPU_IDS=${GPU_IDS:-"4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

read -r -a goal_offsets <<< "$GOAL_OFFSETS"
read -r -a eval_seeds <<< "$EVAL_SEEDS"
for goal_offset in "${goal_offsets[@]}"; do
  eval_budget=$((goal_offset * 2))
  for eval_seed in "${eval_seeds[@]}"; do
    echo "START goal_offset=$goal_offset eval_budget=$eval_budget eval_seed=$eval_seed variants=[$RUN_VARIANTS]"
    GOAL_OFFSET_STEPS="$goal_offset" \
    EVAL_BUDGET="$eval_budget" \
    EVAL_SEED="$eval_seed" \
    RUN_VARIANTS="$RUN_VARIANTS" \
    GPU_IDS="$GPU_IDS" \
    NUM_EVAL="$NUM_EVAL" \
    POLICY_SEED="$POLICY_SEED" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
      bash "$BASE_SCRIPT"
    echo "DONE goal_offset=$goal_offset eval_seed=$eval_seed"
  done
done
