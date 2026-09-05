#!/usr/bin/env bash
set -euo pipefail

# H25 single-variable ablation of the canonical LeWM++ result:
# replace min-over-horizon (moh) with terminal/last-state cost.  Everything
# else stays fixed: H25 goalmax25 generator, Policy mode, policy seed777,
# final-goal policy conditioning, FlowPath ns1, H2/RH1/J5, CEM300x5,
# 50 episodes, and evaluation seeds 0/1/666.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260905_eval_node4_goalmax25_ns1_policy_combinations_seed42.sh"

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 666"}
GPU_IDS=${GPU_IDS:-"4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_BASE=${TMP_BASE:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-goalmax25-ns1-policy-mode-terminal}

read -r -a eval_seeds <<< "$EVAL_SEEDS"
for eval_seed in "${eval_seeds[@]}"; do
  output_root="$EVAL_ROOT/20260906_goalmax25_ns1_policy_mode_terminal_sd777_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${eval_seed}"
  tmp_root="$TMP_BASE/ep${NUM_EVAL}/seed${eval_seed}"
  echo "RUN LeWM++ w/o MoH (terminal): seed=$eval_seed H=25"
  EVAL_SEED="$eval_seed" \
  NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="$GPU_IDS" \
  RUN_VARIANTS=policy_mode \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  GOAL_OFFSET_STEPS=25 \
  EVAL_BUDGET=50 \
  CEM_COST_MODE=last \
  OUTPUT_ROOT="$output_root" \
  TMP_ROOT="$tmp_root" \
  bash "$BASE_SCRIPT"
done
