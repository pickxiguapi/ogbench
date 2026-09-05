#!/usr/bin/env bash
set -euo pipefail

# Run the four primary policy-combination variants with paired evaluation seeds.
# The underlying launcher fixes goalmax25 LatentPathFlow ns=1, policy seed777,
# final-goal policy conditioning, MoH, H2/RH1, and CEM300x5.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260905_eval_node4_goalmax25_ns1_policy_combinations_seed42.sh"

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
RUN_VARIANTS=${RUN_VARIANTS:-"zero_init policy_mode policy_mode_anchor policy_population64_t03"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

read -r -a eval_seeds <<< "$EVAL_SEEDS"
for eval_seed in "${eval_seeds[@]}"; do
  echo "START eval_seed=$eval_seed variants=[$RUN_VARIANTS]"
  EVAL_SEED="$eval_seed" \
  RUN_VARIANTS="$RUN_VARIANTS" \
  GPU_IDS="$GPU_IDS" \
  NUM_EVAL="$NUM_EVAL" \
  POLICY_SEED="$POLICY_SEED" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
    bash "$BASE_SCRIPT"
  echo "DONE eval_seed=$eval_seed"
done
