#!/usr/bin/env bash
set -euo pipefail

# A800 node4：episode 内 staged hybrid。远距离阶段使用 K10 8-sample
# path-medoid + H2；当名义剩余距离为 25 时，切换到 final-goal H5。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}

run_setting() {
  local gpu_ids=$1
  local goal_offset=$2
  local eval_budget=$3
  local switch_steps=$4
  env GPU_IDS="$gpu_ids" \
    NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" POLICY_SEED="$POLICY_SEED" \
    NUM_SAMPLES=8 GOAL_OFFSET_STEPS="$goal_offset" EVAL_BUDGET="$eval_budget" \
    FINAL_GOAL_SWITCH_STEPS="$switch_steps" \
    bash "$BASE_SCRIPT"
}

run_setting "0 1 2 3" 50 100 25 &
pid_50=$!
run_setting "4 5 6 7" 75 150 50 &
pid_75=$!

failed=0
if ! wait "$pid_50"; then failed=1; fi
if ! wait "$pid_75"; then failed=1; fi
exit "$failed"
