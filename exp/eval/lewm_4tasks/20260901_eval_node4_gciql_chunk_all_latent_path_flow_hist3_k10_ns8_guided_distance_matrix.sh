#!/usr/bin/env bash
set -euo pipefail

# A800 node4：正式评测 seed777 policy-guided K10 LatentPathFlow 的
# 8-sample path-medoid inference。先并行跑 25/50 与 50/100，再跑 75/150。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}

run_setting() {
  local gpu_ids=$1
  local goal_offset=$2
  local eval_budget=$3
  env GPU_IDS="$gpu_ids" \
    NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" POLICY_SEED="$POLICY_SEED" \
    NUM_SAMPLES=8 GOAL_OFFSET_STEPS="$goal_offset" EVAL_BUDGET="$eval_budget" \
    bash "$BASE_SCRIPT"
}

run_setting "0 1 2 3" 25 50 &
pid_25=$!
run_setting "4 5 6 7" 50 100 &
pid_50=$!

failed=0
if ! wait "$pid_25"; then failed=1; fi
if ! wait "$pid_50"; then failed=1; fi
if (( failed )); then exit "$failed"; fi

run_setting "0 1 2 3" 75 150
