#!/usr/bin/env bash
set -euo pipefail

# A800 node4：八卡并行评测 K10 LatentPathFlow subgoal + shared-all AWR
# seed777 Policy Mode。测试 Goal/Budget=50/100 与 75/150；两组均使用
# single sample、MoH、自动 H2、J5、300/top-30、action_block=5、RH1、
# 50 episodes、evaluation seed42，Q/V 不参与。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-ns1-actor-mode-moh-h2-j5-long-distance

run_setting() {
  local gpu_ids=$1
  local goal_offset=$2
  local eval_budget=$3
  local output_root="$EVAL_ROOT/20260901_k10_ns1_actor_cem_mode_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${EVAL_SEED}"

  env GPU_IDS="$gpu_ids" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
    POLICY_SEED="$POLICY_SEED" NUM_SAMPLES=1 \
    GOAL_OFFSET_STEPS="$goal_offset" EVAL_BUDGET="$eval_budget" \
    CEM_ITERATIONS=5 CEM_COST_MODE=moh POLICY_GUIDANCE=mode \
    GUIDANCE_POPULATION_SIZE=0 GUIDANCE_TEMPERATURE=1.0 \
    OUTPUT_ROOT="$output_root" TMP_ROOT="$TMP_ROOT/g${goal_offset}_b${eval_budget}" \
    bash "$BASE_SCRIPT"
}

run_setting "0 1 2 3" 50 100 &
pid_50=$!
run_setting "4 5 6 7" 75 150 &
pid_75=$!

failed=0
if ! wait "$pid_50"; then failed=1; fi
if ! wait "$pid_75"; then failed=1; fi
exit "$failed"
