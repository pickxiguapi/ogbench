#!/usr/bin/env bash
set -euo pipefail

# A800 node4：只使用 actor 与 LeWM cost，Q/V 均不参与。统一协议：
# K10 single-sample、MoH、CEM J=5、300 samples、top-30、H2、
# action_block=5、RH1、50 episodes、evaluation seed 42。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_BASE=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-ns1-actor-cem-h2-j5-ablation

run_variant() {
  local name=$1
  local gpu_ids=$2
  local guidance=$3
  local population_size=$4
  local temperature=$5
  local first_block_std=$6
  local output_root="$EVAL_ROOT/20260901_k10_ns1_actor_cem_${name}_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}"

  env GPU_IDS="$gpu_ids" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
    POLICY_SEED="$POLICY_SEED" NUM_SAMPLES=1 GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 \
    CEM_ITERATIONS=5 POLICY_GUIDANCE="$guidance" \
    GUIDANCE_POPULATION_SIZE="$population_size" \
    GUIDANCE_TEMPERATURE="$temperature" \
    GUIDANCE_FIRST_BLOCK_STD="$first_block_std" \
    OUTPUT_ROOT="$output_root" TMP_ROOT="$TMP_BASE/$name" \
    bash "$BASE_SCRIPT"
}

run_pair() {
  local -a pids=()
  run_variant "$1" "0 1 2 3" "$2" "$3" "$4" "$5" &
  pids+=("$!")
  run_variant "$6" "4 5 6 7" "$7" "$8" "$9" "${10}" &
  pids+=("$!")
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

run_pair \
  mode mode 0 1.0 "" \
  mode_anchor mode_anchor 0 1.0 ""

run_pair \
  mode_std03 mode 0 1.0 0.3 \
  population64_t03 population 64 0.3 ""

run_variant population64_t10 "0 1 2 3" population 64 1.0 ""
