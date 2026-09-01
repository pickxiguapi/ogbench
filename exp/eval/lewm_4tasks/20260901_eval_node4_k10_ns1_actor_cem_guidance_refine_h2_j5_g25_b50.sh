#!/usr/bin/env bash
set -euo pipefail

# A800 node4：actor-only CEM 第二轮。比较 population size，以及先用
# LeWM cost 从 actor proposals 中选中心的 select/elite 方式；Q/V 不参与。
# 协议固定为 K10 single-sample、MoH、H2、J5、300/top-30、RH1、50 ep。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_BASE=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-ns1-actor-cem-h2-j5-refine

run_variant() {
  local name=$1
  local gpu_ids=$2
  local guidance=$3
  local population_size=$4
  local temperature=$5
  local elite_size=$6
  local output_root="$EVAL_ROOT/20260901_k10_ns1_actor_cem_${name}_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}"

  env GPU_IDS="$gpu_ids" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
    POLICY_SEED="$POLICY_SEED" NUM_SAMPLES=1 GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 \
    CEM_ITERATIONS=5 POLICY_GUIDANCE="$guidance" \
    GUIDANCE_POPULATION_SIZE="$population_size" \
    GUIDANCE_TEMPERATURE="$temperature" GUIDANCE_ELITE_SIZE="$elite_size" \
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
  population32_t03 population 32 0.3 8 \
  population128_t03 population 128 0.3 8

run_pair \
  lewm_select64_t03 lewm_select 64 0.3 8 \
  lewm_elite64_t03_e8 lewm_elite 64 0.3 8
