#!/usr/bin/env bash
set -euo pipefail

# A800 node4：四卡并行评测 K10 LatentPathFlow 的时间对齐 path-mean cost。
# CEM rollout K5/K10 分别与 predicted K5/K10 比较后取均值；shared-all
# AWR seed777 actor mode 仍以 terminal K10 为 goal。H2、J5、300/top-30、
# action_block=5、RH1、50 episodes、evaluation seed42，Q/V 不参与。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVAL_ROOT/20260901_k10_ns1_actor_cem_mode_sd${POLICY_SEED}_path_mean_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-ns1-actor-mode-path-mean-h2-j5}

env GPU_IDS="0 1 2 3" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
  POLICY_SEED="$POLICY_SEED" NUM_SAMPLES=1 \
  GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 \
  CEM_ITERATIONS=5 CEM_COST_MODE=path_mean POLICY_GUIDANCE=mode \
  GUIDANCE_POPULATION_SIZE=0 GUIDANCE_TEMPERATURE=1.0 \
  OUTPUT_ROOT="$OUTPUT_ROOT" TMP_ROOT="$TMP_ROOT" \
  bash "$BASE_SCRIPT"
