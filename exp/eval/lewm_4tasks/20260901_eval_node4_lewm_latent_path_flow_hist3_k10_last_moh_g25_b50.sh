#!/usr/bin/env bash
set -euo pipefail

# A800 node4 八卡并行评测三帧历史 LatentPathFlow K10 subgoal + LeWM CEM：
# goal/budget=25/50，GPU0-3 跑 last，GPU4-7 跑 MoH；自动 H2、RH1、
# 单样本 subgoal、CEM300x30、50 episodes、evaluation seed42。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON_BASH="$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
path_flow_root=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10
eval_root=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
tmp_root=/data-training/yyf/ogbench-lewm-policy-runs/tmp

run_cost() {
  local cost_mode=$1
  local gpu_ids=$2

  CLIENT_ID=node4 \
  CONTROLLER=lewm_cem \
  POLICY_GUIDANCE=none \
  USE_SUBGOAL=1 \
  NUM_EVAL="$NUM_EVAL" \
  EVAL_SEED="$EVAL_SEED" \
  GPU_IDS="$gpu_ids" \
  GOAL_OFFSET_STEPS=25 \
  EVAL_BUDGET=50 \
  CEM_RECEDING_HORIZON=1 \
  CEM_NUM_SAMPLES=300 \
  CEM_ITERATIONS=30 \
  CEM_TOPK=30 \
  CEM_COST_MODE="$cost_mode" \
  LATENT_SUBGOAL_STEPS=200000 \
  LATENT_SUBGOAL_CUBE_DIR="$path_flow_root/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
  LATENT_SUBGOAL_PUSHT_DIR="$path_flow_root/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
  LATENT_SUBGOAL_REACHER_DIR="$path_flow_root/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
  LATENT_SUBGOAL_TWOROOM_DIR="$path_flow_root/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
  EVAL_TAG="latest_latent_path_flow_hist3_k10_n200k_ns1_${cost_mode}_cem300x30_h2_rh1_g25_b50_ep${NUM_EVAL}" \
  OUTPUT_ROOT="$eval_root/20260901_latest_latent_path_flow_hist3_k10_ns1_${cost_mode}_cem300x30_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}" \
  EVAL_TMP_ROOT="$tmp_root/lewm-latent-path-flow-hist3-k10-ns1-${cost_mode}-g25-b50-eval" \
  LEWM_CUBE_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  LEWM_PUSHT_DIR=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
  LEWM_REACHER_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  LEWM_TWOROOM_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  bash "$COMMON_BASH"
}

run_cost last "0 1 2 3" &
last_pid=$!
run_cost moh "4 5 6 7" &
moh_pid=$!

failed=0
if ! wait "$last_pid"; then failed=1; fi
if ! wait "$moh_pid"; then failed=1; fi
exit "$failed"
