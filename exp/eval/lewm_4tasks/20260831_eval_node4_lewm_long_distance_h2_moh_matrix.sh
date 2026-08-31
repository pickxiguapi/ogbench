#!/usr/bin/env bash
set -euo pipefail

# A800 node4：远距离 dataset-goal 严格配对评测。
# 对每个 goal offset，同时运行 predicted K10 subgoal 与无 subgoal global goal；
# 两者均为 50 episodes、seed42、CEM300x30、H2/RH1、action block5、MoH。
# K50 使用 budget100，K75 使用 budget150。每个距离的两种方法各占四张 GPU，距离之间顺序执行。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
path_flow_root=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10
eval_root=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
tmp_root=/data-training/yyf/ogbench-lewm-policy-runs/tmp

lewm_cube=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
lewm_pusht=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666
lewm_reacher=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
lewm_tworoom=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95

run_subgoal() {
  local goal_offset=$1
  local eval_budget=$2
  CLIENT_ID=node4 \
  MODE=subgoal_lewm \
  NUM_EVAL=${NUM_EVAL:-50} \
  EVAL_SEED=${EVAL_SEED:-42} \
  GPU_IDS="0 1 2 3" \
  GOAL_OFFSET_STEPS="$goal_offset" \
  EVAL_BUDGET="$eval_budget" \
  CEM_HORIZON=2 \
  CEM_RECEDING_HORIZON=1 \
  CEM_NUM_SAMPLES=300 \
  CEM_STEPS=30 \
  CEM_TOPK=30 \
  CEM_COST_MODE=min_over_horizon \
  LATENT_SUBGOAL_STEPS=200000 \
  LATENT_SUBGOAL_REFRESH_STEPS=5 \
  LATENT_SUBGOAL_CUBE_DIR="$path_flow_root/latent_pathflow_cube_lewm3072_k5_10_cfm_n200000_b1024_s0" \
  LATENT_SUBGOAL_PUSHT_DIR="$path_flow_root/latent_pathflow_pusht_lewm666_k5_10_cfm_n200000_b1024_s0" \
  LATENT_SUBGOAL_REACHER_DIR="$path_flow_root/latent_pathflow_reacher_lewm3072_k5_10_cfm_n200000_b1024_s0" \
  LATENT_SUBGOAL_TWOROOM_DIR="$path_flow_root/latent_pathflow_tworoom_lewm3072_k5_10_cfm_n200000_b1024_s0" \
  EVAL_TAG="latent_path_flow_k10_goal${goal_offset}_budget${eval_budget}_refresh5_moh_cem300x30_h2_rh1_ep50" \
  OUTPUT_ROOT="$eval_root/20260831_latent_path_flow_k10_goal${goal_offset}_budget${eval_budget}_refresh5_moh_cem300x30_h2_rh1_ep50_seed42" \
  EVAL_TMP_ROOT="$tmp_root/lewm-latent-path-flow-k10-goal${goal_offset}-budget${eval_budget}-h2-moh-eval" \
  LEWM_CUBE_DIR="$lewm_cube" \
  LEWM_PUSHT_DIR="$lewm_pusht" \
  LEWM_REACHER_DIR="$lewm_reacher" \
  LEWM_TWOROOM_DIR="$lewm_tworoom" \
  bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
}

run_global() {
  local goal_offset=$1
  local eval_budget=$2
  CLIENT_ID=node4 \
  MODE=lewm \
  NUM_EVAL=${NUM_EVAL:-50} \
  EVAL_SEED=${EVAL_SEED:-42} \
  GPU_IDS="4 5 6 7" \
  GOAL_OFFSET_STEPS="$goal_offset" \
  EVAL_BUDGET="$eval_budget" \
  CEM_HORIZON=2 \
  CEM_RECEDING_HORIZON=1 \
  CEM_NUM_SAMPLES=300 \
  CEM_STEPS=30 \
  CEM_TOPK=30 \
  CEM_COST_MODE=min_over_horizon \
  EVAL_TAG="global_goal_goal${goal_offset}_budget${eval_budget}_moh_cem300x30_h2_rh1_ep50" \
  OUTPUT_ROOT="$eval_root/20260831_global_goal_goal${goal_offset}_budget${eval_budget}_moh_cem300x30_h2_rh1_ep50_seed42" \
  EVAL_TMP_ROOT="$tmp_root/lewm-global-goal-goal${goal_offset}-budget${eval_budget}-h2-moh-eval" \
  LEWM_CUBE_DIR="$lewm_cube" \
  LEWM_PUSHT_DIR="$lewm_pusht" \
  LEWM_REACHER_DIR="$lewm_reacher" \
  LEWM_TWOROOM_DIR="$lewm_tworoom" \
  bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
}

run_pair() {
  local goal_offset=$1
  local eval_budget=$2
  run_subgoal "$goal_offset" "$eval_budget" &
  local subgoal_pid=$!
  run_global "$goal_offset" "$eval_budget" &
  local global_pid=$!
  wait "$subgoal_pid"
  wait "$global_pid"
}

run_pair 50 100
run_pair 75 150
