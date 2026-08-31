#!/usr/bin/env bash
set -euo pipefail

# A800 node4 八卡并行评测原始 global-goal LeWM CEM：goal/budget=25/50，
# GPU0-3 跑 last，GPU4-7 跑 MoH；H5/RH5、CEM300x30、
# 50 episodes、evaluation seed42，不加载 policy 或 subgoal predictor。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON_BASH="$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
eval_root=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
tmp_root=/data-training/yyf/ogbench-lewm-policy-runs/tmp

run_cost() {
  local cost_mode=$1
  local gpu_ids=$2

  CLIENT_ID=node4 \
  CONTROLLER=lewm_cem \
  POLICY_GUIDANCE=none \
  USE_SUBGOAL=0 \
  NUM_EVAL="$NUM_EVAL" \
  EVAL_SEED="$EVAL_SEED" \
  GPU_IDS="$gpu_ids" \
  GOAL_OFFSET_STEPS=25 \
  EVAL_BUDGET=50 \
  CEM_HORIZON=5 \
  CEM_RECEDING_HORIZON=5 \
  CEM_NUM_SAMPLES=300 \
  CEM_ITERATIONS=30 \
  CEM_TOPK=30 \
  CEM_COST_MODE="$cost_mode" \
  EVAL_TAG="latest_global_goal_${cost_mode}_cem300x30_h5_rh5_g25_b50_ep${NUM_EVAL}" \
  OUTPUT_ROOT="$eval_root/20260901_latest_global_goal_${cost_mode}_cem300x30_h5_rh5_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}" \
  EVAL_TMP_ROOT="$tmp_root/lewm-global-goal-${cost_mode}-h5-rh5-g25-b50-eval" \
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
