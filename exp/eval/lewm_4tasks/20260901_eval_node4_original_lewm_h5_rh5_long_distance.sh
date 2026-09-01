#!/usr/bin/env bash
set -euo pipefail

# A800 node4：八卡并行评测完全原始的纯 LeWM 官方协议。
# Goal/Budget=50/100 与 75/150；无 policy、无 subgoal、final goal、last cost、
# CEM300x30、H5/RH5、action_block=5、50 episodes、evaluation seed42。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON_BASH="$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
eval_root=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
tmp_root=/data-training/yyf/ogbench-lewm-policy-runs/tmp/original-lewm-h5-rh5-long-distance

run_setting() {
  local gpu_ids=$1
  local goal_offset=$2
  local eval_budget=$3

  CLIENT_ID=node4 \
  CONTROLLER=lewm_cem \
  POLICY_GUIDANCE=none \
  USE_SUBGOAL=0 \
  NUM_EVAL="$NUM_EVAL" \
  EVAL_SEED="$EVAL_SEED" \
  GPU_IDS="$gpu_ids" \
  GOAL_OFFSET_STEPS="$goal_offset" \
  EVAL_BUDGET="$eval_budget" \
  CEM_HORIZON=5 \
  CEM_RECEDING_HORIZON=5 \
  CEM_NUM_SAMPLES=300 \
  CEM_ITERATIONS=30 \
  CEM_TOPK=30 \
  CEM_COST_MODE=last \
  EVAL_TAG="original_lewm_last_cem300x30_h5_rh5_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}" \
  OUTPUT_ROOT="$eval_root/20260901_original_lewm_last_cem300x30_h5_rh5_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${EVAL_SEED}" \
  EVAL_TMP_ROOT="$tmp_root/g${goal_offset}_b${eval_budget}" \
  LEWM_CUBE_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  LEWM_PUSHT_DIR=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
  LEWM_REACHER_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  LEWM_TWOROOM_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
  bash "$COMMON_BASH"
}

run_setting "0 1 2 3" 50 100 &
pid_50=$!
run_setting "4 5 6 7" 75 150 &
pid_75=$!

failed=0
if ! wait "$pid_50"; then failed=1; fi
if ! wait "$pid_75"; then failed=1; fi
exit "$failed"
