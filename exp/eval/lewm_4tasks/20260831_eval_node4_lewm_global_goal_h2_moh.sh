#!/usr/bin/env bash
set -euo pipefail

# A800 node4：四卡并行评测无 subgoal 的 K25 global-goal CEM，作为 K10-subgoal H2 MoH 的严格规划长度对照。
# mixed LeWM、50 episodes、seed42、goal offset25、budget50、CEM300x30、H2/RH1、topk30、min-over-horizon。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CLIENT_ID=node4 \
MODE=lewm \
NUM_EVAL=${NUM_EVAL:-50} \
EVAL_SEED=${EVAL_SEED:-42} \
GPU_IDS="${GPU_IDS:-0 1 2 3}" \
GOAL_OFFSET_STEPS=25 \
EVAL_BUDGET=50 \
CEM_HORIZON=2 \
CEM_RECEDING_HORIZON=1 \
CEM_NUM_SAMPLES=300 \
CEM_STEPS=30 \
CEM_TOPK=30 \
CEM_COST_MODE=min_over_horizon \
EVAL_TAG=${EVAL_TAG:-global_goal_mixed_cube3072_pusht666_reacher3072_tworoom3072_cem300x30_h2_rh1_moh_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_global_goal_moh_cem300x30_h2_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/lewm-global-goal-h2-moh-eval} \
LEWM_CUBE_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
