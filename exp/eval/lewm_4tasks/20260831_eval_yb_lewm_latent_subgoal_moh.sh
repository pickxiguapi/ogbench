#!/usr/bin/env bash
set -euo pipefail

# 英博云：纯 CEM 使用 frozen latent-GCBC K10 predicted subgoal 作为规划目标。
# 与 20260830 MoH 基线严格配对：混合 LeWM checkpoint、50 episodes、seed42、
# goal offset25、budget50、CEM300x30、H5/RH1、topk30、min-over-horizon。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CLIENT_ID=yb \
MODE=subgoal_lewm \
NUM_EVAL=${NUM_EVAL:-50} \
EVAL_SEED=${EVAL_SEED:-42} \
GPU_IDS="${GPU_IDS:-0 1 2 3}" \
GOAL_OFFSET_STEPS=25 \
EVAL_BUDGET=50 \
CEM_HORIZON=5 \
CEM_RECEDING_HORIZON=1 \
CEM_NUM_SAMPLES=300 \
CEM_STEPS=30 \
CEM_TOPK=30 \
CEM_COST_MODE=min_over_horizon \
LATENT_SUBGOAL_STEPS=100000 \
LATENT_SUBGOAL_REFRESH_STEPS=10 \
EVAL_TAG=${EVAL_TAG:-latent_subgoal_k10_mixed_cube3072_pusht666_reacher3072_tworoom3072_cem300x30_h5_rh1_moh_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/data/yyf/lewm-final/evals/lewm-4tasks/20260831_latent_subgoal_k10_moh_cem300x30_h5_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/root/data/yyf/tmp/lewm-latent-subgoal-eval} \
LEWM_CUBE_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/root/data/yyf/lewm-jax-seed666-s23/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
