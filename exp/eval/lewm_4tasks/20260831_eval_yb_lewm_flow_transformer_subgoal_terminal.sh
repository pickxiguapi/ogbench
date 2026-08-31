#!/usr/bin/env bash
set -euo pipefail

# 英博云：不用 MoH，纯 CEM 以 H5 terminal cost 追踪 200k Transformer-CFM K10 subgoal。
# 除 cost mode 改为 terminal 外，与 Flow-MoH 严格配对：mixed LeWM、50 episodes、
# seed42、goal offset25、budget50、CEM300x30、H5/RH1、topk30。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
flow_root=/root/data/yyf/lewm-final/latent-subgoal-flow-transformer-k10

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
CEM_COST_MODE=terminal \
LATENT_SUBGOAL_STEPS=200000 \
LATENT_SUBGOAL_REFRESH_STEPS=10 \
LATENT_SUBGOAL_CUBE_DIR="$flow_root/latent_flowtf_cube_lewm3072_k10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_PUSHT_DIR="$flow_root/latent_flowtf_pusht_lewm666_k10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_REACHER_DIR="$flow_root/latent_flowtf_reacher_lewm3072_k10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_TWOROOM_DIR="$flow_root/latent_flowtf_tworoom_lewm3072_k10_cfm_n200000_b1024_s0" \
EVAL_TAG=${EVAL_TAG:-flow_transformer_subgoal_k10_n200k_terminal_mixed_cube3072_pusht666_reacher3072_tworoom3072_cem300x30_h5_rh1_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/data/yyf/lewm-final/evals/lewm-4tasks/20260831_flow_transformer_subgoal_k10_terminal_cem300x30_h5_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/root/data/yyf/tmp/lewm-flow-transformer-subgoal-terminal-eval} \
LEWM_CUBE_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/root/data/yyf/lewm-jax-seed666-s23/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
