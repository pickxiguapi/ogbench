#!/usr/bin/env bash
set -euo pipefail

# A800 node4：四卡并行评测 200k LatentPathFlow，只取 K10 token；每次 RH1 replan（5步）重新预测 K10。
# CEM H2 rollout 在 t+5/t+10 上对 predicted K10 使用 min-over-horizon；其余与 refresh5 H2 terminal 严格配对：
# dataset K25 条件 goal、50 episodes、seed42、budget50、CEM300x30、RH1、topk30，不使用 K5 token。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
path_flow_root=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10

CLIENT_ID=node4 \
MODE=subgoal_lewm \
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
LATENT_SUBGOAL_STEPS=200000 \
LATENT_SUBGOAL_REFRESH_STEPS=5 \
LATENT_SUBGOAL_CUBE_DIR="$path_flow_root/latent_pathflow_cube_lewm3072_k5_10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_PUSHT_DIR="$path_flow_root/latent_pathflow_pusht_lewm666_k5_10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_REACHER_DIR="$path_flow_root/latent_pathflow_reacher_lewm3072_k5_10_cfm_n200000_b1024_s0" \
LATENT_SUBGOAL_TWOROOM_DIR="$path_flow_root/latent_pathflow_tworoom_lewm3072_k5_10_cfm_n200000_b1024_s0" \
EVAL_TAG=${EVAL_TAG:-latent_path_flow_k10_only_n200k_refresh5_moh_mixed_cube3072_pusht666_reacher3072_tworoom3072_cem300x30_h2_rh1_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_latent_path_flow_k10_only_refresh5_moh_cem300x30_h2_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/lewm-latent-path-flow-k10-refresh5-h2-moh-eval} \
LEWM_CUBE_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
