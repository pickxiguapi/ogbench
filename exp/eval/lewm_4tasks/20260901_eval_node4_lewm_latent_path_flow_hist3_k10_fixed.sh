#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4–7：评测新训练的 3 帧历史 K5/K10 LatentPathFlow。
# Generator 条件为数据集 K25 final goal，每次 RH1（5 步）重算一次 K10 subgoal；
# CEM 只 rollout 到 K10，并在固定 K10 checkpoint 打分，不使用 MoH。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
path_flow_root=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10

CLIENT_ID=node4 \
MODE=subgoal_lewm \
NUM_EVAL=${NUM_EVAL:-50} \
EVAL_SEED=${EVAL_SEED:-42} \
GPU_IDS="${GPU_IDS:-4 5 6 7}" \
GOAL_OFFSET_STEPS=25 \
EVAL_BUDGET=50 \
CEM_HORIZON=2 \
CEM_RECEDING_HORIZON=1 \
CEM_NUM_SAMPLES=300 \
CEM_STEPS=30 \
CEM_TOPK=30 \
CEM_COST_MODE=fixed_subgoal_horizon \
LATENT_SUBGOAL_STEPS=200000 \
LATENT_SUBGOAL_REFRESH_STEPS=5 \
NUM_SAMPLES=8 \
LATENT_SUBGOAL_CUBE_DIR="$path_flow_root/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
LATENT_SUBGOAL_PUSHT_DIR="$path_flow_root/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
LATENT_SUBGOAL_REACHER_DIR="$path_flow_root/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
LATENT_SUBGOAL_TWOROOM_DIR="$path_flow_root/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0" \
EVAL_TAG=${EVAL_TAG:-latent_path_flow_hist3_k10_n200k_ns8_refresh5_fixedk10_cem300x30_h2_rh1_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260901_latent_path_flow_hist3_k10_fixed_cem300x30_h2_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/lewm-latent-path-flow-hist3-k10-fixed-eval} \
LEWM_CUBE_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
