#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行评测纯 LeWM；PushT 使用 training seed 666，Cube、Reacher、TwoRoom 使用 training seed 3072。
# 与官方 CEM300x30、H5/RH5 对照严格匹配，仅将 receding horizon 改为 1；其余设置保持不变。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CLIENT_ID=yb \
MODE=lewm \
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
EVAL_TAG=${EVAL_TAG:-mixed_cube3072_pusht666_reacher3072_tworoom3072_cem300x30_h5_rh1_ep50} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/data/yyf/lewm-final/evals/lewm-4tasks/20260830_mixed_ckpts_cem300x30_h5_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/root/data/yyf/tmp/lewm-mixed-rh1-eval} \
LEWM_CUBE_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_PUSHT_DIR=/root/data/yyf/lewm-jax-seed666-s23/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666 \
LEWM_REACHER_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
LEWM_TWOROOM_DIR=/root/data/yyf/lewm-jax-seed3072-s23/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95 \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
