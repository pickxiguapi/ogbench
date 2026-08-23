#!/usr/bin/env bash
set -euo pipefail

# Server23：四卡并行评测 seed 666 的四任务 IMPALA-Small LeWM checkpoint，每任务默认 50 episodes。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=/data/dzb/stablewm-data/lewm-jax
RUN_PREFIX=2026-08-19_23_LeWMJAX_impala_lance

CLIENT_ID=23 \
MODE=lewm \
LEWM_SEED=666 \
NUM_EVAL=${NUM_EVAL:-50} \
EVAL_SEED=${EVAL_SEED:-42} \
GPU_IDS="${GPU_IDS:-2 3 4 5}" \
EVAL_TAG=${EVAL_TAG:-seed666_cem300x5_h5_rh1_ep50_20260823} \
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/dzb/stablewm-data/lewm-jax-evals/20260823_seed666_cem300x5_h5_rh1_ep50_seed42} \
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/data/dzb/stablewm-data/tmp/lewm-seed666-eval} \
LEWM_CUBE_DIR="$RUN_ROOT/${RUN_PREFIX}_cube_single_bs128_e10_seed666" \
LEWM_PUSHT_DIR="$RUN_ROOT/${RUN_PREFIX}_pusht_expert_bs128_e10_seed666" \
LEWM_REACHER_DIR="$RUN_ROOT/${RUN_PREFIX}_reacher_bs128_e10_seed666" \
LEWM_TWOROOM_DIR="$RUN_ROOT/${RUN_PREFIX}_tworoom_bs128_e10_seed666" \
bash "$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
