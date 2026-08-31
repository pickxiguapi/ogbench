#!/usr/bin/env bash
set -euo pipefail

# A800 node4：8 卡并行正式评测 seed777 shared-all policy-guided K10
# LatentPathFlow CEM 的 50/100 与 75/150 两个长距离设置。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_guided.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}

env GPU_IDS="0 1 2 3" \
  NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" POLICY_SEED="$POLICY_SEED" \
  GOAL_OFFSET_STEPS=50 EVAL_BUDGET=100 \
  bash "$BASE_SCRIPT" &
pid_50=$!

env GPU_IDS="4 5 6 7" \
  NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" POLICY_SEED="$POLICY_SEED" \
  GOAL_OFFSET_STEPS=75 EVAL_BUDGET=150 \
  bash "$BASE_SCRIPT" &
pid_75=$!

failed=0
if ! wait "$pid_50"; then failed=1; fi
if ! wait "$pid_75"; then failed=1; fi
exit "$failed"
