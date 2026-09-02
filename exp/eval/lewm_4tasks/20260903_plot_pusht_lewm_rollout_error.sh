#!/usr/bin/env bash
set -euo pipefail

# 本地绘图：从 PushT H=100 frozen LeWM rollout diagnostic 的 summary 绘图；横轴使用 environment steps，action chunk=5，LeWM++ local horizon k=10 env steps。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SUMMARY_CSV=${SUMMARY_CSV:-$OGBENCH_ROOT/results/20260903_pusht_lewm_rollout_error_h100/summary.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$OGBENCH_ROOT/results/20260903_pusht_lewm_rollout_error_h100}
ACTION_BLOCK=${ACTION_BLOCK:-5}
LOCAL_HORIZON=${LOCAL_HORIZON:-10}

cd "$OGBENCH_ROOT/impls"
"$PYTHON_BIN" plot_pusht_lewm_rollout_error.py \
  --summary-csv "$SUMMARY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --action-block "$ACTION_BLOCK" \
  --local-horizon "$LOCAL_HORIZON"
