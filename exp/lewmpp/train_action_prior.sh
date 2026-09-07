#!/usr/bin/env bash
set -euo pipefail

# 公共 action prior 训练入口：训练正文使用的 final-goal-conditioned GCIQL-Chunk-AWR。
# 冻结 LeWM 编码器，并统一使用 shared-all 表征、chunk5、seed777 和无图像增强配置。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${TASK:?Set TASK}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${LEWM_CHECKPOINT:?Set LEWM_CHECKPOINT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
require_file "$LEWM_CHECKPOINT" "LeWM checkpoint"

TRAIN_SEED=${TRAIN_SEED:-777}
run_dir="$OUTPUT_ROOT/$TASK/seed${TRAIN_SEED}"
mkdir -p "$run_dir"
cd "$OGBENCH_ROOT/impls"
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" train_action_prior.py \
    --dataset_path="$DATASET_PATH" \
    --lewm_checkpoint="$LEWM_CHECKPOINT" \
    --save_dir="$run_dir" \
    --train_steps=100000 \
    --save_interval=100000 \
    --log_interval=5000 \
    --batch_size=256 \
    --seed="$TRAIN_SEED" \
    --lr=3e-4 \
    --discount=0.99 \
    --expectile=0.9 \
    --tau=0.005 \
    --chunk_size=5 \
    --alpha=3.0 \
    --p_aug=0.0
