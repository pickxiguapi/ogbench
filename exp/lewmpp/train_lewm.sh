#!/usr/bin/env bash
set -euo pipefail

# 公共 LeWM 训练入口：训练正文控制器使用的 IMPALA-small、history3、frameskip5 世界模型。
# 所有训练从本仓库 Bash 发起，数据和输出路径由环境变量显式提供。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${DATASET_PATH:?Set DATASET_PATH to a JPEG-backed Lance table}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
EXP_NAME=${EXP_NAME:-lewm_impala_history3}
mkdir -p "$OUTPUT_DIR"
cd "$OGBENCH_ROOT/impls"
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$DATASET_PATH" \
    --save_dir="$OUTPUT_DIR" \
    --exp_name="$EXP_NAME" \
    --seed="${TRAIN_SEED:-3072}" \
    --epochs=10 \
    --batch_size=128 \
    --frameskip=5 \
    --image_size=224 \
    --learning_rate=5e-5 \
    --weight_decay=1e-3 \
    --sigreg_weight=0.09
