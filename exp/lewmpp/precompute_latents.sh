#!/usr/bin/env bash
set -euo pipefail

# 公共 frozen-latent 预处理入口：从训练 LeWM 时使用的 JPEG-backed Lance 数据生成 HDF5 cache。
# Cache is bound to the frozen LeWM checkpoint SHA-256 for all generator types.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${TASK:?Set TASK}"
: "${LANCE_PATH:?Set LANCE_PATH}"
: "${LEWM_CHECKPOINT:?Set LEWM_CHECKPOINT}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH}"
require_file "$LEWM_CHECKPOINT" "LeWM checkpoint"

mkdir -p "$(dirname "$OUTPUT_PATH")"
cd "$OGBENCH_ROOT/impls"
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" precompute_lewm_latents.py \
    --task="$TASK" \
    --lance-path="$LANCE_PATH" \
    --checkpoint="$LEWM_CHECKPOINT" \
    --output="$OUTPUT_PATH" \
    --batch-size="${BATCH_SIZE:-512}" \
    --decode-workers="${DECODE_WORKERS:-12}" \
    --output-dtype=float32
