#!/usr/bin/env bash
set -euo pipefail

# 英博云：顺序复现 LeWM-4Tasks 主训练矩阵；先独立 GCIQL-Chunk，再 LeWM-JAX，最后训练 pi/qv/all frozen-LeWM 共享消融。
RUN_INDEPENDENT=${RUN_INDEPENDENT:-1}
RUN_LEWM=${RUN_LEWM:-1}
RUN_SHARED=${RUN_SHARED:-1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "$RUN_INDEPENDENT" == 1 ]]; then
  REPRESENTATION_MODE=independent \
    bash "$SCRIPT_DIR/20260823_train_yb_gciql_chunk_4tasks.sh"
fi

if [[ "$RUN_LEWM" == 1 ]]; then
  bash "$SCRIPT_DIR/20260823_train_yb_lewm_4tasks.sh"
fi

if [[ "$RUN_SHARED" == 1 ]]; then
  for mode in pi qv all; do
    REPRESENTATION_MODE="$mode" \
      bash "$SCRIPT_DIR/20260823_train_yb_gciql_chunk_4tasks.sh"
  done
fi
