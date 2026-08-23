#!/usr/bin/env bash
set -euo pipefail

# A800 node2：复现 OGBench-Env-8Tasks 主评测矩阵；LeWM-only 跑一次，四种表征分别跑 policy、guided 和 native-Q。
RUN_POLICY=${RUN_POLICY:-1}
RUN_LEWM=${RUN_LEWM:-1}
RUN_GUIDED=${RUN_GUIDED:-1}
RUN_NATIVE_Q=${RUN_NATIVE_Q:-1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXECUTOR="$SCRIPT_DIR/20260823_eval_node2_ogbench_env_8tasks.sh"

if [[ "$RUN_LEWM" == 1 ]]; then
  MODE=lewm REPRESENTATION_MODE=independent bash "$EXECUTOR"
fi

for representation in independent pi qv all; do
  if [[ "$RUN_POLICY" == 1 ]]; then
    MODE=policy REPRESENTATION_MODE="$representation" bash "$EXECUTOR"
  fi
  if [[ "$RUN_GUIDED" == 1 ]]; then
    MODE=guided REPRESENTATION_MODE="$representation" bash "$EXECUTOR"
  fi
  if [[ "$RUN_NATIVE_Q" == 1 ]]; then
    MODE=native_q REPRESENTATION_MODE="$representation" bash "$EXECUTOR"
  fi
done
