#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT
CONFIG_FILE=${1:?Usage: bash eval.sh configs/eval_<setting>.env}
source "$CONFIG_FILE"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$REPO_ROOT/impls"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" eval_lewm_4tasks.py "${EVAL_ARGS[@]}" --validate-only
"$PYTHON_BIN" eval_lewm_4tasks.py "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
