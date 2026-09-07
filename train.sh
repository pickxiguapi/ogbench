#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT
CONFIG_FILE=${1:?Usage: bash train.sh configs/train_<component>.env}
source "$CONFIG_FILE"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$REPO_ROOT/impls"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$TRAIN_ENTRYPOINT" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
