#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/impls${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"${PYTHON_BIN}" -m unittest \
  impls.tests.test_chunk_utils \
  impls.tests.test_gciql_chunk \
  -v
