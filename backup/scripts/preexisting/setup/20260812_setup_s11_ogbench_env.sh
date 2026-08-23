#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
VENV_DIR="${VENV_DIR:-/data/yyf/H-LeWM/envs/ogbench}"
UV_BIN="${UV_BIN:-/home/yyf/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || true)}"

[[ -x "${UV_BIN}" ]] || { echo "ERROR: uv not found: ${UV_BIN}" >&2; exit 1; }
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python 3.11 not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/pyproject.toml" ]] || { echo "ERROR: OGBench checkout not found: ${OGBENCH_ROOT}" >&2; exit 1; }

cd "${OGBENCH_ROOT}"
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
  "${UV_BIN}" sync \
  --extra train \
  --frozen \
  --no-install-project \
  --python "${PYTHON_BIN}"

PYTHONPATH="${OGBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${VENV_DIR}/bin/python" - <<'PY'
import distrax
import flax
import jax
import lancedb
import ogbench
import wandb

print('OGBench environment ready')
print('jax', jax.__version__)
print('flax', flax.__version__)
print('lancedb', lancedb.__version__)
PY
