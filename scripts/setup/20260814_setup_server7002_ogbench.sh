#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
UV_BIN="${UV_BIN:-/home/yyf/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/yyf/.cache/uv}"
DATA_DIR="${DATA_DIR:-/mnt/18T/yyf/ogbench-data}"
DEFAULT_DATA_LINK="${HOME}/.ogbench/data"

[[ -x "$UV_BIN" ]] || { echo "ERROR: uv not found: ${UV_BIN}" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/pyproject.toml" ]] || { echo "ERROR: OGBench checkout not found: ${OGBENCH_ROOT}" >&2; exit 1; }

mkdir -p "$UV_CACHE_DIR" "$DATA_DIR" "$(dirname "$DEFAULT_DATA_LINK")"
if [[ -L "$DEFAULT_DATA_LINK" ]]; then
  [[ "$(readlink "$DEFAULT_DATA_LINK")" == "$DATA_DIR" ]] || {
    echo "ERROR: existing data symlink points elsewhere: ${DEFAULT_DATA_LINK}" >&2
    exit 1
  }
elif [[ -e "$DEFAULT_DATA_LINK" ]]; then
  echo "ERROR: existing data path is not the expected symlink: ${DEFAULT_DATA_LINK}" >&2
  exit 1
else
  ln -s "$DATA_DIR" "$DEFAULT_DATA_LINK"
fi

cd "$OGBENCH_ROOT"
UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="${OGBENCH_ROOT}/.venv" \
  "$UV_BIN" sync --extra train --frozen --offline --python "$PYTHON_BIN"

"${OGBENCH_ROOT}/.venv/bin/python" - <<'PY'
from ogbench import download_datasets

download_datasets([
    'visual-cube-single-noisy-v0',
    'visual-cube-double-noisy-v0',
    'visual-cube-triple-noisy-v0',
])
PY

MUJOCO_GL=egl PYTHONPATH="$OGBENCH_ROOT" "${OGBENCH_ROOT}/.venv/bin/python" - <<'PY'
import jax
import ogbench
import gymnasium

env = gymnasium.make('visual-cube-single-v0')
env.reset(seed=0)
frame = env.render()
env.close()
assert frame is not None
print('OGBench environment ready')
print('jax devices:', jax.devices())
print('render shape:', frame.shape)
PY
