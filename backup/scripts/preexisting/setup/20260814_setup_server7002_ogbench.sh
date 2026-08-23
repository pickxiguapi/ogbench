#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
UV_BIN="${UV_BIN:-/home/yyf/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/yyf/.cache/uv}"
DATA_DIR="${DATA_DIR:-/mnt/18T/yyf/ogbench-data}"
DEFAULT_DATA_LINK="${HOME}/.ogbench/data"
VENV_DIR="${VENV_DIR:-${OGBENCH_ROOT}/.venv-s23}"
PYTHON_RUNTIME="${PYTHON_RUNTIME:-/mnt/18T/yyf/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}"

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
if [[ -e "${VENV_DIR}/bin/python" && ! -x "${VENV_DIR}/bin/python" && -x "$PYTHON_RUNTIME" ]]; then
  ln -sfn "$PYTHON_RUNTIME" "${VENV_DIR}/bin/python"
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    "$UV_BIN" sync --extra train --frozen --offline --python "$PYTHON_BIN"
fi

for name in visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0; do
  [[ -s "${DATA_DIR}/${name}.npz" ]] || { echo "ERROR: dataset missing: ${DATA_DIR}/${name}.npz" >&2; exit 1; }
  [[ -s "${DATA_DIR}/${name}-val.npz" ]] || { echo "ERROR: dataset missing: ${DATA_DIR}/${name}-val.npz" >&2; exit 1; }
done

MUJOCO_GL=egl PYTHONPATH="$OGBENCH_ROOT" "${VENV_DIR}/bin/python" - <<'PY'
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
