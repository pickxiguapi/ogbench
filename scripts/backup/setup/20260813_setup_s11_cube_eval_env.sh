#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-/home/yyf/.local/bin/uv}"
OGBENCH_PYTHON="${OGBENCH_PYTHON:-/data/yyf/H-LeWM/envs/ogbench/bin/python}"
OGBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$(cd "$(dirname "${OGBENCH_PYTHON}")/.." && pwd)"

[[ -x "${UV_BIN}" ]] || { echo "ERROR: uv not found" >&2; exit 1; }
[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }

UV_PROJECT_ENVIRONMENT="${VENV_DIR}" "${UV_BIN}" sync \
  --project "${OGBENCH_ROOT}" \
  --python "${OGBENCH_PYTHON}" \
  --extra train \
  --frozen

"${OGBENCH_PYTHON}" - <<'PY'
import cv2
import pygame
import pymunk

print('pygame', pygame.__version__)
print('pymunk', pymunk.version)
print('opencv', cv2.__version__)
PY
