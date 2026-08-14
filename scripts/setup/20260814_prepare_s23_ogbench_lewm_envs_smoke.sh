#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench-lewm-envs-test"
OGBENCH_PYTHON="/home/dzb/ogbench/.venv/bin/python"

[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/pyproject.toml" ]] || { echo "ERROR: isolated OGBench test checkout not found" >&2; exit 1; }

cd "${OGBENCH_ROOT}"
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_PYTHON}" - <<'PY'
import cv2
import gymnasium
import ogbench
import pygame
import pymunk

print('OGBench LeWM environment smoke prerequisites ready')
print('pygame', pygame.__version__)
print('pymunk', pymunk.version)
print('opencv', cv2.__version__)
PY
