#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench-release-audit-20260814"
PYTHON_BIN="/root/data/yyf/ogbench/.venv/bin/python"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || { echo "ERROR: launch through recorded_run.sh" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -d "${OGBENCH_ROOT}/ogbench/lewm_envs" ]] || { echo "ERROR: built-in LeWM environments not found" >&2; exit 1; }

cd "${OGBENCH_ROOT}"
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" "${PYTHON_BIN}" - <<'PY'
import cv2
import flax
import gymnasium as gym
import jax
import ogbench  # noqa: F401
import pygame
import pymunk

expected = {
    'ogbench-lewm/CubeSingle-v0',
    'ogbench-lewm/PushT-v1',
    'ogbench-lewm/TwoRoom-v1',
    'ogbench-lewm/Reacher-v0',
}
missing = expected.difference(gym.registry)
if missing:
    raise RuntimeError(f'Missing OGBench LeWM environments: {sorted(missing)}')
print(
    {
        'jax': jax.__version__,
        'flax': flax.__version__,
        'pygame': pygame.version.ver,
        'pymunk': getattr(pymunk, 'version', None),
        'opencv': cv2.__version__,
        'env_ids': sorted(expected),
    }
)
PY
