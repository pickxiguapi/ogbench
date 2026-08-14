#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench-lewm-envs-test"
OGBENCH_PYTHON="/home/dzb/ogbench/.venv/bin/python"
OUTPUT_DIR="/data/dzb/stablewm-data/ogbench-lewm-envs-smoke/20260814"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh --eval-only" >&2
  exit 2
}
[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/pyproject.toml" ]] || { echo "ERROR: isolated OGBench test checkout not found" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
cd "${OGBENCH_ROOT}"

PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
"${OGBENCH_PYTHON}" - <<'PY' 2>&1 | tee "${OUTPUT_DIR}/smoke.log"
import json

import gymnasium as gym
import numpy as np
import ogbench

results = {}
cases = {
    'cube': (
        'ogbench-lewm/CubeSingle-v0',
        {'env_type': 'single', 'ob_type': 'states', 'multiview': False, 'width': 224, 'height': 224},
    ),
    'pusht': ('ogbench-lewm/PushT-v1', {}),
    'tworoom': ('ogbench-lewm/TwoRoom-v1', {}),
    'reacher': ('ogbench-lewm/Reacher-v0', {'task': 'qpos_match'}),
}
for task, (env_id, kwargs) in cases.items():
    env = gym.make(env_id, render_mode='rgb_array', **kwargs)
    observation, info = env.reset(seed=0)
    frame = np.asarray(env.render())
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, reward, terminated, truncated, step_info = env.step(action)
    results[task] = {
        'observation_type': type(observation).__name__,
        'frame_shape': list(frame.shape),
        'frame_dtype': str(frame.dtype),
        'action_shape': list(env.action_space.shape),
        'reward': float(reward),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'info_keys': sorted(step_info),
    }
    env.close()

output = '/data/dzb/stablewm-data/ogbench-lewm-envs-smoke/20260814/results.json'
with open(output, 'w') as file:
    json.dump(results, file, indent=2)
print(json.dumps(results, indent=2))
PY
