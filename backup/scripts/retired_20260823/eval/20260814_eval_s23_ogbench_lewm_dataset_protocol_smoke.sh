#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench-lewm-envs-test"
OGBENCH_PYTHON="/home/dzb/ogbench/.venv/bin/python"
DATA_ROOT="/data/dzb/stablewm-data/datasets"
OUTPUT_DIR="/data/dzb/stablewm-data/ogbench-lewm-envs-smoke/20260814_dataset_protocol"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh" >&2
  exit 2
}
[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
for dataset in cube_single_expert.h5 pusht_expert_train.h5 tworoom.h5 reacher.h5; do
  [[ -s "${DATA_ROOT}/${dataset}" ]] || { echo "ERROR: missing dataset ${dataset}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
cd "${OGBENCH_ROOT}"

PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
"${OGBENCH_PYTHON}" - <<'PY' 2>&1 | tee "${OUTPUT_DIR}/protocol.log"
import json

import numpy as np

from ogbench.lewm_envs.evaluation import HDF5EvaluationDataset, evaluate_dataset_goals, task_paths


class ZeroPolicy:
    def reset(self, action_space, num_envs):
        self.action_shape = action_space.shape
        self.num_envs = num_envs

    def get_actions(self, pixels, goals, alive):
        return np.zeros((self.num_envs, *self.action_shape), dtype=np.float32)


data_root = '/data/dzb/stablewm-data/datasets'
results = {}
for task in ('cube', 'pusht', 'tworoom', 'reacher'):
    hdf5_path, _ = task_paths(task, data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(num_eval=1, goal_offset=25, seed=42)
        metrics = evaluate_dataset_goals(
            task=task,
            dataset=dataset,
            episodes=episodes,
            starts=starts,
            goal_offset=25,
            eval_budget=1,
            policy=ZeroPolicy(),
        )
        results[task] = {
            'episode': int(episodes[0]),
            'start': int(starts[0]),
            'success_rate': metrics['success_rate'],
            'episode_successes': metrics['episode_successes'].tolist(),
            'seeds': metrics['seeds'],
        }
    finally:
        dataset.close()

output = '/data/dzb/stablewm-data/ogbench-lewm-envs-smoke/20260814_dataset_protocol/results.json'
with open(output, 'w') as file:
    json.dump(results, file, indent=2)
print(json.dumps(results, indent=2))
PY
