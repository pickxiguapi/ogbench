#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
GOAL_OFFSET="${2:-}"
EVAL_BUDGET="${3:-}"

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/dzb/stablewm-data/gciql-chunk-proposals-s11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dzb/stablewm-data/gciql-chunk-policy-distance-runs}"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || { echo "ERROR: launch through recorded_run.sh" >&2; exit 2; }
case "${TASK}" in cube|pusht|reacher|tworoom) ;; *) echo "Invalid task: ${TASK}" >&2; exit 2 ;; esac
case "${GOAL_OFFSET}:${EVAL_BUDGET}" in
  50:100|75:150) ;;
  *) echo "Expected goal/budget pair 50/100 or 75/150, got ${GOAL_OFFSET}/${EVAL_BUDGET}" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${TASK}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_RUN_ID}/${TASK}_g${GOAL_OFFSET}_b${EVAL_BUDGET}_seed42"
RESULT_JSON="${OUTPUT_DIR}/${TASK}_gciql_chunk_awr_g${GOAL_OFFSET}_b${EVAL_BUDGET}.json"

[[ -s "${CHECKPOINT_DIR}/params_100000.pkl" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT_DIR}" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags not found: ${CHECKPOINT_DIR}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_ROOT}/.venv/bin/python" eval_ogbench_agent_lewm_envs.py \
  --task="${TASK}" --method=gciql_chunk \
  --checkpoint-dir="${CHECKPOINT_DIR}" --checkpoint-step=100000 \
  --data-root="${DATA_ROOT}" \
  --num-eval=50 --seed=42 --goal-offset-steps="${GOAL_OFFSET}" --eval-budget="${EVAL_BUDGET}" \
  --video-dir="${OUTPUT_DIR}/videos" --output="${RESULT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" \
  --run-id "${EXPERIMENT_RUN_ID}" \
  --database "${EXPERIMENT_RECORDER_ROOT}/data/experiments.json" \
  --events "${EXPERIMENT_RECORDER_ROOT}/data/run_events.csv" \
  --catalog "${EXPERIMENT_RECORDER_ROOT}/data/experiment_catalog.json" \
  "${RESULT_JSON}"
