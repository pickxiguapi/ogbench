#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
VARIANT="${2:-}"
CEM_STEPS="${3:-}"
COST_MODE="${4:-terminal}"
GOAL_OFFSET="${5:-25}"
EVAL_BUDGET="${6:-50}"

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"
LEWM_RUNS_ROOT="${LEWM_RUNS_ROOT:-/data/dzb/stablewm-data/lewm-jax-runs}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-/data/dzb/stablewm-data/gciql-chunk-proposals-s11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dzb/stablewm-data/lewm-jax-rh1-paired-runs}"

[[ -n "${EXPERIMENT_RUN_ID:-}" ]] || {
  echo "ERROR: launch through experiment-dashboard/scripts/recorded_run.sh" >&2
  exit 2
}

case "${TASK}" in
  cube)
    LEWM_EXP="LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
    ;;
  pusht)
    LEWM_EXP="LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
    ;;
  reacher)
    LEWM_EXP="LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
    ;;
  tworoom)
    LEWM_EXP="LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
    ;;
  *) echo "Usage: bash $0 {cube|pusht|reacher|tworoom} {vanilla|guided} {1|5|10|30} [terminal|min_over_horizon]" >&2; exit 2 ;;
esac

case "${VARIANT}" in
  vanilla|guided) ;;
  *) echo "Usage: bash $0 {cube|pusht|reacher|tworoom} {vanilla|guided} {1|5|10|30} [terminal|min_over_horizon]" >&2; exit 2 ;;
esac

case "${CEM_STEPS}" in
  1|5|10|30) ;;
  *) echo "Usage: bash $0 {cube|pusht|reacher|tworoom} {vanilla|guided} {1|5|10|30} [terminal|min_over_horizon]" >&2; exit 2 ;;
esac

case "${COST_MODE}" in
  terminal|min_over_horizon) ;;
  *) echo "Invalid CEM cost mode: ${COST_MODE}" >&2; exit 2 ;;
esac

[[ "${GOAL_OFFSET}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid goal offset: ${GOAL_OFFSET}" >&2; exit 2; }
[[ "${EVAL_BUDGET}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid evaluation budget: ${EVAL_BUDGET}" >&2; exit 2; }

LEWM_CHECKPOINT="${LEWM_RUNS_ROOT}/${LEWM_EXP}/weights_epoch_10.msgpack"
PROPOSAL_DIR="${PROPOSAL_ROOT}/${TASK}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_RUN_ID}/${TASK}_${VARIANT}_j${CEM_STEPS}_g${GOAL_OFFSET}_b${EVAL_BUDGET}_seed42"
RESULT_JSON="${OUTPUT_DIR}/${TASK}_${VARIANT}_j${CEM_STEPS}_g${GOAL_OFFSET}_b${EVAL_BUDGET}.json"

[[ -s "${LEWM_CHECKPOINT}" ]] || { echo "ERROR: LeWM checkpoint not found: ${LEWM_CHECKPOINT}" >&2; exit 1; }
if [[ "${VARIANT}" == "guided" ]]; then
  [[ -s "${PROPOSAL_DIR}/params_100000.pkl" ]] || { echo "ERROR: proposal checkpoint not found: ${PROPOSAL_DIR}" >&2; exit 1; }
  [[ -s "${PROPOSAL_DIR}/flags.json" ]] || { echo "ERROR: proposal flags not found: ${PROPOSAL_DIR}" >&2; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

PROPOSAL_ARGS=()
if [[ "${VARIANT}" == "guided" ]]; then
  PROPOSAL_ARGS=(
    --proposal-method=gciql_chunk
    --proposal-checkpoint-dir="${PROPOSAL_DIR}"
    --proposal-checkpoint-step=100000
    --proposal-temperature=0.0
  )
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_ROOT}/.venv/bin/python" eval_lewm_jax_cem.py \
  --task="${TASK}" --checkpoint="${LEWM_CHECKPOINT}" \
  --data-root="${DATA_ROOT}" \
  --num-eval=50 --seed=42 --goal-offset-steps="${GOAL_OFFSET}" --eval-budget="${EVAL_BUDGET}" \
  --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
  --cem-num-samples=300 --cem-steps="${CEM_STEPS}" --cem-topk=30 --cem-var-scale=1.0 \
  --cem-cost-mode="${COST_MODE}" \
  --paired-plan-keys \
  "${PROPOSAL_ARGS[@]}" \
  --video-dir="${OUTPUT_DIR}/videos" --output="${RESULT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" \
  --run-id "${EXPERIMENT_RUN_ID}" \
  --database "${EXPERIMENT_RECORDER_ROOT}/data/experiments.json" \
  --events "${EXPERIMENT_RECORDER_ROOT}/data/run_events.csv" \
  --catalog "${EXPERIMENT_RECORDER_ROOT}/data/experiment_catalog.json" \
  "${RESULT_JSON}"
