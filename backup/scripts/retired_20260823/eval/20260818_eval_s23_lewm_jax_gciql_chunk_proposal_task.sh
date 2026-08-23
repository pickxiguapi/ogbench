#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"
LEWM_RUNS_ROOT="${LEWM_RUNS_ROOT:-/data/dzb/stablewm-data/lewm-jax-runs}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-/data/dzb/stablewm-data/gciql-chunk-proposals-s11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dzb/stablewm-data/lewm-jax-guided-runs}"

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
  *) echo "Usage: bash $0 {cube|pusht|reacher|tworoom}" >&2; exit 2 ;;
esac

LEWM_CHECKPOINT="${LEWM_RUNS_ROOT}/${LEWM_EXP}/weights_epoch_10.msgpack"
PROPOSAL_DIR="${PROPOSAL_ROOT}/${TASK}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_RUN_ID}/${TASK}_seed42"

[[ -s "${LEWM_CHECKPOINT}" ]] || { echo "ERROR: LeWM checkpoint not found: ${LEWM_CHECKPOINT}" >&2; exit 1; }
[[ -s "${PROPOSAL_DIR}/params_100000.pkl" ]] || { echo "ERROR: proposal checkpoint not found: ${PROPOSAL_DIR}" >&2; exit 1; }
[[ -s "${PROPOSAL_DIR}/flags.json" ]] || { echo "ERROR: proposal flags not found: ${PROPOSAL_DIR}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/videos"
cd "${OGBENCH_ROOT}/impls"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="${OGBENCH_ROOT}:${OGBENCH_ROOT}/impls" \
"${OGBENCH_ROOT}/.venv/bin/python" eval_lewm_jax_cem.py \
  --task="${TASK}" --checkpoint="${LEWM_CHECKPOINT}" \
  --data-root="${DATA_ROOT}" \
  --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
  --proposal-method=gciql_chunk \
  --proposal-checkpoint-dir="${PROPOSAL_DIR}" --proposal-checkpoint-step=100000 \
  --proposal-temperature=0.0 \
  --video-dir="${OUTPUT_DIR}/videos" --output="${OUTPUT_DIR}/${TASK}.json" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" \
  --run-id "${EXPERIMENT_RUN_ID}" \
  --database "${EXPERIMENT_RECORDER_ROOT}/data/experiments.json" \
  --events "${EXPERIMENT_RECORDER_ROOT}/data/run_events.csv" \
  --catalog "${EXPERIMENT_RECORDER_ROOT}/data/experiment_catalog.json" \
  "${OUTPUT_DIR}/${TASK}.json"
