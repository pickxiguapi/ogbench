#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
VARIANT="${2:-}"
CEM_STEPS="${3:-}"
COST_MODE="${4:-terminal}"

LEWM_RUNS_ROOT="${LEWM_RUNS_ROOT:-/data/dzb/stablewm-data/lewm-jax-runs}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-/data/dzb/stablewm-data/gciql-chunk-proposals-s11}"

case "${TASK}" in
  cube) LEWM_EXP="LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95" ;;
  pusht) LEWM_EXP="LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95" ;;
  reacher) LEWM_EXP="LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95" ;;
  tworoom) LEWM_EXP="LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95" ;;
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

LEWM_CHECKPOINT="${LEWM_RUNS_ROOT}/${LEWM_EXP}/weights_epoch_10.msgpack"
[[ -s "${LEWM_CHECKPOINT}" ]] || { echo "ERROR: LeWM checkpoint not found: ${LEWM_CHECKPOINT}" >&2; exit 1; }

if [[ "${VARIANT}" == "guided" ]]; then
  PROPOSAL_DIR="${PROPOSAL_ROOT}/${TASK}"
  [[ -s "${PROPOSAL_DIR}/params_100000.pkl" ]] || { echo "ERROR: proposal checkpoint not found: ${PROPOSAL_DIR}" >&2; exit 1; }
  [[ -s "${PROPOSAL_DIR}/flags.json" ]] || { echo "ERROR: proposal flags not found: ${PROPOSAL_DIR}" >&2; exit 1; }
fi

echo "Validated ${TASK} ${VARIANT} J=${CEM_STEPS}: rh=1, block=5, samples=300, cost=${COST_MODE}, paired planner keys."
