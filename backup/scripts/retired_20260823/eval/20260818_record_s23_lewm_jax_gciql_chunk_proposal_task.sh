#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
RUN_STAMP="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/dzb/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"

case "${TASK}" in
  cube|pusht|reacher|tworoom) ;;
  *) echo "Usage: bash $0 {cube|pusht|reacher|tworoom} [RUN_STAMP]" >&2; exit 2 ;;
esac

RUN_ID="EXP-023-s23-${TASK}-${RUN_STAMP}"
EXP_NAME="LeWMJAX_GCIQLChunkProposal_${TASK}_lewmE10_policyS100k_k5_cem300x30_h5_rh5_seed42"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_eval_s23_lewm_jax_gciql_chunk_proposal_task.sh"
PREPARE_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_prepare_s23_lewm_jax_gciql_chunk_proposal_task.sh"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation Bash unavailable" >&2; exit 1; }
[[ -f "${PREPARE_SCRIPT}" ]] || { echo "ERROR: preparation Bash unavailable" >&2; exit 1; }

payload="{\"algorithm\":\"LeWM-JAX+CEM+GCIQL-Chunk-proposal\",\"task\":\"${TASK}\",\"lewm_epoch\":10,\"proposal_step\":100000,\"proposal_injection\":\"first_block_initial_mean\",\"evaluation\":{\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":25,\"eval_budget\":50,\"cem_samples\":300,\"cem_steps\":30,\"horizon\":5,\"receding_horizon\":5,\"action_block\":5}}"

EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-023 "${EXP_NAME}" "${RUN_ID}" \
  --train bash "${PREPARE_SCRIPT}" "${TASK}" \
  --eval bash "${EVAL_SCRIPT}" "${TASK}"
