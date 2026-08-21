#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
VARIANT="${2:-}"
COST_MODE="${3:-}"
RUN_STAMP="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"
CEM_STEPS=5

DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/dzb/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"

case "${TASK}" in reacher|tworoom) ;; *) echo "Invalid task: ${TASK}" >&2; exit 2 ;; esac
case "${VARIANT}" in vanilla|guided) ;; *) echo "Invalid variant: ${VARIANT}" >&2; exit 2 ;; esac
case "${COST_MODE}" in terminal|min_over_horizon) ;; *) echo "Invalid cost mode: ${COST_MODE}" >&2; exit 2 ;; esac

if [[ "${COST_MODE}" == "terminal" ]]; then
  COST_TAG="terminalcost"
else
  COST_TAG="mincost"
fi

RUN_ID="EXP-026-s23-${TASK}-${VARIANT}-j5-${COST_TAG}-${RUN_STAMP}"
EXP_NAME="LeWMJAX_${VARIANT}_${TASK}_lewmE10_policyS100k_cem300x5_h5_rh1_ab5_${COST_TAG}_pairkey_seed42"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_eval_s23_lewm_jax_rh1_paired_task.sh"
PREPARE_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_prepare_s23_lewm_jax_rh1_paired_task.sh"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation Bash unavailable" >&2; exit 1; }
[[ -f "${PREPARE_SCRIPT}" ]] || { echo "ERROR: preparation Bash unavailable" >&2; exit 1; }

if [[ "${VARIANT}" == "guided" ]]; then
  ALGORITHM="LeWM-JAX+CEM+GCIQL-Chunk-proposal"
  PROPOSAL_INJECTION="first_block_initial_mean"
  PROPOSAL_STEP_JSON="100000"
else
  ALGORITHM="LeWM-JAX+CEM"
  PROPOSAL_INJECTION="none"
  PROPOSAL_STEP_JSON="null"
fi

payload="{\"algorithm\":\"${ALGORITHM}\",\"task\":\"${TASK}\",\"variant\":\"${VARIANT}\",\"lewm_epoch\":10,\"proposal_step\":${PROPOSAL_STEP_JSON},\"proposal_injection\":\"${PROPOSAL_INJECTION}\",\"evaluation\":{\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":25,\"eval_budget\":50,\"cem_samples\":300,\"cem_steps\":5,\"cem_topk\":30,\"horizon\":5,\"receding_horizon\":1,\"action_block\":5,\"cost_mode\":\"${COST_MODE}\",\"paired_plan_keys\":true}}"

EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-026 "${EXP_NAME}" "${RUN_ID}" \
  --train bash "${PREPARE_SCRIPT}" "${TASK}" "${VARIANT}" "${CEM_STEPS}" "${COST_MODE}" \
  --eval bash "${EVAL_SCRIPT}" "${TASK}" "${VARIANT}" "${CEM_STEPS}" "${COST_MODE}"
