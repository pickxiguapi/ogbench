#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
GOAL_OFFSET="${2:-}"
EVAL_BUDGET="${3:-}"
RUN_STAMP="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"

DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/dzb/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
VARIANT="guided"
CEM_STEPS=5
COST_MODE="min_over_horizon"

case "${TASK}" in cube|pusht|reacher|tworoom) ;; *) echo "Invalid task: ${TASK}" >&2; exit 2 ;; esac
case "${GOAL_OFFSET}:${EVAL_BUDGET}" in
  50:100|75:150) ;;
  *) echo "Expected goal/budget pair 50/100 or 75/150, got ${GOAL_OFFSET}/${EVAL_BUDGET}" >&2; exit 2 ;;
esac

RUN_ID="EXP-027-s23-${TASK}-guided-j5-mincost-g${GOAL_OFFSET}-b${EVAL_BUDGET}-${RUN_STAMP}"
EXP_NAME="LeWMJAX_guided_${TASK}_lewmE10_policyS100k_cem300x5_h5_rh1_ab5_mincost_g${GOAL_OFFSET}_b${EVAL_BUDGET}_pairkey_seed42"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_eval_s23_lewm_jax_rh1_paired_task.sh"
PREPARE_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_prepare_s23_lewm_jax_rh1_paired_task.sh"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation Bash unavailable" >&2; exit 1; }
[[ -f "${PREPARE_SCRIPT}" ]] || { echo "ERROR: preparation Bash unavailable" >&2; exit 1; }

payload="{\"algorithm\":\"LeWM-JAX+CEM+GCIQL-Chunk-proposal\",\"task\":\"${TASK}\",\"variant\":\"guided\",\"lewm_epoch\":10,\"proposal_step\":100000,\"proposal_injection\":\"first_block_initial_mean\",\"evaluation\":{\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":${GOAL_OFFSET},\"eval_budget\":${EVAL_BUDGET},\"cem_samples\":300,\"cem_steps\":5,\"cem_topk\":30,\"horizon\":5,\"receding_horizon\":1,\"action_block\":5,\"cost_mode\":\"min_over_horizon\",\"paired_plan_keys\":true}}"

EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-027 "${EXP_NAME}" "${RUN_ID}" \
  --train bash "${PREPARE_SCRIPT}" "${TASK}" "${VARIANT}" "${CEM_STEPS}" "${COST_MODE}" \
  --eval bash "${EVAL_SCRIPT}" "${TASK}" "${VARIANT}" "${CEM_STEPS}" "${COST_MODE}" "${GOAL_OFFSET}" "${EVAL_BUDGET}"
