#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
GOAL_OFFSET="${2:-}"
EVAL_BUDGET="${3:-}"
RUN_STAMP="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"

DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/dzb/experiment-dashboard}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"

case "${TASK}" in cube|pusht|reacher|tworoom) ;; *) echo "Invalid task: ${TASK}" >&2; exit 2 ;; esac
case "${GOAL_OFFSET}:${EVAL_BUDGET}" in
  50:100|75:150) ;;
  *) echo "Expected goal/budget pair 50/100 or 75/150, got ${GOAL_OFFSET}/${EVAL_BUDGET}" >&2; exit 2 ;;
esac

RUN_ID="EXP-029-s23-${TASK}-gciqlchunk-policy-g${GOAL_OFFSET}-b${EVAL_BUDGET}-${RUN_STAMP}"
EXP_NAME="GCIQLChunkAWR_policyonly_${TASK}_k5_s100k_alpha3_g${GOAL_OFFSET}_b${EVAL_BUDGET}_seed42"
PREPARE_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_prepare_s23_gciql_chunk_awr_policy_distance_task.sh"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_eval_s23_gciql_chunk_awr_policy_distance_task.sh"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: recorder unavailable" >&2; exit 1; }
[[ -f "${PREPARE_SCRIPT}" ]] || { echo "ERROR: preparation Bash unavailable" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation Bash unavailable" >&2; exit 1; }

payload="{\"algorithm\":\"GCIQL-Chunk-AWR-policy-only\",\"task\":\"${TASK}\",\"variant\":\"policy_only\",\"checkpoint_step\":100000,\"actor_loss\":\"awr\",\"alpha\":3.0,\"action_chunk\":5,\"evaluation\":{\"num_eval\":50,\"seed\":42,\"goal_offset_steps\":${GOAL_OFFSET},\"eval_budget\":${EVAL_BUDGET},\"temperature\":0.0}}"

EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
  EXP-029 "${EXP_NAME}" "${RUN_ID}" \
  --train bash "${PREPARE_SCRIPT}" "${TASK}" "${GOAL_OFFSET}" "${EVAL_BUDGET}" \
  --eval bash "${EVAL_SCRIPT}" "${TASK}" "${GOAL_OFFSET}" "${EVAL_BUDGET}"
