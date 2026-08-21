#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-2}"
WRAPPER="${OGBENCH_ROOT}/scripts/eval/20260818_record_s23_gciql_chunk_awr_policy_distance_task.sh"
SESSION="s23-gciqlchunk-policy-distance-${RUN_STAMP}"

[[ -f "${WRAPPER}" ]] || { echo "ERROR: wrapper unavailable: ${WRAPPER}" >&2; exit 1; }

QUEUE_SCRIPT=""
for goal_offset in 50 75; do
  if [[ "${goal_offset}" == "50" ]]; then
    eval_budget=100
  else
    eval_budget=150
  fi
  for task in cube pusht reacher tworoom; do
    printf -v QUEUE_SCRIPT '%s CUDA_VISIBLE_DEVICES=%q bash %q %q %q %q %q;' \
      "${QUEUE_SCRIPT}" "${GPU}" "${WRAPPER}" "${task}" "${goal_offset}" "${eval_budget}" "${RUN_STAMP}"
  done
done

tmux has-session -t "${SESSION}" 2>/dev/null && { echo "ERROR: tmux session exists: ${SESSION}" >&2; exit 1; }
tmux new-session -d -s "${SESSION}" "${QUEUE_SCRIPT}"
echo "Started ${SESSION} on GPU ${GPU}: GCIQL-Chunk-AWR policy-only, four tasks, goal/budget 50/100 then 75/150."
