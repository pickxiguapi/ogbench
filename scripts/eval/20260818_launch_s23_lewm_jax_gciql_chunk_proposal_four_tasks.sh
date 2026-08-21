#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WRAPPER="${OGBENCH_ROOT}/scripts/eval/20260818_record_s23_lewm_jax_gciql_chunk_proposal_task.sh"

[[ -f "${WRAPPER}" ]] || { echo "ERROR: wrapper unavailable: ${WRAPPER}" >&2; exit 1; }

launch_queue() {
  local gpu="$1"
  local session="$2"
  shift 2
  local queue_script=""
  local task
  for task in "$@"; do
    printf -v queue_script '%s CUDA_VISIBLE_DEVICES=%q bash %q %q %q;' \
      "${queue_script}" "${gpu}" "${WRAPPER}" "${task}" "${RUN_STAMP}"
  done
  tmux has-session -t "${session}" 2>/dev/null && { echo "ERROR: tmux session exists: ${session}" >&2; exit 1; }
  tmux new-session -d -s "${session}" "${queue_script}"
  echo "Started ${session} on GPU ${gpu}: $*"
}

launch_queue 0 "s23-lewm-guided-a-${RUN_STAMP}" cube reacher
launch_queue 5 "s23-lewm-guided-b-${RUN_STAMP}" pusht tworoom
