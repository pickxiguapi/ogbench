#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-2}"
WRAPPER="${OGBENCH_ROOT}/scripts/eval/20260818_record_s23_lewm_jax_rh1_min_cost_task.sh"
SESSION="s23-rh1-mincost-pusht-cube-${RUN_STAMP}"

[[ -f "${WRAPPER}" ]] || { echo "ERROR: wrapper unavailable: ${WRAPPER}" >&2; exit 1; }

QUEUE_SCRIPT=""
for task in pusht cube; do
  for steps in 1 5; do
    for variant in vanilla guided; do
      printf -v QUEUE_SCRIPT '%s CUDA_VISIBLE_DEVICES=%q bash %q %q %q %q %q;' \
        "${QUEUE_SCRIPT}" "${GPU}" "${WRAPPER}" "${task}" "${variant}" "${steps}" "${RUN_STAMP}"
    done
  done
done

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "ERROR: tmux session exists: ${SESSION}" >&2
  exit 1
}
tmux new-session -d -s "${SESSION}" "${QUEUE_SCRIPT}"
echo "Started ${SESSION} on GPU ${GPU}: PushT then Cube, min-over-horizon cost, J=1/5."
