#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-2}"
WRAPPER="${OGBENCH_ROOT}/scripts/eval/20260818_record_s23_lewm_jax_rh1_costpair_j5_task.sh"
SESSION="s23-rh1-costpair-j5-reacher-tworoom-${RUN_STAMP}"

[[ -f "${WRAPPER}" ]] || { echo "ERROR: wrapper unavailable: ${WRAPPER}" >&2; exit 1; }

QUEUE_SCRIPT=""
for task in reacher tworoom; do
  for cost_mode in terminal min_over_horizon; do
    for variant in vanilla guided; do
      printf -v QUEUE_SCRIPT '%s CUDA_VISIBLE_DEVICES=%q bash %q %q %q %q %q;' \
        "${QUEUE_SCRIPT}" "${GPU}" "${WRAPPER}" "${task}" "${variant}" "${cost_mode}" "${RUN_STAMP}"
    done
  done
done

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "ERROR: tmux session exists: ${SESSION}" >&2
  exit 1
}
tmux new-session -d -s "${SESSION}" "${QUEUE_SCRIPT}"
echo "Started ${SESSION} on GPU ${GPU}: Reacher then TwoRoom, terminal/min-over-horizon, Vanilla/guided, J=5."
