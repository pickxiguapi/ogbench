#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench-new}"
DATA_ROOT="${OGBENCH_DATA_DIR:-/root/data/yyf/ogbench-cache/data}"
POLL_SECONDS="${POLL_SECONDS:-30}"

CHUNK_LAUNCHER="${OGBENCH_ROOT}/scripts/train/20260818_launch_yb_hiql_chunk_two_v_visual_five_tasks.sh"
OFFICIAL_LAUNCHER="${OGBENCH_ROOT}/scripts/train/20260818_launch_yb_hiql_official_visual_tasks.sh"

wait_for_exact_file() {
  local path="$1"
  local expected_bytes="$2"
  local actual_bytes=0
  while true; do
    if [[ -f "${path}" ]]; then
      actual_bytes="$(stat -c %s "${path}")"
    fi
    if [[ "${actual_bytes}" == "${expected_bytes}" ]]; then
      echo "ready: ${path} (${actual_bytes} bytes)"
      return 0
    fi
    if (( actual_bytes > expected_bytes )); then
      echo "ERROR: ${path} is larger than expected (${actual_bytes} > ${expected_bytes})" >&2
      return 1
    fi
    echo "waiting: ${path} (${actual_bytes}/${expected_bytes} bytes)"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_session_exit() {
  local session="$1"
  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
}

wait_for_dataset() {
  local task="$1"
  case "${task}" in
    cube-triple)
      wait_for_exact_file "${DATA_ROOT}/visual-cube-triple-play-v0.npz" 5184220872
      wait_for_exact_file "${DATA_ROOT}/visual-cube-triple-play-v0-val.npz" 519456147
      ;;
    cube-quadruple)
      wait_for_exact_file "${DATA_ROOT}/visual-cube-quadruple-play-v0.npz" 8749433724
      wait_for_exact_file "${DATA_ROOT}/visual-cube-quadruple-play-v0-val.npz" 871637205
      ;;
    *)
      echo "ERROR: unsupported queued task: ${task}" >&2
      return 2
      ;;
  esac
}

run_chunk_queue() {
  wait_for_dataset cube-triple
  RUN_ATTEMPT=R01 bash "${CHUNK_LAUNCHER}" cube-triple
  wait_for_session_exit exp021-hiqlchunk2v-cube-triple
  wait_for_dataset cube-quadruple
  RUN_ATTEMPT=R01 bash "${CHUNK_LAUNCHER}" cube-quadruple
}

run_official_queue() {
  wait_for_dataset cube-triple
  RUN_ATTEMPT=R01 bash "${OFFICIAL_LAUNCHER}" cube-triple
  wait_for_session_exit exp022-hiql-cube-triple
  wait_for_dataset cube-quadruple
  RUN_ATTEMPT=R01 bash "${OFFICIAL_LAUNCHER}" cube-quadruple
}

run_chunk_queue &
chunk_queue_pid=$!
run_official_queue &
official_queue_pid=$!
wait "${chunk_queue_pid}"
wait "${official_queue_pid}"
