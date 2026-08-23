#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
TRAIN_SCRIPT="${OGBENCH_ROOT}/scripts/train/20260819_train_server7002_lewm_jax_visual_cube_play_e10.sh"

tasks=(single double triple)
gpus=(0 1 2)

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not found" >&2; exit 1; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: training Bash not found: ${TRAIN_SCRIPT}" >&2; exit 1; }

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  gpu="${gpus[$index]}"
  session="lewm-jax-cube-${task}-play-e10-fs1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
done

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  gpu="${gpus[$index]}"
  session="lewm-jax-cube-${task}-play-e10-fs1"
  tmux new-session -d -s "${session}" -c "${OGBENCH_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      bash "${TRAIN_SCRIPT}" "${task}"
  echo "started ${session}: GPU ${gpu}, visual-cube-${task}-play-v0"
done

echo "All three LeWM JAX runs were launched from workspace Bash scripts."
