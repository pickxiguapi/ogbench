#!/usr/bin/env bash
set -euo pipefail

cd /home/dzb/ogbench
worker=/home/dzb/ogbench/scripts/eval/20260820_eval_s23_lewm_jax_visual_gciql_chunk_awr_rh1_task.sh
stamp=$(date +%Y%m%d_%H%M%S)

for task in single double triple scene; do
  session="s23-lewm-visual-gcawr-rh1-${task}-${stamp}"
  tmux new-session -d -s "$session" "bash '$worker' '$task'"
  echo "Started $session"
done
