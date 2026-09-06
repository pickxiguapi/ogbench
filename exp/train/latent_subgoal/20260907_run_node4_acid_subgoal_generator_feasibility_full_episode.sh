#!/usr/bin/env bash
set -euo pipefail

# Full-episode H50 comparison of subgoal-generator action feasibility.
# Fixed generator training seed 0; 50 episodes for each of three eval seeds.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/20260906_run_node4_acid_subgoal_reachability.sh"

ROOT=/data-training/yyf/ogbench-lewm-policy-runs
EVAL_ROOT=${EVAL_ROOT:-$ROOT/evals/lewm-4tasks/20260907_acid_subgoal_generator_feasibility_full_episode_h50_general_uniform_future_lewmpp_policy777_ns1_cem300x5_h2_rh1_train0_eval0-1-42_ep50}
TMP_ROOT=${TMP_ROOT:-$ROOT/tmp/20260907-acid-subgoal-generator-feasibility-full-episode}

env \
  MODE="${MODE:-launch}" \
  PIPELINE=selected_plans \
  SESSION="${SESSION:-acid-subgoal-generators-h50-full-ep50}" \
  WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}" \
  GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}" \
  TRAIN_STEPS=200000 \
  NUM_EVAL="${NUM_EVAL:-50}" \
  ARCHITECTURES="${ARCHITECTURES:-history_mlp endpoint_flow latent_path_flow}" \
  TRAIN_SEEDS="${TRAIN_SEEDS:-0}" \
  EVAL_SEEDS="${EVAL_SEEDS:-0 1 42}" \
  EVAL_ROOT="$EVAL_ROOT" \
  TMP_ROOT="$TMP_ROOT" \
  bash "$RUNNER"
