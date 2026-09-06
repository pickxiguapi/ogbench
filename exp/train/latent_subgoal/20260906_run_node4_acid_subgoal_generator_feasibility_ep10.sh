#!/usr/bin/env bash
set -euo pipefail

# Fast generator-level action-feasibility screen on A800 node4.
# Everything in LeWM++ is fixed except the H50 subgoal generator:
# history MLP, endpoint-only flow, or LatentPathFlow.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/20260906_run_node4_acid_subgoal_reachability.sh"

ROOT=/data-training/yyf/ogbench-lewm-policy-runs
EVAL_ROOT=${EVAL_ROOT:-$ROOT/evals/lewm-4tasks/20260906_acid_subgoal_generator_feasibility_h50_general_uniform_future_lewmpp_policy777_ns1_cem300x5_h2_rh1_train0_eval42_ep10}
TMP_ROOT=${TMP_ROOT:-$ROOT/tmp/20260906-acid-subgoal-generator-feasibility-ep10}

env \
  MODE="${MODE:-launch}" \
  PIPELINE=selected_plans \
  SESSION="${SESSION:-acid-subgoal-generators-h50-ep10}" \
  WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}" \
  GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}" \
  TRAIN_STEPS=200000 \
  NUM_EVAL="${NUM_EVAL:-10}" \
  ARCHITECTURES="${ARCHITECTURES:-history_mlp endpoint_flow latent_path_flow}" \
  TRAIN_SEEDS="${TRAIN_SEEDS:-0}" \
  EVAL_SEEDS="${EVAL_SEEDS:-42}" \
  EVAL_ROOT="$EVAL_ROOT" \
  TMP_ROOT="$TMP_ROOT" \
  bash "$RUNNER"
