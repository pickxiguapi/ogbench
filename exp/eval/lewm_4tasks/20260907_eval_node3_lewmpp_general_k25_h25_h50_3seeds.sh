#!/usr/bin/env bash
set -euo pipefail

# K=25 general LatentPathFlow evaluation. The shared launcher performs strict
# checkpoint/config validation and evaluates H25/H50 for seeds 0/1/42.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBGOAL_STEPS=25 \
SUBGOAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k25 \
bash "$SCRIPT_DIR/20260907_eval_node3_lewmpp_general_k15_h25_h50_3seeds.sh"
