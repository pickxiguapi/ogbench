#!/usr/bin/env bash
set -euo pipefail

# Strict single-variable ablation around the canonical H50 LeWM++ controller:
# fixed mixed LeWM checkpoints, shared-all GCIQL-Chunk seed777 mode guidance to
# the final goal, ns1, MoH, H2/RH1/J5, CEM300x5, budget100, 50 episodes, and
# eval seeds 0/1/42. Only the subgoal predictor architecture/checkpoint changes.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"} \
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1} \
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-30} \
ARCHITECTURES=${ARCHITECTURES:-"history_mlp endpoint_flow latent_path_flow"} \
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 42"} \
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"} \
NUM_EVAL=${NUM_EVAL:-50} \
POLICY_GUIDANCE=mode \
GUIDANCE_GOAL_MODE=final \
POLICY_SEED=777 \
POLICY_STEPS=100000 \
CEM_ITERATIONS=5 \
bash "$SCRIPT_DIR/20260904_eval_node4_h50_predictor_ablation.sh"
