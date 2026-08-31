#!/usr/bin/env bash
set -euo pipefail

# A800 node2：GPU0 使用 Cube seed3072 frozen LeWM z192 cache，训练 K5/K10 LatentPathFlow；仅 CFM loss、200k、bs1024、seed0。
CLIENT_ID=node2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
GPU_ID=${GPU_ID:-0}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
LATENT_DATASET="$LEWM_LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
EXP_NAME="latent_pathflow_cube_lewm3072_k5_10_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"

export CLIENT_ID OGBENCH_ROOT GPU_ID TRAIN_STEPS TRAIN_BATCH_SIZE TRAIN_SEED
export LATENT_DATASET SUBGOAL_RUNS_ROOT EXP_NAME
bash "$SCRIPT_DIR/20260831_train_latent_path_flow_k10_common.sh"
