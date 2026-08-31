#!/usr/bin/env bash
set -euo pipefail

# Cube uses the canonical seed3072 LeWM latent cache on A800 node2.
CLIENT_ID=node2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
GPU_ID=${GPU_ID:-0}
LATENT_BATCH_SIZE=${LATENT_BATCH_SIZE:-512}
DECODE_WORKERS=${DECODE_WORKERS:-16}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-subgoal-flow-transformer-k10}
LATENT_DATASET="$LEWM_LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
EXP_NAME="latent_flowtf_cube_lewm3072_k10_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"

GPU_ID="$GPU_ID" BATCH_SIZE="$LATENT_BATCH_SIZE" DECODE_WORKERS="$DECODE_WORKERS" SMOKE_ROWS=0 \
bash "$OGBENCH_ROOT/exp/preprocess/lewm_latents/20260830_precompute_node2_cube_lewm_s3072_z192.sh"

export CLIENT_ID OGBENCH_ROOT GPU_ID TRAIN_STEPS TRAIN_BATCH_SIZE TRAIN_SEED
export LATENT_DATASET SUBGOAL_RUNS_ROOT EXP_NAME
bash "$SCRIPT_DIR/20260831_train_flow_transformer_k10_common.sh"
