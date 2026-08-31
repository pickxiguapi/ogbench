#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU0 使用 PushT seed666 frozen LeWM z192 cache，训练 K10 单点 Transformer-CFM，并按 NUM_SAMPLES 做 medoid validation。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
GPU_ID=${GPU_ID:-0}
LATENT_BATCH_SIZE=${LATENT_BATCH_SIZE:-512}
DECODE_WORKERS=${DECODE_WORKERS:-16}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
NUM_SAMPLES=${NUM_SAMPLES:-8}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/root/data/yyf/lewm-latent-datasets}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/root/data/yyf/lewm-final/latent-subgoal-flow-transformer-k10}
LATENT_DATASET="$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
EXP_NAME="latent_flowtf_pusht_lewm666_k10_cfm_ns${NUM_SAMPLES}_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"

GPU_ID="$GPU_ID" BATCH_SIZE="$LATENT_BATCH_SIZE" DECODE_WORKERS="$DECODE_WORKERS" SMOKE_ROWS=0 \
bash "$OGBENCH_ROOT/exp/preprocess/lewm_latents/20260830_precompute_yb_pusht_lewm_s666_z192.sh"

export CLIENT_ID OGBENCH_ROOT GPU_ID TRAIN_STEPS TRAIN_BATCH_SIZE TRAIN_SEED NUM_SAMPLES
export LATENT_DATASET SUBGOAL_RUNS_ROOT EXP_NAME
bash "$SCRIPT_DIR/20260831_train_flow_transformer_k10_common.sh"
