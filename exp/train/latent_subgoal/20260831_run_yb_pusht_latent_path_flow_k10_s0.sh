#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU0 使用 PushT seed666 frozen LeWM z192 cache，训练 3 帧历史 K5/K10 LatentPathFlow；仅 CFM loss、200k、bs1024、seed0。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
GPU_ID=${GPU_ID:-0}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
SUBGOAL_STEPS=${SUBGOAL_STEPS:-10}
ACTION_BLOCK=${ACTION_BLOCK:-5}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/root/data/yyf/lewm-latent-datasets}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/root/data/yyf/lewm-final/latent-path-flow-k10}
LATENT_DATASET="$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
EXP_NAME="latent_pathflow_pusht_lewm666_hist3_sg${SUBGOAL_STEPS}_ab${ACTION_BLOCK}_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"

export CLIENT_ID OGBENCH_ROOT GPU_ID TRAIN_STEPS TRAIN_BATCH_SIZE TRAIN_SEED SUBGOAL_STEPS ACTION_BLOCK
export LATENT_DATASET SUBGOAL_RUNS_ROOT EXP_NAME
bash "$SCRIPT_DIR/20260831_train_latent_path_flow_k10_common.sh"
