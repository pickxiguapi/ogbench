#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU0 先生成 PushT seed666 LeWM z192 cache，再训练 direct-z Latent-GCBC；K10、MSE、HIQL future goal、100k、bs1024、seed0。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
GPU_ID=${GPU_ID:-0}
LATENT_BATCH_SIZE=${LATENT_BATCH_SIZE:-512}
DECODE_WORKERS=${DECODE_WORKERS:-16}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
source "$OGBENCH_ROOT/scripts/client_env.sh"

LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/root/data/yyf/lewm-latent-datasets}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/root/data/yyf/lewm-final/latent-subgoal-gcbc-k10}
latent_dataset="$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
exp_name="latent_gcbc_pusht_lewm666_k10_mse_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
run_dir="$SUBGOAL_RUNS_ROOT/$exp_name"

GPU_ID="$GPU_ID" BATCH_SIZE="$LATENT_BATCH_SIZE" DECODE_WORKERS="$DECODE_WORKERS" SMOKE_ROWS=0 \
bash "$OGBENCH_ROOT/exp/preprocess/lewm_latents/20260830_precompute_yb_pusht_lewm_s666_z192.sh"

mkdir -p "$run_dir"
cd "$OGBENCH_ROOT/impls"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" train_latent_subgoal_gcbc.py \
  --latent-dataset="$latent_dataset" \
  --save-dir="$run_dir" \
  --exp-name="$exp_name" \
  --seed="$TRAIN_SEED" \
  --split-seed=0 \
  --train-fraction=0.95 \
  --subgoal-steps=10 \
  --train-steps="$TRAIN_STEPS" \
  --batch-size="$TRAIN_BATCH_SIZE" \
  --hidden-dims 512 512 512 \
  --learning-rate=3e-4 \
  --final-learning-rate=3e-5 \
  --warmup-steps=2000 \
  --weight-decay=1e-4 \
  --gradient-clip=1.0 \
  --validation-pairs=50000 \
  --eval-batch-size=5000 \
  --log-interval=1000 \
  --eval-interval=5000 \
  --checkpoint-interval=25000 \
  --resume \
  2>&1 | tee -a "$run_dir/train.log"
