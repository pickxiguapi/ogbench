#!/usr/bin/env bash
set -euo pipefail

# A800 node4：训练 OGBench 8 Tasks 的主 subgoal generator。八个数据集分别
# 使用 node3 已评测的 seed3072/epoch10 LeWM，先生成 checkpoint-bound z192
# cache，再训练一套 history3 K5/K10 LatentPathFlow；full uniform-future、
# CFM、200k、bs1024、EMA0.9999、Euler16、ns1、seed0。这里只跑主模型，
# 不训练 MLP 或 EndpointFlow 消融；默认 8 卡并行，也支持少卡分批续跑。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
DATASET_INDICES=${DATASET_INDICES:-"0 1 2 3 4 5 6 7"}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
VALIDATION_PAIRS=${VALIDATION_PAIRS:-10000}
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8-node3-evaluated-mirror}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents-ogbench8-node3-e10}
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-ogbench8-k10}
MANIFEST_ROOT=${MANIFEST_ROOT:-$RUNS_ROOT/manifests}

source "$OGBENCH_ROOT/scripts/client_env.sh"

envs=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)

GPU_IDS="$GPU_IDS" DATASET_INDICES="$DATASET_INDICES" LEWM_RUN_ROOT="$LEWM_RUN_ROOT" \
LEWM_LATENT_ROOT="$LEWM_LATENT_ROOT" \
  bash "$OGBENCH_ROOT/exp/preprocess/lewm_latents/20260904_precompute_node4_ogbench8_node3_lewm_z192.sh"

read -r -a gpu_ids <<< "$GPU_IDS"
read -r -a dataset_indices <<< "$DATASET_INDICES"
if (( ${#gpu_ids[@]} == 0 || ${#gpu_ids[@]} > 8 )); then
  echo "GPU_IDS must contain between one and eight GPU IDs." >&2
  exit 2
fi
if (( ${#dataset_indices[@]} == 0 )); then
  echo "DATASET_INDICES must contain at least one dataset index." >&2
  exit 2
fi
for index in "${dataset_indices[@]}"; do
  if (( index < 0 || index >= ${#tags[@]} )); then
    echo "Invalid DATASET_INDICES entry: $index" >&2
    exit 2
  fi
done
gpu_count=${#gpu_ids[@]}
mkdir -p "$MANIFEST_ROOT"

for index in "${dataset_indices[@]}"; do
  tag=${tags[$index]}
  latent_dataset="$LEWM_LATENT_ROOT/ogbench8_${tag}__node3_lewm_s3072_e10_z192.h5"
  (
    cd "$OGBENCH_ROOT/impls"
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" create_latent_subgoal_validation_manifest.py \
      --latent-dataset="$latent_dataset" \
      --output="$MANIFEST_ROOT/ogbench8_${tag}_h50_hist3_k10_n${VALIDATION_PAIRS}.npz" \
      --split-seed=0 --train-fraction=0.95 \
      --num-pairs="$VALIDATION_PAIRS" --history-size=3 \
      --goal-offset=50 --subgoal-steps=10 --action-block=5 --seed=1
  )
done

run_generator() {
  local gpu_id=$1
  local index=$2
  local tag=${tags[$index]}
  local latent_dataset="$LEWM_LATENT_ROOT/ogbench8_${tag}__node3_lewm_s3072_e10_z192.h5"
  local manifest="$MANIFEST_ROOT/ogbench8_${tag}_h50_hist3_k10_n${VALIDATION_PAIRS}.npz"
  local exp_name="latent_pathflow_ogbench8_${tag}_node3lewm3072e10_hist3_sg10_ab5_uniform_ns1_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
  local run_dir="$RUNS_ROOT/$exp_name"
  mkdir -p "$run_dir"

  (
    cd "$OGBENCH_ROOT/impls"
    CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" train_latent_subgoal_gcbc.py \
      --latent-dataset="$latent_dataset" \
      --validation-manifest="$manifest" \
      --save-dir="$run_dir" --exp-name="$exp_name" \
      --architecture=latent_path_flow \
      --history-size=3 --hidden-dim=512 --depth=4 \
      --num-heads=8 --ff-dim=2048 --time-dim=64 \
      --seed="$TRAIN_SEED" --split-seed=0 --train-fraction=0.95 \
      --subgoal-steps=10 --action-block=5 --goal-sampling=uniform_future \
      --train-steps="$TRAIN_STEPS" --batch-size="$TRAIN_BATCH_SIZE" \
      --flow-sampling-steps=16 --flow-solver=euler --num-samples=1 \
      --ema-decay=0.9999 --learning-rate=1e-4 --final-learning-rate=1e-5 \
      --warmup-steps=5000 --weight-decay=1e-4 --gradient-clip=1.0 \
      --validation-pairs="$VALIDATION_PAIRS" --eval-batch-size=1024 \
      --log-interval=1000 --eval-interval=10000 --checkpoint-interval=25000 \
      --resume 2>&1 | tee -a "$run_dir/train.log"
  )
}

failed=0
for (( base=0; base<${#dataset_indices[@]}; base+=gpu_count )); do
  pids=()
  for (( slot=0; slot<gpu_count && base+slot<${#dataset_indices[@]}; slot++ )); do
    run_generator "${gpu_ids[$slot]}" "${dataset_indices[$((base + slot))]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
