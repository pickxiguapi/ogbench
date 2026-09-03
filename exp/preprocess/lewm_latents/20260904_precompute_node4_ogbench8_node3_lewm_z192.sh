#!/usr/bin/env bash
set -euo pipefail

# A800 node4：把 OGBench 8 Tasks 官方 train NPZ 分别编码成 node3 已完成
# 正式评测的 seed3072/epoch10 LeWM frozen z192 cache。每个数据集严格绑定
# 自己的 checkpoint SHA-256，输出 float32 HDF5，支持 incomplete 断点续跑；
# 默认 8 卡并行，也可通过 GPU_IDS 使用较少空闲卡分批执行。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
DATASET_INDICES=${DATASET_INDICES:-"0 1 2 3 4 5 6 7"}
BATCH_SIZE=${BATCH_SIZE:-512}
SMOKE_ROWS=${SMOKE_ROWS:-0}
JAX_PLATFORM=${JAX_PLATFORM:-cuda}
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8-node3-evaluated-mirror}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents-ogbench8-node3-e10}

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
  if (( index < 0 || index >= ${#envs[@]} )); then
    echo "Invalid DATASET_INDICES entry: $index" >&2
    exit 2
  fi
done
gpu_count=${#gpu_ids[@]}
mkdir -p "$LEWM_LATENT_ROOT"

run_cache() {
  local gpu_id=$1
  local index=$2
  local env_name=${envs[$index]}
  local tag=${tags[$index]}
  local lewm_dir="$LEWM_RUN_ROOT/lewm_ogbench8_${tag}_e10_bs128_s3072"
  local checkpoint="$lewm_dir/weights_epoch_10.msgpack"
  local output="$LEWM_LATENT_ROOT/ogbench8_${tag}__node3_lewm_s3072_e10_z192.h5"
  local log="$LEWM_LATENT_ROOT/ogbench8_${tag}__node3_lewm_s3072_e10_z192.log"

  (
    cd "$OGBENCH_ROOT/impls"
    CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS="$JAX_PLATFORM" PYTHONUNBUFFERED=1 \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" precompute_lewm_npz_latents.py \
      --env-name="$env_name" \
      --npz-path="$OGBENCH_DATA_DIR/$env_name.npz" \
      --checkpoint="$checkpoint" \
      --output="$output" \
      --batch-size="$BATCH_SIZE" \
      --output-dtype=float32 \
      --flush-every-batches=20 \
      --log-every-batches=20 \
      --smoke-rows="$SMOKE_ROWS" \
      2>&1 | tee -a "$log"
  )
}

failed=0
for (( base=0; base<${#dataset_indices[@]}; base+=gpu_count )); do
  pids=()
  for (( slot=0; slot<gpu_count && base+slot<${#dataset_indices[@]}; slot++ )); do
    run_cache "${gpu_ids[$slot]}" "${dataset_indices[$((base + slot))]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
