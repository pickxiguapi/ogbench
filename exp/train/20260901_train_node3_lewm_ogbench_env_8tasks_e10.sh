#!/usr/bin/env bash
set -euo pipefail

# A800 node3：八卡并行训练 OGBench-Env-8Tasks canonical LeWM-JAX；10 epochs、bs128、seed3072、fs5、history3、SigReg0.09，每个 epoch 保存 checkpoint。
CLIENT_ID=node3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LEWM_EPOCHS=${LEWM_EPOCHS:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
FRAMESKIP=${FRAMESKIP:-5}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.09}
XLA_MEMORY_FRACTION=${XLA_MEMORY_FRACTION:-0.80}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
pids=()

for i in "${!envs[@]}"; do
  exp_name="lewm_ogbench8_${tags[$i]}_e${LEWM_EPOCHS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
  run_dir="$CLIENT_ROOT/lewm-final/lewm-ogbench8/$exp_name"
  mkdir -p "$run_dir/tmp"
  TMPDIR="$run_dir/tmp" CUDA_VISIBLE_DEVICES=$i XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEMORY_FRACTION" JAX_PLATFORMS=cuda \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --validation_dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}-val.npz" \
    --dataset_format=npz --save_dir="$run_dir" --exp_name="$exp_name" \
    --epochs="$LEWM_EPOCHS" --batch_size="$LEWM_BATCH_SIZE" --seed="$LEWM_SEED" \
    --frameskip="$FRAMESKIP" --image_size=64 \
    --learning_rate="$LEARNING_RATE" --weight_decay=1e-3 \
    --sigreg_weight="$SIGREG_WEIGHT" --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1 \
    >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
