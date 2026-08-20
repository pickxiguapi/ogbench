#!/usr/bin/env bash
set -euo pipefail

# 英博云：依次训练 Cube Single/Double/Triple 与 Scene Play 的 LeWM-JAX IMPALA 世界模型；s500k、bs512、seed3072、frameskip/action-block5、SigReg0.09。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0)
tags=(cube_single cube_double cube_triple scene)
gpus=(0 1 2 3)

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWMJAX_${tags[$i]}_play_impalasmall_bs512_s500k_s3072_fs5_h3_sigreg009"
  run_dir="$CLIENT_ROOT/lewm-jax-runs/$exp_name"
  mkdir -p "$run_dir/tmp"
  TMPDIR="$run_dir/tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --validation_dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}-val.npz" \
    --dataset_format=npz --save_dir="$run_dir" --exp_name="$exp_name" \
    --train_steps=500000 --save_interval_steps=100000 \
    --batch_size=512 --seed=3072 --frameskip=5 --image_size=64 \
    --learning_rate=5e-5 --weight_decay=1e-3 \
    --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1
done
