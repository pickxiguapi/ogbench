#!/usr/bin/env bash
set -euo pipefail

# A800 node2：8 卡并行训练 Visual Cube Single/Double/Triple 与 Scene 的 Play/Noisy 八个 LeWM-JAX；e10、bs128、seed3072、frameskip5/history3、SigReg0.09。
CLIENT_ID=node2
DATE=$(date +%Y-%m-%d)
source /data-training/yyf/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
gpus=(0 1 2 3 4 5 6 7)
pids=()

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWMJAX_${tags[$i]}_npz_impalasmall_bs128_e10_s3072_fs5_h3_sigreg009"
  run_dir="$CLIENT_ROOT/lewm-jax-visual-runs/$exp_name"
  mkdir -p "$run_dir/tmp"
  TMPDIR="$run_dir/tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 JAX_PLATFORMS=cuda \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --validation_dataset_path="$OGBENCH_DATA_DIR/${envs[$i]}-val.npz" \
    --dataset_format=npz --save_dir="$run_dir" --exp_name="$exp_name" \
    --epochs=10 --batch_size=128 --seed=3072 --frameskip=5 --image_size=64 \
    --learning_rate=5e-5 --weight_decay=1e-3 \
    --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1 \
    >"$run_dir/train.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched gpu=${gpus[$i]} env=${envs[$i]} pid=$pid log=$run_dir/train.log"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
