#!/usr/bin/env bash
set -euo pipefail

# Server 23：依次训练 Cube、PushT、Reacher、TwoRoom 的 LeWM-JAX IMPALA；e10、bs128、seed3072、frameskip5、JPEG95 Lance。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube_single pusht_expert reacher tworoom)
gpus=(2 3 4 5)

for i in "${!datasets[@]}"; do
  exp_name="LeWMJAX_impala_lance_${tags[$i]}_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
  run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" --save_dir="$run_dir" --exp_name="$exp_name" \
    --epochs=10 --batch_size=128 --seed=3072 --frameskip=5 \
    --learning_rate=5e-5 --weight_decay=1e-3 \
    --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=6
done
