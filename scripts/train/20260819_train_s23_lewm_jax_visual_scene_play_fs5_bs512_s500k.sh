#!/usr/bin/env bash
set -euo pipefail

# Server 23 GPU 5：训练 LeWM-JAX IMPALA Visual Scene play；500k steps、bs512、frameskip/action block 5，后续 CEM horizon 5。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

env_name=visual-scene-play-v0
exp_name=LeWMJAX_ogbench_visual_scene_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23
data_root=/home/dzb/.ogbench/data
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
[[ ! -e "$run_dir" ]] || { echo "ERROR: run directory already exists: $run_dir" >&2; exit 1; }
mkdir -p "$run_dir" /data/dzb/stablewm-data/lewm-jax-runs/tmp
cd "$OGBENCH_ROOT/impls"

TMPDIR=/data/dzb/stablewm-data/lewm-jax-runs/tmp \
CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" train_lewm_jax.py \
  --dataset_path="$data_root/$env_name.npz" \
  --validation_dataset_path="$data_root/$env_name-val.npz" \
  --dataset_format=npz --save_dir="$run_dir" --exp_name="$exp_name" \
  --train_steps=500000 --save_interval_steps=100000 \
  --batch_size=512 --seed=3072 --frameskip=5 --image_size=64 \
  --learning_rate=5e-5 --weight_decay=1e-3 \
  --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1 \
  2>&1 | tee "$run_dir/train.log"
