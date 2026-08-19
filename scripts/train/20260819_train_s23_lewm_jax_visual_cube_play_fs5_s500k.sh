#!/usr/bin/env bash
set -euo pipefail

# Server 23：训练 LeWM-JAX IMPALA Visual Cube single/double/triple；500k steps、bs128、frameskip/action block 5，后续 CEM horizon 5。
task="${1:-}"
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

case "$task" in
  single) gpu=2 ;;
  double) gpu=3 ;;
  triple) gpu=4 ;;
  *)
    echo "Usage: bash scripts/train/20260819_train_s23_lewm_jax_visual_cube_play_fs5_s500k.sh {single|double|triple}" >&2
    exit 2
    ;;
esac

env_name="visual-cube-${task}-play-v0"
exp_name="LeWMJAX_ogbench_visual_cube_${task}_play_impalasmall_bs128_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23"
data_root=/home/dzb/.ogbench/data
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
[[ ! -e "$run_dir" ]] || { echo "ERROR: run directory already exists: $run_dir" >&2; exit 1; }
mkdir -p "$run_dir" /data/dzb/stablewm-data/lewm-jax-runs/tmp
cd "$OGBENCH_ROOT/impls"

TMPDIR=/data/dzb/stablewm-data/lewm-jax-runs/tmp \
CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" train_lewm_jax.py \
  --dataset_path="$data_root/$env_name.npz" \
  --validation_dataset_path="$data_root/$env_name-val.npz" \
  --dataset_format=npz --save_dir="$run_dir" --exp_name="$exp_name" \
  --train_steps=500000 --save_interval_steps=100000 \
  --batch_size=128 --seed=3072 --frameskip=5 --image_size=64 \
  --learning_rate=5e-5 --weight_decay=1e-3 \
  --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1 \
  2>&1 | tee "$run_dir/train.log"
