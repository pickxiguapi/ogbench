#!/usr/bin/env bash
set -euo pipefail

# Server 23：LeWM-JAX IMPALA 四任务训练，e10、bs128、seed3072、frameskip5、JPEG95 Lance。
task=$1
case "$task" in
  cube) dataset=cube_single_expert; tag=cube_single; gpu=2 ;;
  pusht) dataset=pusht_expert_train; tag=pusht_expert; gpu=3 ;;
  reacher) dataset=reacher; tag=reacher; gpu=4 ;;
  tworoom) dataset=tworoom; tag=tworoom; gpu=5 ;;
esac

CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
exp_name="LeWMJAX_impala_lance_${tag}_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95"
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
mkdir -p "$run_dir"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
"$PYTHON_BIN" train_lewm_jax.py \
  --dataset_path="$LEWM_DATA_ROOT/$dataset.lance" --save_dir="$run_dir" --exp_name="$exp_name" \
  --epochs=10 --batch_size=128 --seed=3072 --frameskip=5 \
  --learning_rate=5e-5 --weight_decay=1e-3 \
  --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=6
