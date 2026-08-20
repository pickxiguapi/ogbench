#!/usr/bin/env bash
set -euo pipefail

# 英博云：在 GPU 4–7 训练指定 LeWM Task 的 LeWM-JAX IMPALA 世界模型；e10、bs128、seed42、frameskip5、SigReg0.09。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

case "${1:?Usage: bash $0 <cube|pusht|reacher|tworoom>}" in
  cube) dataset=cube_single_expert; gpu=4 ;;
  pusht) dataset=pusht_expert_train; gpu=5 ;;
  reacher) dataset=reacher; gpu=6 ;;
  tworoom) dataset=tworoom; gpu=7 ;;
  *) echo "Unknown task: $1" >&2; exit 2 ;;
esac

task=$1
exp_name="${DATE}_${CLIENT_ID}_LeWMJAX_${task}_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009"
run_dir="$CLIENT_ROOT/lewm-jax-runs/$exp_name"
mkdir -p "$run_dir/tmp"
TMPDIR="$run_dir/tmp" CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" train_lewm_jax.py \
  --dataset_path="$LEWM_DATA_ROOT/${dataset}.lance" \
  --save_dir="$run_dir" --exp_name="$exp_name" \
  --epochs=10 --batch_size=128 --seed=42 --frameskip=5 \
  --learning_rate=5e-5 --weight_decay=1e-3 \
  --sigreg_weight=0.09 --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=6
