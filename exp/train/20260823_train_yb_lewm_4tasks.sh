#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行训练 LeWM-4Tasks 的 canonical LeWM-JAX；默认 e10、bs128、seed3072、fs5、SigReg0.09，参数可在下方修改。
CLIENT_ID=yb
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
FRAMESKIP=${FRAMESKIP:-5}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.09}
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
pids=()

for i in "${!datasets[@]}"; do
  exp_name="lewm_4tasks_${tags[$i]}_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
  run_dir="$CLIENT_ROOT/lewm-final/lewm-4tasks/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --dataset_format=lance --save_dir="$run_dir" --exp_name="$exp_name" \
    --epochs="$LEWM_EPOCH" --batch_size="$LEWM_BATCH_SIZE" --seed="$LEWM_SEED" \
    --frameskip="$FRAMESKIP" --image_size=224 --learning_rate="$LEARNING_RATE" --weight_decay=1e-3 \
    --sigreg_weight="$SIGREG_WEIGHT" --sigreg_knots=17 --sigreg_num_proj=1024 \
    >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
