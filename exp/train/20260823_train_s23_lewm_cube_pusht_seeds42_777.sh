#!/usr/bin/env bash
set -euo pipefail

# Server23：GPU2–5 并行训练 Cube/PushT × seed42/777 四个 LeWM-JAX；IMPALA-Small、e10、bs128、fs5、history3、SigReg0.09。
CLIENT_ID=23
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
FRAMESKIP=${FRAMESKIP:-5}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.09}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train cube_single_expert pusht_expert_train)
tags=(cube_single pusht_expert cube_single pusht_expert)
seeds=(42 42 777 777)
gpus=(2 3 4 5)
pids=()

for i in "${!datasets[@]}"; do
  exp_name="LeWMJAX_impala_lance_${tags[$i]}_bs${LEWM_BATCH_SIZE}_e${LEWM_EPOCH}_seed${seeds[$i]}_fs${FRAMESKIP}_h3_sigreg009_main20260823"
  run_dir="$RUN_DIR/lewm-jax-runs/$exp_name"
  task_tmp="$RUN_DIR/tmp/$exp_name"
  mkdir -p "$run_dir" "$task_tmp"
  TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_jax.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --dataset_format=lance --save_dir="$run_dir" --exp_name="$exp_name" \
    --epochs="$LEWM_EPOCH" --batch_size="$LEWM_BATCH_SIZE" --seed="${seeds[$i]}" \
    --frameskip="$FRAMESKIP" --image_size=224 --learning_rate="$LEARNING_RATE" --weight_decay=1e-3 \
    --sigreg_weight="$SIGREG_WEIGHT" --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=6 \
    >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
