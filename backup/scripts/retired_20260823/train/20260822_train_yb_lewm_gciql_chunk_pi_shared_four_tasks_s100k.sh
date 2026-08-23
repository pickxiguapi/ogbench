#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU 4–7 并行训练 LeWM 四任务的 GCIQL-Chunk AWR；仅 π 使用从 Server 23 同步的 seed3072 冻结 LeWM post-projector 表征，Q/V 保留独立 IMPALA Small；s100k、k5、bs256、seed0、alpha3、像素分支 p_aug0.5。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
lewm_checkpoints=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
gpus=(4 5 6 7)

pids=()
for i in "${!datasets[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWM_with_GCIQL_Chunk_AWR_shared_pi_only_${tags[$i]}_k5_bs256_s100k_s0"
  run_dir="$CLIENT_ROOT/lewm-gciql-chunk-shared-runs/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --lewm_checkpoint="$CLIENT_ROOT/lewm-jax-seed3072-s23/${lewm_checkpoints[$i]}" \
    --save_dir="$run_dir" --share_pi_encoder \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug=0.5 \
    --train_steps=100000 --batch_size=256 --seed=0 --chunk_size=5 \
    --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 \
    >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
