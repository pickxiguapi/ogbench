#!/usr/bin/env bash
set -euo pipefail

# Server 23：GPU 2–5 并行训练 LeWM 四任务的 GCIQL-Chunk AWR shared-all；Q/V/π 共用冻结的 seed3072 LeWM epoch-10 表征、heads 独立；s100k、k5、bs256、seed0、alpha3。
CLIENT_ID=23
DATE=$(date +%Y-%m-%d)
source /home/dzb/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(2 3 4 5)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)

pids=()
for i in "${!datasets[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWM_with_GCIQL_Chunk_AWR_shared_all_${tags[$i]}_lewmseed3072_k5_bs256_s100k_s0"
  run_dir="/data/dzb/stablewm-data/lewm-gciql-chunk-shared-runs/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --lewm_checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
    --save_dir="$run_dir" --share_q_encoder --share_v_encoder --share_pi_encoder \
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
