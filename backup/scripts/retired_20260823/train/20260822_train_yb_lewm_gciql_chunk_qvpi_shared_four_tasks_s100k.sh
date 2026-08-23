#!/usr/bin/env bash
set -euo pipefail

# 英博云：依次训练 LeWM 四任务的 GCIQL-Chunk AWR；Q/V/π 全部使用同一冻结 LeWM post-projector 表征但保留独立 heads；s100k、k5、bs256、seed0、alpha3。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(4 5 6 7)
lewm_exps=(
  2026-08-21_yb_LeWMJAX_cube_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009
  2026-08-21_yb_LeWMJAX_pusht_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009
  2026-08-21_yb_LeWMJAX_reacher_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009
  2026-08-21_yb_LeWMJAX_tworoom_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009
)

for i in "${!datasets[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWM_with_GCIQL_Chunk_AWR_shared_all_${tags[$i]}_k5_bs256_s100k_s0"
  run_dir="$CLIENT_ROOT/lewm-gciql-chunk-shared-runs/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --lewm_checkpoint="$CLIENT_ROOT/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_9.msgpack" \
    --save_dir="$run_dir" --share_q_encoder --share_v_encoder --share_pi_encoder \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug=0.5 \
    --train_steps=100000 --batch_size=256 --seed=0 --chunk_size=5 \
    --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 \
    >"$run_dir/train.log" 2>&1
done
