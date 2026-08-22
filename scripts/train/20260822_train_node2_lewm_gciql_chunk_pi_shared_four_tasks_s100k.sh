#!/usr/bin/env bash
set -euo pipefail

# A800 node2：GPU0-3 并行训练 LeWM 四任务 GCIQL-Chunk AWR π-only；π 用冻结 seed3072 LeWM，Q/V 各用独立 IMPALA Small；s100k、k5、bs256、seed0、alpha3。
CLIENT_ID=node2
DATE=$(date +%Y-%m-%d)
source /data-training/yyf/ogbench/scripts/client_env.sh
CODE_ROOT="$CLIENT_ROOT/ogbench-visual-policy-runs/code/ogbench-shared-policy"
cd "$CODE_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
checkpoints=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
gpus=(0 1 2 3)
pids=()

for i in "${!datasets[@]}"; do
  run_dir="$CLIENT_ROOT/lewm-gciql-chunk-shared-runs/${DATE}_${CLIENT_ID}_LeWM_with_GCIQL_Chunk_AWR_shared_pi_only_${tags[$i]}_lewmseed3072_k5_bs256_s100k_s0"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$CODE_ROOT:$CODE_ROOT/impls" "$PYTHON_BIN" train_lewm_gciql_chunk.py \
    --dataset_path="$CLIENT_ROOT/datasets/latent-geometry/${datasets[$i]}.lance" \
    --lewm_checkpoint="$CLIENT_ROOT/models/lewm-jax-seed3072/${checkpoints[$i]}/weights_epoch_10.msgpack" \
    --save_dir="$run_dir" --share_pi_encoder \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug=0.5 \
    --train_steps=100000 --batch_size=256 --seed=0 --chunk_size=5 \
    --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
