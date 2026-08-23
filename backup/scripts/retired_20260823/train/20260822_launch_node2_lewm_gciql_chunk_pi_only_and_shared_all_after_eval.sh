#!/usr/bin/env bash
set -euo pipefail

# A800 node2：评测结束后用 8 卡启动 LeWM 四任务的 π-only 与 shared-all；每个任务仅等待自身 Lance 完整，Cube/PushT 可先行。
CLIENT_ID=node2
DATE=$(date +%Y-%m-%d)
source /data-training/yyf/ogbench/scripts/client_env.sh
CODE_ROOT="$CLIENT_ROOT/ogbench-visual-policy-runs/code/ogbench-shared-policy"
DATA_ROOT="$CLIENT_ROOT/datasets/latent-geometry"
cd "$CODE_ROOT/impls"

settings=(pi_only pi_only pi_only pi_only all all all all)
datasets=(cube_single_expert pusht_expert_train reacher tworoom cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom cube pusht reacher tworoom)
checkpoints=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
sizes=(18971745447 14177663968 17202901640 4063804288 18971745447 14177663968 17202901640 4063804288)
pids=()

for i in "${!datasets[@]}"; do
  (
    while [[ $(find "$DATA_ROOT/${datasets[$i]}.lance" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}') -ne ${sizes[$i]} ]]; do sleep 30; done
    share_flags=(--share_pi_encoder)
    [[ "${settings[$i]}" == all ]] && share_flags=(--share_q_encoder --share_v_encoder --share_pi_encoder)
    run_dir="$CLIENT_ROOT/ogbench-visual-policy-runs/lewm-gciql-chunk-shared-runs/${DATE}_${CLIENT_ID}_LeWM_with_GCIQL_Chunk_AWR_shared_${settings[$i]}_${tags[$i]}_lewmseed3072_k5_bs256_s100k_s0"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES=$i XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH="$CODE_ROOT:$CODE_ROOT/impls" "$PYTHON_BIN" train_lewm_gciql_chunk.py \
      --dataset_path="$DATA_ROOT/${datasets[$i]}.lance" \
      --lewm_checkpoint="$CLIENT_ROOT/models/lewm-jax-seed3072/${checkpoints[$i]}/weights_epoch_10.msgpack" \
      --save_dir="$run_dir" "${share_flags[@]}" \
      --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug=0.5 \
      --train_steps=100000 --batch_size=256 --seed=0 --chunk_size=5 \
      --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
      --log_interval=5000 --save_interval=100000 >"$run_dir/train.log" 2>&1
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
