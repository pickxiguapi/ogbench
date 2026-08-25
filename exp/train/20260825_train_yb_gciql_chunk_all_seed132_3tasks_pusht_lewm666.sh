#!/usr/bin/env bash
set -euo pipefail

# 英博云 GPU5–7：并行训练 Cube、PushT、Reacher 三任务 shared-all GCIQL-Chunk-AWR policy seed132；PushT 使用 LeWM seed666，其余任务使用 LeWM seed3072；100k、k5、bs256、alpha3、无增强。
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
POLICY_SEED=132
POLICY_STEPS=100000
POLICY_BATCH_SIZE=256
P_AUG=0.0
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

lewm_seed3072_root=/root/data/yyf/lewm-jax-seed3072-s23
lewm_seed666_root=/root/data/yyf/lewm-jax-seed666-s23
cube_lewm="$lewm_seed3072_root/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
pusht_lewm="$lewm_seed666_root/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"
reacher_lewm="$lewm_seed3072_root/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"

datasets=(cube_single_expert pusht_expert_train reacher)
tags=(cube pusht reacher)
gpus=(5 6 7)
lewm_checkpoints=("$cube_lewm" "$pusht_lewm" "$reacher_lewm")

for checkpoint in "${lewm_checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Frozen LeWM checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done

pids=()
for i in "${!datasets[@]}"; do
  exp_name="gc4_${tags[$i]}_all_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  run_dir="$GCIQL_RUNS_ROOT/gciql-chunk-4tasks/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$GCIQL_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode=all \
    --lewm_checkpoint="${lewm_checkpoints[$i]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
