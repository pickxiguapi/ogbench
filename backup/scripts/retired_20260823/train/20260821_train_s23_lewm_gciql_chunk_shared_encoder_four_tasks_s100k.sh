#!/usr/bin/env bash
set -euo pipefail

# Server 23：四卡并行训练 LeWM 四任务的 shared-encoder GCIQL-Chunk evaluator；冻结 LeWM IMPALA+projector 和现有 AWR actor，仅训练 192-D latent 上的 V/twin-Q heads；k5、s100k、bs256、seed0、expectile0.9。
CLIENT_ID=23
source /home/dzb/ogbench-shared-q/scripts/client_env.sh
OGBENCH_ROOT=/home/dzb/ogbench-shared-q
PYTHON_BIN=/home/dzb/ogbench/.venv/bin/python
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
gpus=(0 3 4 5)
output_root=/data/dzb/stablewm-data/gciql-chunk-lewm-shared-runs

pids=()
for i in "${!tasks[@]}"; do
  run_dir="$output_root/20260821_s23_${tasks[$i]}_lewm_shared_qv_gciql_chunk_awr_actor_k5_bs256_s100k_s0_e09"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_lewm_with_gciql_chunk.py \
    --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --lewm_checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
    --actor_checkpoint_dir="/data/dzb/stablewm-data/gciql-chunk-proposals-s11/${tasks[$i]}" \
    --actor_checkpoint_step=100000 \
    --save_dir="$run_dir" \
    --train_steps=100000 --save_interval=100000 --log_interval=5000 \
    --batch_size=256 --seed=0 --lr=3e-4 --discount=0.99 \
    --expectile=0.9 --tau=0.005 --chunk_size=5 \
    >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
