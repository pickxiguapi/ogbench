#!/usr/bin/env bash
set -euo pipefail

# node4：8 卡并行训练 LeWM 四任务 independent GCIQL-Chunk actor 对照；GPU 0–3 为标准 DDPG+BC alpha1，GPU 4–7 为 AWR alpha1，均训练 100k。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
GCIQL_DATA_ROOT=${GCIQL_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
GCIQL_RUNS_ROOT=${GCIQL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs}
P_AUG=0.5
POLICY_STEPS=100000
POLICY_BATCH_SIZE=256
POLICY_SEED=0
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom cube pusht reacher tworoom)
gpus=(0 1 2 3 4 5 6 7)
actor_losses=(ddpgbc ddpgbc ddpgbc ddpgbc awr awr awr awr)
alphas=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)
pids=()

for i in "${!datasets[@]}"; do
  exp_name="gc4_${tags[$i]}_ind_${actor_losses[$i]}_alpha1_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  run_dir="$GCIQL_RUNS_ROOT/gciql-chunk-4tasks-actor-ablation/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$GCIQL_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode=independent \
    --actor_loss="${actor_losses[$i]}" --alpha="${alphas[$i]}" \
    --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
