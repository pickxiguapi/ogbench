#!/usr/bin/env bash
set -euo pipefail

# node4: two additional four-task DDPG+BC seed groups. GPU 0-3 use seed 42 and GPU 4-7 use seed 777.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
GCIQL_DATA_ROOT=${GCIQL_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
GCIQL_RUNS_ROOT=${GCIQL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs}
P_AUG=0.5
POLICY_STEPS=100000
POLICY_BATCH_SIZE=256
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

datasets=(cube_single_expert pusht_expert_train reacher tworoom cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom cube pusht reacher tworoom)
gpus=(0 1 2 3 4 5 6 7)
seeds=(42 42 42 42 777 777 777 777)
pids=()

for i in "${!datasets[@]}"; do
  seed=${seeds[$i]}
  exp_name="gc4_${tags[$i]}_ind_ddpgbc_alpha1_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${seed}"
  run_dir="$GCIQL_RUNS_ROOT/gciql-chunk-4tasks-actor-ablation/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$GCIQL_DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode=independent \
    --actor_loss=ddpgbc --alpha=1.0 \
    --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$seed" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
