#!/usr/bin/env bash
set -euo pipefail

# Server 23：依次训练 Cube、PushT、Reacher、TwoRoom 的 LeWM GCIQL 基线；s100k、bs256、seed0、IMPALA Small、p_aug0.5。
CLIENT_ID=23
DATE=$(date +%Y-%m-%d)
source /home/dzb/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-lewm-cube-single-expert-v0 visual-lewm-pusht-expert-train-v0 visual-lewm-reacher-v0 visual-lewm-tworoom-v0)
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube_single pusht_expert reacher tworoom)
gpus=(0 2 6 4)

for i in "${!datasets[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWM_gciql_lance_${tags[$i]}_bs256_s100k_seed0"
  run_dir="${RUN_DIR}/gciql/$exp_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir" \
  "$PYTHON_BIN" main.py \
    --env_name="${envs[$i]}" --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 \
    --agent.encoder=impala_small --agent.p_aug=0.5 \
    --train_steps=100000 --seed=0 --save_dir="$run_dir" \
    --log_interval=5000 --save_interval=100000 \
    --run_group="$exp_name" --wandb_mode=offline --eval_episodes=0 --video_episodes=0
done
