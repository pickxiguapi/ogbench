#!/usr/bin/env bash
set -euo pipefail

# 英博云：依次训练 Cube、PushT、Reacher、TwoRoom 的 LeWM GCIQL 基线；s100k、bs256、seed0、IMPALA Small、p_aug0.5。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-lewm-cube-single-expert-v0 visual-lewm-pusht-expert-train-v0 visual-lewm-reacher-v0 visual-lewm-tworoom-v0)
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_GCIQL_lewm_${tags[$i]}_bs256_s100k_s0_a1_e09_aug05"
  run_dir="$CLIENT_ROOT/ogbench-lewm-policy-runs/$exp_name"
  mkdir -p "$run_dir/wandb" "$run_dir/tmp"
  MUJOCO_GL=egl LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TMPDIR="$run_dir/tmp" \
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir/wandb" \
  "$PYTHON_BIN" main.py \
    --env_name="${envs[$i]}" --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --agent=agents/gciql.py --agent.actor_loss=ddpgbc --agent.alpha=1.0 --agent.batch_size=256 \
    --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.9 --agent.tau=0.005 \
    --agent.encoder=impala_small --agent.p_aug=0.5 \
    --train_steps=100000 --seed=0 --save_dir="$run_dir" \
    --log_interval=5000 --save_interval=100000 --run_group="$exp_name" \
    --wandb_mode=offline --eval_episodes=0 --video_episodes=0
done
