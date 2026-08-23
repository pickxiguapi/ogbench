#!/usr/bin/env bash
set -euo pipefail

# A800 node4：GPU 0/1/5 追加 PushT GCIQL-Chunk AWR 的 seed0/42/123 稳定性复核；s100k、k5、bs256、alpha3。
CLIENT_ID=node4
DATE=$(date +%Y-%m-%d)
source /data-training/yyf/ogbench-lewm-policy-runs/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

seeds=(0 42 123)
gpus=(0 1 5)
pids=()

for i in "${!seeds[@]}"; do
  seed=${seeds[$i]}
  exp_name="${DATE}_${CLIENT_ID}_GCAWR_lewm_pusht_k5_bs256_s100k_s${seed}_a3_e09_aug05"
  run_dir="$CLIENT_ROOT/ogbench-lewm-policy-runs/$exp_name"
  mkdir -p "$run_dir/wandb" "$run_dir/tmp"
  (
    MUJOCO_GL=egl LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TMPDIR="$run_dir/tmp" \
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir/wandb" \
    "$PYTHON_BIN" main.py \
      --env_name=visual-lewm-pusht-expert-train-v0 --dataset_path="$LEWM_DATA_ROOT/pusht_expert_train.lance" \
      --agent=agents/gciql_chunk.py --agent.actor_loss=awr --agent.alpha=3.0 \
      --agent.chunk_size=5 --agent.batch_size=256 --agent.lr=3e-4 --agent.discount=0.99 \
      --agent.expectile=0.9 --agent.tau=0.005 --agent.encoder=impala_small --agent.p_aug=0.5 \
      --train_steps=100000 --seed="$seed" --save_dir="$run_dir" \
      --log_interval=5000 --save_interval=100000 \
      --run_group="node4_GCAWR_lewm_pusht_s${seed}" \
      --wandb_mode=offline --eval_episodes=0 --video_episodes=0
  ) >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
