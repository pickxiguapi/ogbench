#!/usr/bin/env bash
set -euo pipefail

# Server 23 only. Usage: .../recorded_run.sh EXP-ID EXP_NAME RUN-ID -- bash "$0" {cube|pusht|reacher|tworoom}
task="${1:?task: cube|pusht|reacher|tworoom}"
case "$task" in
  cube)    env=visual-lewm-cube-single-expert-v0; dataset=cube_single_expert.lance; gpu=0 ;;
  pusht)   env=visual-lewm-pusht-expert-train-v0; dataset=pusht_expert_train.lance; gpu=2 ;;
  reacher) env=visual-lewm-reacher-v0; dataset=reacher.lance; gpu=6 ;;
  tworoom) env=visual-lewm-tworoom-v0; dataset=tworoom.lance; gpu=4 ;;
  *) echo "unknown task: $task" >&2; exit 2 ;;
esac

: "${EXPERIMENT_RUN_ID:?launch through recorded_run.sh}" "${EXPERIMENT_EXP_NAME:?launch through recorded_run.sh}"
mkdir -p /data/dzb/lewm-runs/wandb
cd /home/dzb/ogbench/impls

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR=/data/dzb/lewm-runs/wandb \
/home/dzb/ogbench/.venv/bin/python main.py \
  --env_name="$env" --dataset_path="/data/dzb/stablewm-data/datasets/$dataset" \
  --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 \
  --agent.encoder=impala_small --agent.p_aug=0.5 \
  --train_steps=100000 --save_dir=/data/dzb/lewm-runs \
  --log_interval=5000 --save_interval=100000 \
  --run_group="$EXPERIMENT_EXP_NAME" --wandb_mode=offline --eval_episodes=0 \
  --video_episodes=0
