#!/usr/bin/env bash
set -euo pipefail

# A800 node2：GPU0-7 并行评测 state-dependent std、train seed0 的八个 Visual Play/Noisy GCIQL-Chunk AWR 任务；300k/400k/500k 各 50 episodes，eval seed42。
CLIENT_ID=node2
DATE=$(date +%Y-%m-%d)
source /data-training/yyf/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

TRAIN_ROOT="$CLIENT_ROOT/ogbench-visual-policy-runs"
EVAL_ROOT="$CLIENT_ROOT/ogbench-visual-policy-evals/${DATE}_node2_GCAWR_sdepstd_trainseed0_evalseed42_s300k_s400k_s500k_ep50"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
gpus=(0 1 2 3 4 5 6 7)
steps=(300000 400000 500000)

mkdir -p "$EVAL_ROOT"

eval_env() {
  local env_name=$1 tag=$2 gpu=$3
  local train_run="$TRAIN_ROOT/2026-08-22_node2_GCAWR_${tag}_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd"
  local restore_dir
  restore_dir=$(find "$train_run" -type f -name params_500000.pkl -printf '%h\n' | sort | tail -1)

  for step in "${steps[@]}"; do
    local step_root="$EVAL_ROOT/$tag/s${step}"
    mkdir -p "$step_root/wandb" "$step_root/tmp"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TMPDIR="$step_root/tmp" CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_DIR="$step_root/wandb" "$PYTHON_BIN" main.py \
      --env_name="$env_name" --dataset_path="$OGBENCH_DATA_DIR/${env_name}.npz" \
      --agent=agents/gciql_chunk.py --agent.actor_loss=awr --agent.alpha=3.0 \
      --agent.chunk_size=5 --agent.batch_size=512 --agent.lr=3e-4 --agent.discount=0.99 \
      --agent.expectile=0.9 --agent.tau=0.005 --agent.encoder=impala_small --agent.p_aug=0.5 \
      --agent.state_dependent_std=True --agent.const_std=False \
      --restore_path="$restore_dir" --restore_epoch="$step" --eval_only=True \
      --save_dir="$step_root" --run_group="node2_eval_sdepstd_${tag}_s${step}_ep50_seed42" \
      --wandb_mode=offline --seed=42 --eval_episodes=50 --eval_on_cpu=0 --video_episodes=0 \
      >"$step_root/eval.log" 2>&1
  done
}

pids=()
for i in "${!envs[@]}"; do
  eval_env "${envs[$i]}" "${tags[$i]}" "${gpus[$i]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON_BIN" - "$EVAL_ROOT" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
steps = (300000, 400000, 500000)
rows = []
for tag_dir in sorted(glob.glob(os.path.join(root, '*'))):
    if not os.path.isdir(tag_dir):
        continue
    tag = os.path.basename(tag_dir)
    values = []
    for step in steps:
        path = sorted(glob.glob(os.path.join(tag_dir, f's{step}', '**', 'eval.csv'), recursive=True))[-1]
        with open(path, newline='') as f:
            values.append(float(list(csv.DictReader(f))[-1]['evaluation/overall_success']))
    rows.append([tag, *values, sum(values) / len(values)])
with open(os.path.join(root, 'summary.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['task', 'success_300000', 'success_400000', 'success_500000', 'mean_success'])
    writer.writerows(rows)
PY
