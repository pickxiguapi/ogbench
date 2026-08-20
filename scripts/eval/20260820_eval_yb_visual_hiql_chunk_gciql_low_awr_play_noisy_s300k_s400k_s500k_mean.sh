#!/usr/bin/env bash
set -euo pipefail

# 英博云：8 卡并行评测 HIQL-Chunk-GCIQL-Low-AWR；每卡依次评测 s300k、s400k、s500k，各 50 episodes，并汇总 checkpoint mean。
CLIENT_ID=yb
TRAIN_DATE=2026-08-19
EVAL_DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

TRAIN_ROOT="$CLIENT_ROOT/ogbench-hiql-chunk-gciql-low-awr-runs"
EVAL_ROOT="$CLIENT_ROOT/ogbench-hiql-chunk-gciql-low-awr-evals/${EVAL_DATE}_b512_s300k_s400k_s500k_mean"
EGL_LIB_DIR="$CLIENT_ROOT/egl-runtime/root/usr/lib/x86_64-linux-gnu"
EVAL_SEED=42
EVAL_EPISODES=50

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
gpus=(0 1 2 3 4 5 6 7)
steps=(300000 400000 500000)

mkdir -p "$EVAL_ROOT"

eval_env() {
  local env_name=$1 tag=$2 gpu=$3
  local train_group="${TRAIN_DATE}_${CLIENT_ID}_HCGLAWR_${tag}_k5sg10_b512_500k_s0"
  local train_group_dir="$TRAIN_ROOT/$train_group/OGBench/$train_group"
  local restore_dir
  restore_dir=$(find "$train_group_dir" -mindepth 1 -maxdepth 1 -type d -name 'sd000_*' | sort | tail -1)
  [[ -n "$restore_dir" ]] || { echo "ERROR: training run not found: $train_group_dir" >&2; return 1; }

  for step in "${steps[@]}"; do
    local checkpoint="$restore_dir/params_${step}.pkl"
    local eval_group="${EVAL_DATE}_${CLIENT_ID}_EHCGA_${tag}_${step}_e${EVAL_EPISODES}_s${EVAL_SEED}"
    local step_root="$EVAL_ROOT/$tag/s${step}"
    local existing_csv
    [[ -s "$checkpoint" ]] || { echo "ERROR: checkpoint not found: $checkpoint" >&2; return 1; }
    mkdir -p "$step_root/wandb" "$step_root/tmp"

    existing_csv=$(find "$step_root" -type f -name eval.csv -size +0c -print 2>/dev/null | sort | tail -1)
    if [[ -n "$existing_csv" ]]; then
      echo "SKIP: $tag s$step already evaluated: $existing_csv"
      continue
    fi

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TMPDIR="$step_root/tmp" CUDA_VISIBLE_DEVICES="$gpu" WANDB_DIR="$step_root/wandb" \
    "$PYTHON_BIN" main.py \
      --env_name="$env_name" --agent=agents/hiql_chunk.py \
      --agent.chunk_size=5 --agent.subgoal_steps=10 --agent.batch_size=512 \
      --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.7 --agent.low_expectile=0.9 \
      --agent.tau=0.005 --agent.high_alpha=3.0 --agent.low_alpha=3.0 \
      --agent.encoder=impala_small --agent.low_actor_rep_grad=True --agent.p_aug=0.5 \
      --restore_path="$restore_dir" --restore_epoch="$step" --eval_only=True \
      --save_dir="$step_root" --run_group="$eval_group" --wandb_mode=offline \
      --seed="$EVAL_SEED" --eval_episodes="$EVAL_EPISODES" --eval_on_cpu=0 --video_episodes=0 \
      > "$step_root/eval.log" 2>&1
  done
}

pids=()
for i in "${!envs[@]}"; do
  eval_env "${envs[$i]}" "${tags[$i]}" "${gpus[$i]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "ERROR: at least one evaluation failed" >&2; exit 1; }

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
        paths = sorted(glob.glob(os.path.join(tag_dir, f's{step}', '**', 'eval.csv'), recursive=True))
        if not paths:
            raise SystemExit(f'ERROR: eval.csv missing for {tag} s{step}')
        with open(paths[-1], newline='') as f:
            records = list(csv.DictReader(f))
        if not records or 'evaluation/overall_success' not in records[-1]:
            raise SystemExit(f'ERROR: overall_success missing in {paths[-1]}')
        values.append(float(records[-1]['evaluation/overall_success']))
    rows.append([tag, *values, sum(values) / len(values)])

summary = os.path.join(root, 'summary.csv')
with open(summary, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['task', 'success_300000', 'success_400000', 'success_500000', 'mean_success'])
    writer.writerows(rows)
print(summary)
for row in rows:
    print(f'{row[0]}: 300k={row[1]:.4f}, 400k={row[2]:.4f}, 500k={row[3]:.4f}, mean={row[4]:.4f}')
PY
