#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzb/ogbench
PYTHON="$ROOT/.venv/bin/python"
MAIN="$ROOT/impls/main.py"
SCRIPT="$ROOT/scripts/eval/20260810_legacy_eval_hiql_chunk_visual_s300k_s400k_s500k.sh"
EVAL_ROOT=/data/dzb/ogbench-evals/hiql_chunk
LOG_ROOT="$ROOT/logs/hiql_chunk_eval"

run_worker() {
  local task="$1"
  local gpu env_name run_name restore_path
  local -a agent_args

  case "$task" in
    antmaze_large)
      gpu=2
      env_name=visual-antmaze-large-navigate-v0
      run_name=visual-antmaze-large
      restore_path=/data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-large-navigate/OGBench/hiql-chunk-visual-antmaze-large-bs512/sd000_20260809_113150
      agent_args=(
        --agent.discount=0.99 --agent.subgoal_steps=25 --agent.chunk_size=2
        --agent.expectile=0.5 --agent.p_aug=0.0
      )
      ;;
    antmaze_giant)
      gpu=3
      env_name=visual-antmaze-giant-navigate-v0
      run_name=visual-antmaze-giant
      restore_path=/data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-giant-navigate/OGBench/hiql-chunk-visual-antmaze-giant-bs512/sd000_20260809_113150
      agent_args=(
        --agent.discount=0.995 --agent.subgoal_steps=25 --agent.chunk_size=2
        --agent.expectile=0.5 --agent.p_aug=0.0
      )
      ;;
    cube_double)
      gpu=4
      env_name=visual-cube-double-play-v0
      run_name=visual-cube-double
      restore_path=/data/dzb/ogbench-runs/hiql_chunk/visual-cube-double-play/OGBench/hiql-chunk-visual-cube-double-bs512/sd000_20260809_113150
      agent_args=(
        --agent.discount=0.99 --agent.subgoal_steps=10 --agent.chunk_size=5
        --agent.expectile=0.93 --agent.p_aug=0.5
      )
      ;;
    humanoidmaze_medium)
      gpu=5
      env_name=visual-humanoidmaze-medium-navigate-v0
      run_name=visual-humanoidmaze-medium
      restore_path=/data/dzb/ogbench-runs/hiql_chunk/visual-humanoidmaze-medium-navigate/OGBench/hiql-chunk-visual-humanoidmaze-medium-bs512/sd000_20260809_113150
      agent_args=(
        --agent.discount=0.995 --agent.subgoal_steps=100 --agent.chunk_size=5
        --agent.expectile=0.5 --agent.p_aug=0.0
      )
      ;;
    *)
      echo "Unknown task: $task" >&2
      exit 2
      ;;
  esac

  mkdir -p "$EVAL_ROOT/$run_name" "$LOG_ROOT"
  exec > >(tee -a "$LOG_ROOT/$run_name.log") 2>&1

  for checkpoint in 300000 400000 500000; do
    echo "[$(date '+%F %T')] Evaluating $env_name checkpoint $checkpoint on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
      "$PYTHON" "$MAIN" \
      --env_name="$env_name" \
      --agent="$ROOT/impls/agents/hiql_chunk.py" \
      --agent.batch_size=512 --agent.encoder=impala_small \
      --agent.lr=0.0003 --agent.tau=0.005 \
      --agent.high_alpha=3.0 --agent.low_alpha=3.0 \
      --agent.low_actor_rep_grad=True --agent.const_std=True \
      "${agent_args[@]}" \
      --restore_path="$restore_path" --restore_epoch="$checkpoint" --eval_only=True \
      --eval_episodes=50 --video_episodes=0 --eval_on_cpu=0 --seed=0 \
      --wandb_mode=online \
      --run_group="eval-hiql-chunk-$run_name-bs512-step$checkpoint" \
      --save_dir="$EVAL_ROOT/$run_name"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker "$2"
  exit 0
fi

for checkpoint_dir in \
  /data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-large-navigate/OGBench/hiql-chunk-visual-antmaze-large-bs512/sd000_20260809_113150 \
  /data/dzb/ogbench-runs/hiql_chunk/visual-antmaze-giant-navigate/OGBench/hiql-chunk-visual-antmaze-giant-bs512/sd000_20260809_113150 \
  /data/dzb/ogbench-runs/hiql_chunk/visual-cube-double-play/OGBench/hiql-chunk-visual-cube-double-bs512/sd000_20260809_113150 \
  /data/dzb/ogbench-runs/hiql_chunk/visual-humanoidmaze-medium-navigate/OGBench/hiql-chunk-visual-humanoidmaze-medium-bs512/sd000_20260809_113150; do
  for checkpoint in 300000 400000 500000; do
    test -f "$checkpoint_dir/params_$checkpoint.pkl"
  done
done

mkdir -p "$EVAL_ROOT" "$LOG_ROOT"

tmux new-session -d -s eval-hiql-chunk-antmaze-large -c "$ROOT" \
  "bash '$SCRIPT' --worker antmaze_large"
tmux new-session -d -s eval-hiql-chunk-antmaze-giant -c "$ROOT" \
  "bash '$SCRIPT' --worker antmaze_giant"
tmux new-session -d -s eval-hiql-chunk-cube-double -c "$ROOT" \
  "bash '$SCRIPT' --worker cube_double"
tmux new-session -d -s eval-hiql-chunk-humanoidmaze-medium -c "$ROOT" \
  "bash '$SCRIPT' --worker humanoidmaze_medium"

tmux ls | grep '^eval-hiql-chunk-'
