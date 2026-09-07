#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/configs/lewmpp_paths.env"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
EVAL_SEEDS=(0 1 42)
LEWM_CHECKPOINTS=("$LEWM_CUBE_CHECKPOINT" "$LEWM_PUSHT_CHECKPOINT" "$LEWM_REACHER_CHECKPOINT" "$LEWM_TWOROOM_CHECKPOINT")

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/impls"

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    for seed in "${EVAL_SEEDS[@]}"; do
      result_dir="$OUTPUT_ROOT/paper_baseline_lewm_h75/general_uniform_future/no_generator/H75/lewm/zero/seed${seed}/$task"
      mkdir -p "$result_dir"
      test ! -e "$result_dir/result.json"
      args=(
        --task="$task"
        --variant=lewm
        --experiment-group=paper_baseline_lewm_h75
        --generator-family=no_generator
        --data-root="$LEWM_DATA_ROOT"
        --lewm-checkpoint="${LEWM_CHECKPOINTS[$index]}"
        --action-prior-mode=zero
        --generator-num-samples=1
        --num-eval=50
        --seed="$seed"
        --goal-offset-steps=75
        --eval-budget=150
        --cem-horizon=2
        --cem-receding-horizon=1
        --action-block=5
        --cem-num-samples=300
        --cem-iterations=5
        --cem-topk=30
        --cem-var-scale=1.0
        --cem-min-std=0.001
        --cem-cost-mode=last
        --output="$result_dir/result.json"
      )
      "$PYTHON_BIN" eval_lewm_4tasks.py "${args[@]}" --validate-only
      "$PYTHON_BIN" eval_lewm_4tasks.py "${args[@]}" 2>&1 | tee "$result_dir/eval.log"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
