#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/configs/lewmpp_paths.env"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
EVAL_SEEDS=(0 1 42)
LEWM_CHECKPOINTS=("$LEWM_CUBE_CHECKPOINT" "$LEWM_PUSHT_CHECKPOINT" "$LEWM_REACHER_CHECKPOINT" "$LEWM_TWOROOM_CHECKPOINT")
ACTION_PRIOR_CHECKPOINTS=("$ACTION_PRIOR_CUBE_CHECKPOINT_DIR" "$ACTION_PRIOR_PUSHT_CHECKPOINT_DIR" "$ACTION_PRIOR_REACHER_CHECKPOINT_DIR" "$ACTION_PRIOR_TWOROOM_CHECKPOINT_DIR")
SUBGOAL_CHECKPOINTS=("$GOALMAX25_CUBE_CHECKPOINT" "$GOALMAX25_PUSHT_CHECKPOINT" "$GOALMAX25_REACHER_CHECKPOINT" "$GOALMAX25_TWOROOM_CHECKPOINT")

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/impls"

for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  "$PYTHON_BIN" "$REPO_ROOT/experiments/validate_generator_checkpoint.py" \
    --checkpoint="${SUBGOAL_CHECKPOINTS[$index]}" \
    --task="$task" \
    --family=goalmax25 \
    --generator-type=latent_path_flow \
    --goal-sampling=uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25 \
    --max-goal-steps=25
done

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    for seed in "${EVAL_SEEDS[@]}"; do
      full_dir="$EXPERIMENT_ROOT/evals/lewm-4tasks/paper_h25_subgoal_path_block/goalmax25/latent_path_flow/H25/full/policy_mode/seed${seed}/$task"
      ablation_dir="$EXPERIMENT_ROOT/evals/lewm-4tasks/paper_h25_subgoal_path_block/goalmax25_protocol/no_generator/H25/no_subgoal/policy_mode/seed${seed}/$task"
      mkdir -p "$full_dir" "$ablation_dir"
      test ! -e "$full_dir/result.json"
      test ! -e "$ablation_dir/result.json"
      full_args=(
        --task="$task" --variant=full --experiment-group=paper_h25_subgoal_path_block
        --generator-family=goalmax25 --generator-type=latent_path_flow
        --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="${LEWM_CHECKPOINTS[$index]}"
        --action-prior-checkpoint-dir="${ACTION_PRIOR_CHECKPOINTS[$index]}"
        --action-prior-checkpoint-step=100000 --action-prior-mode=policy_mode
        --action-prior-representation-mode=all
        --subgoal-generator-checkpoint="${SUBGOAL_CHECKPOINTS[$index]}"
        --flow-sampling-steps=16 --generator-num-samples=1
        --num-eval=50 --seed="$seed" --goal-offset-steps=25 --eval-budget=50
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30
        --cem-var-scale=1.0 --cem-min-std=0.001 --cem-cost-mode=moh
        --output="$full_dir/result.json"
      )
      ablation_args=(
        --task="$task" --variant=no_subgoal --experiment-group=paper_h25_subgoal_path_block
        --generator-family=no_generator
        --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="${LEWM_CHECKPOINTS[$index]}"
        --action-prior-checkpoint-dir="${ACTION_PRIOR_CHECKPOINTS[$index]}"
        --action-prior-checkpoint-step=100000 --action-prior-mode=policy_mode
        --action-prior-representation-mode=all
        --generator-num-samples=1
        --num-eval=50 --seed="$seed" --goal-offset-steps=25 --eval-budget=50
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30
        --cem-var-scale=1.0 --cem-min-std=0.001 --cem-cost-mode=moh
        --output="$ablation_dir/result.json"
      )
      "$PYTHON_BIN" eval_lewm_4tasks.py "${full_args[@]}" --validate-only
      "$PYTHON_BIN" eval_lewm_4tasks.py "${ablation_args[@]}" --validate-only
      "$PYTHON_BIN" eval_lewm_4tasks.py "${full_args[@]}" 2>&1 | tee "$full_dir/eval.log"
      "$PYTHON_BIN" eval_lewm_4tasks.py "${ablation_args[@]}" 2>&1 | tee "$ablation_dir/eval.log"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
