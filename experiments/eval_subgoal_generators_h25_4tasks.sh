#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/configs/lewmpp_paths.env"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
EVAL_SEEDS=(0 1 42)
GENERATOR_TYPES=(mlp endpoint_flow latent_path_flow)
LEWM_CHECKPOINTS=("$LEWM_CUBE_CHECKPOINT" "$LEWM_PUSHT_CHECKPOINT" "$LEWM_REACHER_CHECKPOINT" "$LEWM_TWOROOM_CHECKPOINT")
ACTION_PRIOR_CHECKPOINTS=("$ACTION_PRIOR_CUBE_CHECKPOINT_DIR" "$ACTION_PRIOR_PUSHT_CHECKPOINT_DIR" "$ACTION_PRIOR_REACHER_CHECKPOINT_DIR" "$ACTION_PRIOR_TWOROOM_CHECKPOINT_DIR")
SUBGOAL_CHECKPOINTS=(
  "$GOALMAX25_MLP_CUBE_CHECKPOINT" "$GOALMAX25_ENDPOINT_FLOW_CUBE_CHECKPOINT" "$GOALMAX25_CUBE_CHECKPOINT"
  "$GOALMAX25_MLP_PUSHT_CHECKPOINT" "$GOALMAX25_ENDPOINT_FLOW_PUSHT_CHECKPOINT" "$GOALMAX25_PUSHT_CHECKPOINT"
  "$GOALMAX25_MLP_REACHER_CHECKPOINT" "$GOALMAX25_ENDPOINT_FLOW_REACHER_CHECKPOINT" "$GOALMAX25_REACHER_CHECKPOINT"
  "$GOALMAX25_MLP_TWOROOM_CHECKPOINT" "$GOALMAX25_ENDPOINT_FLOW_TWOROOM_CHECKPOINT" "$GOALMAX25_TWOROOM_CHECKPOINT"
)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/impls"

for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  for generator_index in "${!GENERATOR_TYPES[@]}"; do
    generator_type=${GENERATOR_TYPES[$generator_index]}
    checkpoint_index=$((index * 3 + generator_index))
    "$PYTHON_BIN" "$REPO_ROOT/experiments/validate_generator_checkpoint.py" \
      --checkpoint="${SUBGOAL_CHECKPOINTS[$checkpoint_index]}" \
      --task="$task" \
      --family=goalmax25 \
      --generator-type="$generator_type" \
      --goal-sampling=uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25 \
      --max-goal-steps=25
  done
done

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    for generator_index in "${!GENERATOR_TYPES[@]}"; do
      generator_type=${GENERATOR_TYPES[$generator_index]}
      checkpoint_index=$((index * 3 + generator_index))
      for seed in "${EVAL_SEEDS[@]}"; do
        result_dir="$OUTPUT_ROOT/subgoal_generator_comparison_h25/goalmax25/${generator_type}/H25/full/policy_mode/seed${seed}/$task"
      mkdir -p "$result_dir"
      test ! -e "$result_dir/result.json"
        args=(
          --task="$task" --variant=full --experiment-group=subgoal_generator_comparison_h25
          --generator-family=goalmax25 --generator-type="$generator_type"
          --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="${LEWM_CHECKPOINTS[$index]}"
          --action-prior-checkpoint-dir="${ACTION_PRIOR_CHECKPOINTS[$index]}"
          --action-prior-checkpoint-step=100000 --action-prior-mode=policy_mode
          --subgoal-generator-checkpoint="${SUBGOAL_CHECKPOINTS[$checkpoint_index]}"
          --flow-sampling-steps=16 --generator-num-samples=1
          --num-eval=50 --seed="$seed" --goal-offset-steps=25 --eval-budget=50
          --cem-horizon=2 --cem-receding-horizon=1 --action-block=5
          --cem-num-samples=300 --cem-iterations=5 --cem-topk=30
          --cem-var-scale=1.0 --cem-min-std=0.001 --cem-cost-mode=moh
          --output="$result_dir/result.json"
        )
        "$PYTHON_BIN" eval_lewm_4tasks.py "${args[@]}" --validate-only
        "$PYTHON_BIN" eval_lewm_4tasks.py "${args[@]}" 2>&1 | tee "$result_dir/eval.log"
      done
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
