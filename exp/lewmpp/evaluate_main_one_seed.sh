#!/usr/bin/env bash
set -euo pipefail

# Reproduce the LeWM++ main result for one evaluation seed on the four LeWM tasks
# and H25/H50/H75/H100. Jobs are dispatched in batches over GPU_IDS.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

PATHS_FILE=${PATHS_FILE:-$OGBENCH_ROOT/configs/lewmpp_paths.env}
if [[ -f "$PATHS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PATHS_FILE"
fi

: "${LEWM_DATA_ROOT:?Set LEWM_DATA_ROOT or provide PATHS_FILE}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT or provide PATHS_FILE}"

TASKS=${TASKS:-"cube pusht reacher tworoom"}
HORIZONS=${HORIZONS:-"25 50 75 100"}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
EXPERIMENT_GROUP=${EXPERIMENT_GROUP:-main_one_seed_seed${EVAL_SEED}}
ACTION_PRIOR_MODE=${ACTION_PRIOR_MODE:-policy_mode}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

read -r -a tasks <<< "$TASKS"
read -r -a horizons <<< "$HORIZONS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU." >&2
  exit 2
fi

lookup() {
  local prefix=$1
  local task=$2
  local suffix=$3
  local name="${prefix}_${task^^}_${suffix}"
  printf '%s' "${!name:-}"
}

validate_job() {
  local task=$1
  local horizon=$2
  local family_prefix=GOALMAX25
  local generator_family=goalmax25
  if (( horizon > 25 )); then
    family_prefix=GENERAL
    generator_family=general_uniform_future
  elif (( horizon != 25 )); then
    echo "Unsupported horizon: $horizon (expected 25, 50, 75, or 100)." >&2
    return 2
  fi

  local lewm policy generator
  lewm=$(lookup LEWM "$task" CHECKPOINT)
  policy=$(lookup POLICY "$task" CHECKPOINT_DIR)
  generator=$(lookup "$family_prefix" "$task" CHECKPOINT)
  : "${lewm:?Missing LEWM checkpoint variable for $task}"
  : "${policy:?Missing policy checkpoint variable for $task}"
  : "${generator:?Missing $generator_family checkpoint variable for $task}"
  require_file "$lewm" "$task LeWM checkpoint"
  require_file "$policy/flags.json" "$task action-prior flags"
  require_file "$policy/params_100000.pkl" "$task action-prior checkpoint"
  verify_generator "$generator_family" latent_path_flow "$generator"
}

run_job() {
  local task=$1
  local horizon=$2
  local gpu=$3
  local family_prefix=GOALMAX25
  if (( horizon > 25 )); then family_prefix=GENERAL; fi

  local lewm policy generator
  lewm=$(lookup LEWM "$task" CHECKPOINT)
  policy=$(lookup POLICY "$task" CHECKPOINT_DIR)
  generator=$(lookup "$family_prefix" "$task" CHECKPOINT)

  echo "START task=$task H=$horizon seed=$EVAL_SEED gpu=$gpu group=$EXPERIMENT_GROUP"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
    export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}
    export TASK=$task
    export VARIANT=full
    export GENERATOR_TYPE=latent_path_flow
    export ACTION_PRIOR_MODE
    export EXPERIMENT_GROUP
    export GOAL_OFFSET_STEPS=$horizon
    export EVAL_BUDGET=$((horizon * 2))
    export EVAL_SEED
    export NUM_EVAL
    export LEWM_CHECKPOINT=$lewm
    export POLICY_CHECKPOINT_DIR=$policy
    export SUBGOAL_GENERATOR_CHECKPOINT=$generator
    bash "$SCRIPT_DIR/evaluate.sh"
  )
  echo "DONE task=$task H=$horizon seed=$EVAL_SEED gpu=$gpu"
}

for horizon in "${horizons[@]}"; do
  for task in "${tasks[@]}"; do
    validate_job "$task" "$horizon"
  done
done
echo "Validated ${#tasks[@]} tasks x ${#horizons[@]} horizons."
if [[ "$VALIDATE_ONLY" == 1 ]]; then
  exit 0
fi

failed=0
pids=()
labels=()
for horizon in "${horizons[@]}"; do
  for task in "${tasks[@]}"; do
    slot=${#pids[@]}
    run_job "$task" "$horizon" "${gpus[$slot]}" &
    pids+=("$!")
    labels+=("$task/H$horizon")
    if (( ${#pids[@]} == ${#gpus[@]} )); then
      for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
          echo "FAILED ${labels[$i]}" >&2
          failed=1
        fi
      done
      pids=()
      labels=()
    fi
  done
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED ${labels[$i]}" >&2
    failed=1
  fi
done
exit "$failed"
