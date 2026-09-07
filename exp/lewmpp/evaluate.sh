#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint for LeWM++, its ablations, and the two public baselines.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${TASK:?Set TASK to cube, pusht, reacher, or tworoom}"
: "${VARIANT:?Set VARIANT; see README.md}"
: "${EXPERIMENT_GROUP:?Set EXPERIMENT_GROUP to keep reruns separate}"
: "${LEWM_DATA_ROOT:?Set LEWM_DATA_ROOT}"
: "${LEWM_CHECKPOINT:?Set LEWM_CHECKPOINT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-$((GOAL_OFFSET_STEPS * 2))}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_CHECKPOINT_STEP=${POLICY_CHECKPOINT_STEP:-100000}
ACTION_PRIOR_MODE=${ACTION_PRIOR_MODE:-policy_mode}
GENERATOR_TYPE=${GENERATOR_TYPE:-latent_path_flow}
FLOW_SAMPLING_STEPS=${FLOW_SAMPLING_STEPS:-16}
GENERATOR_NUM_SAMPLES=${GENERATOR_NUM_SAMPLES:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

case "$TASK" in cube|pusht|reacher|tworoom) ;; *) echo "Unsupported TASK: $TASK" >&2; exit 2 ;; esac
case "$VARIANT" in
  full|no_subgoal|no_action_prior|no_moh|lewm|gciql_chunk) ;;
  *) echo "Unsupported VARIANT: $VARIANT" >&2; exit 2 ;;
esac
case "$GENERATOR_TYPE" in mlp|endpoint_flow|latent_path_flow) ;; *) echo "Unsupported GENERATOR_TYPE" >&2; exit 2 ;; esac
case "$ACTION_PRIOR_MODE" in zero|policy_mode|policy_mode_anchor) ;; *) echo "Unsupported ACTION_PRIOR_MODE" >&2; exit 2 ;; esac
if [[ "$VARIANT" == gciql_chunk && "$ACTION_PRIOR_MODE" != policy_mode ]]; then
  echo "gciql_chunk requires ACTION_PRIOR_MODE=policy_mode." >&2
  exit 2
fi

if (( GOAL_OFFSET_STEPS == 25 )); then
  generator_family=goalmax25
elif (( GOAL_OFFSET_STEPS > 25 )); then
  generator_family=general_uniform_future
else
  echo "The release protocol only defines H25 and H>25 evaluations." >&2
  exit 2
fi

use_generator=1
use_prior=1
cost_mode=moh
case "$VARIANT" in
  no_subgoal) use_generator=0 ;;
  no_action_prior) use_prior=0 ;;
  no_moh) cost_mode=last ;;
  lewm) use_generator=0; use_prior=0; cost_mode=last ;;
  gciql_chunk) use_generator=0 ;;
esac

prior_args=()
generator_args=()
if (( use_prior )); then
  if [[ "$ACTION_PRIOR_MODE" == zero ]]; then
    echo "$VARIANT requires policy_mode or policy_mode_anchor." >&2
    exit 2
  fi
  : "${POLICY_CHECKPOINT_DIR:?Set POLICY_CHECKPOINT_DIR for $VARIANT}"
  require_file "$POLICY_CHECKPOINT_DIR/flags.json" "action-prior flags"
  require_file "$POLICY_CHECKPOINT_DIR/params_${POLICY_CHECKPOINT_STEP}.pkl" "action-prior checkpoint"
  prior_args=(
    --action-prior-checkpoint-dir="$POLICY_CHECKPOINT_DIR"
    --action-prior-checkpoint-step="$POLICY_CHECKPOINT_STEP"
  )
else
  ACTION_PRIOR_MODE=zero
fi

if (( use_generator )); then
  : "${SUBGOAL_GENERATOR_CHECKPOINT:?Set SUBGOAL_GENERATOR_CHECKPOINT for $VARIANT}"
  verify_generator "$generator_family" "$GENERATOR_TYPE" "$SUBGOAL_GENERATOR_CHECKPOINT"
  output_family=$generator_family
  output_type=$GENERATOR_TYPE
  generator_args=(
    --subgoal-generator-checkpoint="$SUBGOAL_GENERATOR_CHECKPOINT"
    --generator-type="$GENERATOR_TYPE"
    --generator-num-samples="$GENERATOR_NUM_SAMPLES"
    --flow-sampling-steps="$FLOW_SAMPLING_STEPS"
  )
else
  output_family=no_generator
  output_type=no_generator
fi

require_file "$LEWM_CHECKPOINT" "LeWM checkpoint"
result_dir="$OUTPUT_ROOT/$EXPERIMENT_GROUP/$output_family/$output_type/H${GOAL_OFFSET_STEPS}/$VARIANT/$ACTION_PRIOR_MODE/seed${EVAL_SEED}/$TASK"
result_file="$result_dir/result.json"
if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
  echo "SKIP completed $result_file"
  exit 0
fi
mkdir -p "$result_dir"

cd "$OGBENCH_ROOT/impls"
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="$TASK" \
    --variant="$VARIANT" \
    --experiment-group="$EXPERIMENT_GROUP" \
    --generator-family="$output_family" \
    --data-root="$LEWM_DATA_ROOT" \
    --lewm-checkpoint="$LEWM_CHECKPOINT" \
    --action-prior-mode="$ACTION_PRIOR_MODE" \
    "${prior_args[@]}" \
    "${generator_args[@]}" \
    --num-eval="$NUM_EVAL" \
    --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET_STEPS" \
    --eval-budget="$EVAL_BUDGET" \
    --cem-horizon=2 \
    --cem-receding-horizon=1 \
    --action-block=5 \
    --cem-num-samples=300 \
    --cem-iterations=5 \
    --cem-topk=30 \
    --cem-var-scale=1.0 \
    --cem-min-std=0.001 \
    --cem-cost-mode="$cost_mode" \
    --output="$result_file" \
    >"$result_dir/eval.log" 2>&1
