#!/usr/bin/env bash
set -euo pipefail

# A800 node4：LeWM++ 推理计算量消融。固定 mixed LeWM、shared-all AWR
# policy seed777、K10 LatentPathFlow ns1、policy 看 final goal、CEM 看局部
# subgoal、MoH、H2/RH1/J5、300 samples、50 episodes、eval seeds 0/1/42。
# SWEEP=cem 在 Flow Euler16 下评测 CEM iterations 1/5/15/30；SWEEP=flow
# 在 CEM5 下评测 Flow Euler8/32，Euler16 基准由并行的 CEM sweep 产生。
# H25 强制使用 goalmax25，H50/H75/H100 强制使用 general uniform-future。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

SWEEP=${SWEEP:?Set SWEEP=cem or SWEEP=flow}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50 75 100"}
CEM_ITERATION_VALUES=${CEM_ITERATION_VALUES:-"1 5 15 30"}
FLOW_STEP_VALUES=${FLOW_STEP_VALUES:-"8 32"}
NUM_EVAL=${NUM_EVAL:-50}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000

POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
GOALMAX25_ROOT=${GOALMAX25_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
GENERAL_ROOT=${GENERAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-lewmpp-compute-scaling}
LOG_ROOT=${LOG_ROOT:-$EVAL_ROOT/20260906_lewmpp_compute_scaling_launchers}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
goalmax25_checkpoints=(
  "$GOALMAX25_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX25_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX25_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX25_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)
general_checkpoints=(
  "$GENERAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GENERAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GENERAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GENERAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

read -r -a gpus <<< "$GPU_IDS"
read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
if (( ${#gpus[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi
if (( ${#eval_seeds[@]} != 3 )); then
  echo "EVAL_SEEDS must contain exactly three whitespace-separated seeds." >&2
  exit 2
fi
if [[ "$SWEEP" == cem ]]; then
  read -r -a values <<< "$CEM_ITERATION_VALUES"
elif [[ "$SWEEP" == flow ]]; then
  read -r -a values <<< "$FLOW_STEP_VALUES"
else
  echo "SWEEP must be cem or flow; got $SWEEP." >&2
  exit 2
fi
for value in "$NUM_EVAL" "${goal_offsets[@]}" "${values[@]}"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EVAL, horizons, and sweep values must be positive integers; got $value." >&2
    exit 2
  fi
done
for value in "${eval_seeds[@]}"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Evaluation seeds must be non-negative integers; got $value." >&2
    exit 2
  fi
done
for goal_offset in "${goal_offsets[@]}"; do
  if [[ "$goal_offset" != 25 && "$goal_offset" != 50 && "$goal_offset" != 75 && "$goal_offset" != 100 ]]; then
    echo "GOAL_OFFSETS only accepts 25, 50, 75, or 100; got $goal_offset." >&2
    exit 2
  fi
done

# Read and validate every generator config before launching any task.  This is
# intentionally duplicated here rather than delegated to a historical bash.
"$PYTHON_BIN" - \
  "${goalmax25_checkpoints[@]}" -- "${general_checkpoints[@]}" <<'PY'
import json
import pathlib
import sys

separator = sys.argv.index('--')
families = (
    (
        'goalmax25',
        sys.argv[1:separator],
        'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        25,
    ),
    (
        'general_uniform_future',
        sys.argv[separator + 1:],
        'hiql_uniform_future_same_trajectory',
        None,
    ),
)
for family, checkpoints, expected_sampling, expected_max in families:
    for checkpoint_arg in checkpoints:
        checkpoint = pathlib.Path(checkpoint_arg)
        config_path = checkpoint.parent / 'config.json'
        if not checkpoint.is_file():
            raise SystemExit(f'missing subgoal checkpoint: {checkpoint}')
        if not config_path.is_file():
            raise SystemExit(f'missing subgoal config: {config_path}')
        config = json.loads(config_path.read_text())
        if config.get('goal_sampling') != expected_sampling:
            raise SystemExit(
                f'wrong {family} goal_sampling at {checkpoint.parent}: '
                f'{config.get("goal_sampling")!r}'
            )
        if config.get('max_goal_steps') != expected_max:
            raise SystemExit(
                f'wrong {family} max_goal_steps at {checkpoint.parent}: '
                f'{config.get("max_goal_steps")!r}'
            )
        if int(config.get('subgoal_steps', -1)) != 10:
            raise SystemExit(f'wrong subgoal_steps at {checkpoint.parent}')
        if int(config.get('action_block', -1)) != 5:
            raise SystemExit(f'wrong action_block at {checkpoint.parent}')
        if int(config.get('flow_sampling_steps', -1)) != 16:
            raise SystemExit(f'unexpected configured flow steps at {checkpoint.parent}')
        print(f'verified {family}: {checkpoint.parent.name}')
PY

for i in "${!tasks[@]}"; do
  policy_dir="$POLICY_ROOT/gc4_${tasks[$i]}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  for path in \
    "${lewm_checkpoints[$i]}" \
    "$policy_dir/flags.json" \
    "$policy_dir/params_${POLICY_STEPS}.pkl"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing required artifact: $path" >&2
      exit 2
    fi
  done
done

mkdir -p "$LOG_ROOT/$SWEEP" "$TMP_ROOT/$SWEEP"
exec > >(tee -a "$LOG_ROOT/$SWEEP/launcher.log") 2>&1
echo "started_at=$(date --iso-8601=seconds)"
echo "sweep=$SWEEP values=${values[*]} eval_seeds=$EVAL_SEEDS horizons=$GOAL_OFFSETS"

run_setting() {
  local value=$1
  local eval_seed=$2
  local goal_offset=$3
  local eval_budget=$((goal_offset * 2))
  local cem_iterations=5
  local flow_steps=16
  if [[ "$SWEEP" == cem ]]; then
    cem_iterations=$value
  else
    flow_steps=$value
  fi

  local generator_family
  local -a subgoal_checkpoints
  if (( goal_offset == 25 )); then
    generator_family=goalmax25
    subgoal_checkpoints=("${goalmax25_checkpoints[@]}")
  else
    generator_family=general_uniform_future
    subgoal_checkpoints=("${general_checkpoints[@]}")
  fi
  local output_root="$EVAL_ROOT/20260906_compute_scaling_${generator_family}_ns1_policy_mode_sd${POLICY_SEED}_moh_cem300x${cem_iterations}_floweuler${flow_steps}_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${eval_seed}"
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local result_file="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/$SWEEP/${generator_family}/v${value}/seed${eval_seed}/g${goal_offset}/$task"
    if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
      echo "SKIP sweep=$SWEEP value=$value seed=$eval_seed H=$goal_offset task=$task"
      continue
    fi
    mkdir -p "$output_dir" "$task_tmp"
    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=mode \
        --use-subgoal --guidance-goal-mode=final \
        --guidance-population-size=0 --guidance-temperature=1.0 \
        --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" \
        --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --flow-sampling-steps="$flow_steps" --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations="$cem_iterations" --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode=moh \
        --output="$result_file" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for value in "${values[@]}"; do
  for eval_seed in "${eval_seeds[@]}"; do
    for goal_offset in "${goal_offsets[@]}"; do
      echo "RUN sweep=$SWEEP value=$value seed=$eval_seed H=$goal_offset"
      if ! run_setting "$value" "$eval_seed" "$goal_offset"; then
        failed=1
      fi
    done
  done
done

echo "finished_at=$(date --iso-8601=seconds) failed=$failed"
if (( failed == 0 )); then
  touch "$LOG_ROOT/$SWEEP/DONE"
else
  touch "$LOG_ROOT/$SWEEP/FAILED"
fi
exit "$failed"
