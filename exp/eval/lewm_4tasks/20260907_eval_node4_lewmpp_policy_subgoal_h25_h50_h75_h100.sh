#!/usr/bin/env bash
set -euo pipefail

# A800 node4: single-variable ablation of the canonical LeWM++ evaluation.
# The policy is conditioned on the generated local subgoal pi(z, z_subgoal)
# instead of the final task goal pi(z, g).  Everything else remains fixed:
# mixed LeWM checkpoints (seed3072, PushT seed666), shared-all AWR policy
# seed777, LatentPathFlow ns1, MoH, H2/RH1/J5, CEM300x5, budget=2H,
# 50 episodes, and evaluation seeds 0/1/42.
#
# H25 uses the horizon-matched goalmax25 generator.  H50/H75/H100 use the
# general full-offset generator.  Generator configurations are checked before
# any evaluation starts, and family names are included in output paths.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50 75 100"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_WAIT_MAX_USED_MIB=${GPU_WAIT_MAX_USED_MIB:-500}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
GOALMAX25_ROOT=${GOALMAX25_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
GENERAL_ROOT=${GENERAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260907-lewmpp-policy-subgoal-h25-h100}

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

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
read -r -a all_gpus <<< "$GPU_IDS"
if (( ${#all_gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi
for goal_offset in "${goal_offsets[@]}"; do
  if [[ "$goal_offset" != 25 && "$goal_offset" != 50 && "$goal_offset" != 75 && "$goal_offset" != 100 ]]; then
    echo "GOAL_OFFSETS only accepts 25, 50, 75, and 100; got $goal_offset." >&2
    exit 2
  fi
done

verify_generator() {
  local checkpoint=$1
  local family=$2
  "$PYTHON_BIN" - "$checkpoint" "$family" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
family = sys.argv[2]
config_path = checkpoint.parent / "config.json"
if not checkpoint.is_file():
    raise SystemExit(f"missing subgoal checkpoint: {checkpoint}")
if not config_path.is_file():
    raise SystemExit(f"missing subgoal config: {config_path}")
config = json.loads(config_path.read_text())

common_expected = {
    "architecture": "latent_path_flow_transformer_encoder",
    "subgoal_steps": 10,
    "action_block": 5,
}
for key, expected in common_expected.items():
    if config.get(key) != expected:
        raise SystemExit(
            f"wrong generator config at {checkpoint.parent}: "
            f"{key}={config.get(key)!r}, expected {expected!r}"
        )

if family == "goalmax25":
    expected_sampling = (
        "uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25"
    )
    if config.get("goal_sampling") != expected_sampling:
        raise SystemExit(
            f"wrong H25 generator at {checkpoint.parent}: "
            f"goal_sampling={config.get('goal_sampling')!r}"
        )
    if config.get("max_goal_steps") != 25:
        raise SystemExit(
            f"wrong H25 max_goal_steps at {checkpoint.parent}: "
            f"{config.get('max_goal_steps')!r}"
        )
elif family == "general_uniform_future":
    if config.get("goal_sampling") != "hiql_uniform_future_same_trajectory":
        raise SystemExit(
            f"wrong general generator at {checkpoint.parent}: "
            f"goal_sampling={config.get('goal_sampling')!r}"
        )
    if config.get("max_goal_steps") is not None:
        raise SystemExit(
            f"bounded generator cannot be used for H>25: {checkpoint.parent}"
        )
else:
    raise SystemExit(f"unknown generator family: {family}")
print(f"verified {family}: {checkpoint.parent.name}")
PY
}

for i in "${!tasks[@]}"; do
  verify_generator "${goalmax25_checkpoints[$i]}" goalmax25
  verify_generator "${general_checkpoints[$i]}" general_uniform_future
  policy_dir="$POLICY_ROOT/gc4_${tasks[$i]}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  if [[ ! -s "$policy_dir/params_${POLICY_STEPS}.pkl" ]]; then
    echo "missing policy checkpoint: $policy_dir/params_${POLICY_STEPS}.pkl" >&2
    exit 2
  fi
  if [[ ! -s "${lewm_checkpoints[$i]}" ]]; then
    echo "missing LeWM checkpoint: ${lewm_checkpoints[$i]}" >&2
    exit 2
  fi
done

if (( VALIDATE_ONLY == 1 )); then
  echo "Validation completed; no evaluations launched."
  exit 0
fi

wait_for_gpus() {
  (( WAIT_FOR_GPUS == 1 )) || return 0
  while true; do
    local busy=0
    local gpu used
    for gpu in "${all_gpus[@]}"; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
      if (( used > GPU_WAIT_MAX_USED_MIB )); then
        busy=1
      fi
    done
    if (( busy == 0 )); then
      echo "GPUs ${all_gpus[*]} are available."
      return 0
    fi
    echo "Waiting for GPUs ${all_gpus[*]} (threshold ${GPU_WAIT_MAX_USED_MIB} MiB)."
    sleep 30
  done
}

run_setting() {
  local gpu_ids=$1
  local eval_seed=$2
  local goal_offset=$3
  local eval_budget=$((goal_offset * 2))
  local family
  local -a subgoal_checkpoints
  if (( goal_offset == 25 )); then
    family=goalmax25
    subgoal_checkpoints=("${goalmax25_checkpoints[@]}")
  else
    family=general_uniform_future
    subgoal_checkpoints=("${general_checkpoints[@]}")
  fi
  local output_root="$EVAL_ROOT/20260907_lewmpp_policy_subgoal_${family}_ns1_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${eval_seed}"
  local -a gpus
  local -a pids=()
  read -r -a gpus <<< "$gpu_ids"

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local result_file="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/$family/seed${eval_seed}/g${goal_offset}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
      echo "SKIP completed family=$family seed=$eval_seed H=$goal_offset task=$task"
      continue
    fi

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=mode \
        --use-subgoal --guidance-goal-mode=subgoal \
        --guidance-population-size=0 --guidance-temperature=1.0 \
        --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" \
        --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --flow-sampling-steps=16 --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
        --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode=moh \
        --output="$result_file" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

variant_eval_seeds=()
variant_goal_offsets=()
for eval_seed in "${eval_seeds[@]}"; do
  for goal_offset in "${goal_offsets[@]}"; do
    variant_eval_seeds+=("$eval_seed")
    variant_goal_offsets+=("$goal_offset")
  done
done

wait_for_gpus
failed=0
for (( base=0; base<${#variant_eval_seeds[@]}; base+=2 )); do
  batch_pids=()
  run_setting "${all_gpus[*]:0:4}" \
    "${variant_eval_seeds[$base]}" "${variant_goal_offsets[$base]}" &
  batch_pids+=("$!")

  if (( base + 1 < ${#variant_eval_seeds[@]} )); then
    run_setting "${all_gpus[*]:4:4}" \
      "${variant_eval_seeds[$((base + 1))]}" \
      "${variant_goal_offsets[$((base + 1))]}" &
    batch_pids+=("$!")
  fi

  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done
exit "$failed"
