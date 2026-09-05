#!/usr/bin/env bash
set -euo pipefail

# A800 node4: compare how the shared-all GCIQL-Chunk-AWR policy is combined
# with LeWM++ planning.  All non-policy variables are frozen: LeWM seeds 3072
# (Cube/Reacher/TwoRoom) and 666 (PushT),
# goalmax25 K10 LatentPathFlow, one flow sample, policy conditioned on the
# final goal, MoH, H2/RH1, CEM300x5, action block 5, and 50 episodes.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEED=${EVAL_SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
RUN_VARIANTS=${RUN_VARIANTS:-"zero_init policy_mode policy_mode_anchor policy_population64_t03 lewm_select64_t03 lewm_elite64_t03_e8"}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
CEM_COST_MODE=${CEM_COST_MODE:-moh}

POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVAL_ROOT/20260905_goalmax25_ns1_policy_combinations_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}_ep${NUM_EVAL}_seed${EVAL_SEED}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260905-goalmax25-ns1-policy-combinations-g${GOAL_OFFSET_STEPS}-b${EVAL_BUDGET}-seed${EVAL_SEED}}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

all_variant_names=(
  zero_init
  policy_mode
  policy_mode_anchor
  policy_population64_t03
  lewm_select64_t03
  lewm_elite64_t03_e8
)
all_variant_guidance=(none mode mode_anchor population lewm_select lewm_elite)
all_variant_population=(0 0 0 64 64 64)
all_variant_temperature=(1.0 1.0 1.0 0.3 0.3 0.3)
all_variant_elite=(8 8 8 8 8 8)

read -r -a requested_variants <<< "$RUN_VARIANTS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 4 && ${#gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly four or eight whitespace-separated GPU IDs." >&2
  exit 2
fi
if (( GOAL_OFFSET_STEPS != 25 || EVAL_BUDGET != 50 )); then
  echo "This goalmax25 launcher requires GOAL_OFFSET_STEPS=25 and EVAL_BUDGET=50." >&2
  exit 2
fi

# Refuse to run unless every checkpoint is the H25-specific bounded-offset
# generator documented in AGENTS.md.
"$PYTHON_BIN" - "${subgoal_checkpoints[@]}" <<'PY'
import json
import pathlib
import sys

expected_sampling = "uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25"
for checkpoint_arg in sys.argv[1:]:
    checkpoint = pathlib.Path(checkpoint_arg)
    config_path = checkpoint.parent / "config.json"
    if not checkpoint.is_file():
        raise SystemExit(f"missing subgoal checkpoint: {checkpoint}")
    if not config_path.is_file():
        raise SystemExit(f"missing subgoal config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("goal_sampling") != expected_sampling:
        raise SystemExit(
            f"wrong H25 subgoal generator at {checkpoint.parent}: "
            f"goal_sampling={config.get('goal_sampling')!r}, "
            f"expected {expected_sampling!r}"
        )
    if config.get("max_goal_steps") != 25:
        raise SystemExit(
            f"wrong H25 max_goal_steps at {checkpoint.parent}: "
            f"got {config.get('max_goal_steps')!r}, expected 25"
        )
    print(f"verified goalmax25 generator: {checkpoint.parent.name}")
PY

variants_per_batch=$((${#gpus[@]} / 4))

variant_names=()
variant_guidance=()
variant_population=()
variant_temperature=()
variant_elite=()
for requested in "${requested_variants[@]}"; do
  found=0
  for i in "${!all_variant_names[@]}"; do
    if [[ "$requested" == "${all_variant_names[$i]}" ]]; then
      variant_names+=("${all_variant_names[$i]}")
      variant_guidance+=("${all_variant_guidance[$i]}")
      variant_population+=("${all_variant_population[$i]}")
      variant_temperature+=("${all_variant_temperature[$i]}")
      variant_elite+=("${all_variant_elite[$i]}")
      found=1
      break
    fi
  done
  if (( ! found )); then
    echo "Unknown variant: $requested" >&2
    exit 2
  fi
done

run_variant() {
  local name=$1
  local guidance=$2
  local population_size=$3
  local temperature=$4
  local elite_size=$5
  shift 5
  local variant_gpus=("$@")
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$OUTPUT_ROOT/$name/$task"
    local result_file="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/$name/$task"
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local -a policy_args=()
    mkdir -p "$output_dir" "$task_tmp"

    if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
      echo "SKIP completed $name/$task"
      continue
    fi
    if [[ "$guidance" != none ]]; then
      policy_args+=(
        --policy-checkpoint-dir="$policy_dir"
        --policy-checkpoint-step="$POLICY_STEPS"
      )
    fi

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${variant_gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance="$guidance" \
        --use-subgoal --guidance-goal-mode=final \
        --guidance-population-size="$population_size" \
        --guidance-temperature="$temperature" \
        --guidance-elite-size="$elite_size" \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        "${policy_args[@]}" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
        --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
        --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode="$CEM_COST_MODE" \
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
for ((base=0; base<${#variant_names[@]}; base+=variants_per_batch)); do
  batch_pids=()
  for ((slot=0; slot<variants_per_batch && base+slot<${#variant_names[@]}; slot++)); do
    variant=$((base + slot))
    gpu_offset=$((slot * 4))
    run_variant \
      "${variant_names[$variant]}" \
      "${variant_guidance[$variant]}" \
      "${variant_population[$variant]}" \
      "${variant_temperature[$variant]}" \
      "${variant_elite[$variant]}" \
      "${gpus[@]:gpu_offset:4}" &
    batch_pids+=("$!")
  done
  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done

exit "$failed"
