#!/usr/bin/env bash
set -euo pipefail

# H25 single-variable ablation of LatentPathFlow in canonical LeWM++.
# Both variants retain final-goal Policy mode, MoH, effective H2/RH1/J5,
# CEM300x5, the same LeWM/policy checkpoints, and eval seeds 0/1/42.
# The no-subgoal variant removes only LatentPathFlow and makes CEM score the
# final-goal embedding directly.  The full variant uses the H25 goalmax25
# generator and verifies its training config before any evaluation starts.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-goalmax25-ns1-paired-subgoal-ablation}

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

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi

# Refuse to run unless every full-model checkpoint is the bounded-offset H25
# generator.  The no-subgoal branch deliberately loads no generator at all.
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

run_variant() {
  local eval_seed=$1
  local variant=$2
  shift 2
  local variant_gpus=("$@")
  local output_root="$EVAL_ROOT/20260906_goalmax25_vs_no_subgoal_ns1_paired_policy_mode_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${eval_seed}"
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$variant/$task"
    local result_file="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/seed${eval_seed}/$variant/$task"
    local -a subgoal_args=()

    if [[ "$variant" == with_subgoal ]]; then
      subgoal_args=(
        --use-subgoal
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}"
        --num-samples=1
      )
    elif [[ "$variant" != no_subgoal ]]; then
      echo "Unknown variant: $variant" >&2
      return 2
    fi

    if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
      echo "SKIP completed seed=$eval_seed $variant/$task"
      continue
    fi
    mkdir -p "$output_dir" "$task_tmp"

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${variant_gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=mode \
        --guidance-goal-mode=final \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" \
        --policy-checkpoint-step="$POLICY_STEPS" \
        "${subgoal_args[@]}" \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps=25 --eval-budget=50 \
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
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
for eval_seed in "${eval_seeds[@]}"; do
  echo "START paired subgoal ablation: eval_seed=$eval_seed"
  run_variant "$eval_seed" with_subgoal "${gpus[@]:0:4}" &
  with_pid=$!
  run_variant "$eval_seed" no_subgoal "${gpus[@]:4:4}" &
  without_pid=$!
  if ! wait "$with_pid"; then failed=1; fi
  if ! wait "$without_pid"; then failed=1; fi
  echo "DONE paired subgoal ablation: eval_seed=$eval_seed"
done

exit "$failed"
