#!/usr/bin/env bash
set -euo pipefail

# A800 node4: rerun Policy mode on H50/H75/H100 with the correct general
# full-offset LatentPathFlow generator.  Cube/Reacher/TwoRoom use LeWM seed
# 3072, PushT uses seed 666, and the shared-all GCIQL-Chunk-AWR policy uses
# seed 777.  FlowPath inference uses ns=1; policy sees the final goal; planning
# uses MoH, H2/RH1/J5, CEM300x5, budget=2H, 50 episodes, seeds 0/1/42.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"50 75 100"}
GPU_IDS=${GPU_IDS:-"4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260905-general-uniform-future-ns1-policy-mode-3seeds}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi
for goal_offset in "${goal_offsets[@]}"; do
  if (( goal_offset <= 25 )); then
    echo "This general-generator launcher only accepts goal offsets greater than 25; got $goal_offset." >&2
    exit 2
  fi
done

# Refuse to run if a goalmax25 or otherwise incompatible generator is supplied.
"$PYTHON_BIN" - "${subgoal_checkpoints[@]}" <<'PY'
import json
import pathlib
import sys

expected = "hiql_uniform_future_same_trajectory"
for checkpoint_arg in sys.argv[1:]:
    checkpoint = pathlib.Path(checkpoint_arg)
    config_path = checkpoint.parent / "config.json"
    if not checkpoint.is_file():
        raise SystemExit(f"missing subgoal checkpoint: {checkpoint}")
    if not config_path.is_file():
        raise SystemExit(f"missing subgoal config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("goal_sampling") != expected:
        raise SystemExit(
            f"wrong subgoal generator at {checkpoint.parent}: "
            f"goal_sampling={config.get('goal_sampling')!r}, expected {expected!r}"
        )
    if config.get("max_goal_steps") is not None:
        raise SystemExit(
            f"wrong bounded-offset generator at {checkpoint.parent}: "
            f"max_goal_steps={config.get('max_goal_steps')!r}"
        )
    print(f"verified general generator: {checkpoint.parent.name}")
PY

run_setting() {
  local eval_seed=$1
  local goal_offset=$2
  local eval_budget=$((goal_offset * 2))
  local output_root="$EVAL_ROOT/20260905_general_uniform_future_ns1_policy_mode_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${eval_seed}"
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local result_file="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/seed${eval_seed}/g${goal_offset}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
      echo "SKIP completed seed=$eval_seed H=$goal_offset task=$task"
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
        --use-subgoal --guidance-goal-mode=final \
        --guidance-population-size=0 --guidance-temperature=1.0 \
        --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" \
        --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --num-samples=1 \
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
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for eval_seed in "${eval_seeds[@]}"; do
  for goal_offset in "${goal_offsets[@]}"; do
    echo "RUN Policy mode with general generator: seed=$eval_seed H=$goal_offset"
    if ! run_setting "$eval_seed" "$goal_offset"; then
      failed=1
    fi
  done
done
exit "$failed"
