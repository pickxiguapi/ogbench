#!/usr/bin/env bash
set -euo pipefail

# A800 node4: evaluate c=1 and c=10 GCIQL-Chunk-AWR policies as guidance
# initializers for the fixed-c=5 H25 LeWM++ stack on PushT, Reacher, TwoRoom.
#
# Adapter definition:
#   c=1  -> first predicted action + four zero normalized-action means.
#   c=10 -> first five predicted actions.
# These values initialize the first CEM block; CEM optimizes all five actions.
# The frozen LeWM dynamics and goalmax25 generator remain at action block 5.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GPU_IDS=${GPU_IDS:-"1 2 3 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
LEWM_ACTION_BLOCK=5
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_WAIT_MAX_USED_MIB=${GPU_WAIT_MAX_USED_MIB:-500}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-chunk-ablation-20260907-retry1}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVAL_ROOT/20260908_action_chunk_policy_guidance_fixedc5_goalmax25_ns1_policy_mode_prefix_zero_pad_truncate_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260908-action-chunk-policy-guidance-fixedc5-goalmax25-h25}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(pusht reacher tworoom)
chunk_sizes=(1 10)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != ${#tasks[@]} * ${#chunk_sizes[@]} )); then
  echo "GPU_IDS must contain exactly six whitespace-separated GPU IDs." >&2
  exit 2
fi
if (( ${#eval_seeds[@]} == 0 )); then
  echo "EVAL_SEEDS must contain at least one seed." >&2
  exit 2
fi

# Read and validate the config.json adjacent to every H25 subgoal checkpoint
# before any evaluation starts, as required by the workspace contract.
"$PYTHON_BIN" - \
  "$POLICY_ROOT" "$POLICY_SEED" "$POLICY_STEPS" \
  "${subgoal_checkpoints[@]}" -- "${lewm_checkpoints[@]}" <<'PY'
import json
import pathlib
import sys

policy_root = pathlib.Path(sys.argv[1])
policy_seed = int(sys.argv[2])
policy_steps = int(sys.argv[3])
separator = sys.argv.index('--')
subgoal_checkpoints = [pathlib.Path(value) for value in sys.argv[4:separator]]
lewm_checkpoints = [pathlib.Path(value) for value in sys.argv[separator + 1:]]
tasks = ('pusht', 'reacher', 'tworoom')
expected_sampling = (
    'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25'
)

if len(subgoal_checkpoints) != len(tasks) or len(lewm_checkpoints) != len(tasks):
    raise SystemExit('checkpoint/task cardinality mismatch')

for task, checkpoint, lewm_checkpoint in zip(
    tasks, subgoal_checkpoints, lewm_checkpoints
):
    config_path = checkpoint.parent / 'config.json'
    if not checkpoint.is_file():
        raise SystemExit(f'missing subgoal checkpoint: {checkpoint}')
    if not config_path.is_file():
        raise SystemExit(f'missing subgoal config: {config_path}')
    config = json.loads(config_path.read_text())
    expected = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'subgoal_steps': 10,
        'action_block': 5,
        'goal_sampling': expected_sampling,
        'max_goal_steps': 25,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f'wrong goalmax25 generator config at {checkpoint.parent}: '
                f'{key}={config.get(key)!r}, expected {value!r}'
            )
    if not lewm_checkpoint.is_file():
        raise SystemExit(f'missing frozen LeWM checkpoint: {lewm_checkpoint}')
    print(f'verified goalmax25 generator and c5 LeWM: {task}')

for chunk_size in (1, 10):
    for task in tasks:
        policy_dir = policy_root / (
            f'gc4_{task}_all_awr_c{chunk_size}_n100000_b256_a0.0_sd{policy_seed}'
        )
        checkpoint = policy_dir / f'params_{policy_steps}.pkl'
        flags_path = policy_dir / 'flags.json'
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise SystemExit(f'missing policy checkpoint: {checkpoint}')
        if not flags_path.is_file():
            raise SystemExit(f'missing policy flags: {flags_path}')
        flags = json.loads(flags_path.read_text())
        agent = flags.get('agent', {})
        representation = flags.get('representation', {})
        if agent.get('chunk_size') != chunk_size:
            raise SystemExit(
                f'wrong policy chunk size at {policy_dir}: '
                f'{agent.get("chunk_size")!r}, expected {chunk_size}'
            )
        if representation.get('mode') != 'all':
            raise SystemExit(
                f'wrong policy representation at {policy_dir}: '
                f'{representation.get("mode")!r}, expected "all"'
            )
        print(f'verified policy: task={task} chunk={chunk_size}')
PY

echo "generator_family=goalmax25"
echo "goal_offset_steps=$GOAL_OFFSET_STEPS eval_budget=$EVAL_BUDGET"
echo "lewm_action_block=$LEWM_ACTION_BLOCK"
echo "policy_guidance_action_adapter=prefix_zero_pad_truncate"
echo "git_commit=$(git -C "$OGBENCH_ROOT" rev-parse HEAD)"
echo "output_root=$OUTPUT_ROOT"

if (( VALIDATE_ONLY == 1 )); then
  echo "Validation completed; no evaluations launched."
  exit 0
fi

wait_for_gpus() {
  (( WAIT_FOR_GPUS == 1 )) || return 0
  while true; do
    local busy=0
    local gpu used
    for gpu in "${gpus[@]}"; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
      if (( used > GPU_WAIT_MAX_USED_MIB )); then
        busy=1
        echo "GPU $gpu busy: ${used} MiB"
      fi
    done
    if (( busy == 0 )); then
      echo "GPUs ${gpus[*]} are available."
      return 0
    fi
    echo "Waiting for the six evaluation GPUs."
    sleep 30
  done
}

run_seed() {
  local eval_seed=$1
  local -a pids=()
  local slot=0
  local chunk_size task_index task policy_dir output_dir result_file task_tmp

  for chunk_size in "${chunk_sizes[@]}"; do
    for task_index in "${!tasks[@]}"; do
      task=${tasks[$task_index]}
      policy_dir="$POLICY_ROOT/gc4_${task}_all_awr_c${chunk_size}_n100000_b256_a0.0_sd${POLICY_SEED}"
      output_dir="$OUTPUT_ROOT/seed${eval_seed}/c${chunk_size}/$task"
      result_file="$output_dir/result.json"
      task_tmp="$TMP_ROOT/seed${eval_seed}/c${chunk_size}/$task"
      mkdir -p "$output_dir" "$task_tmp"

      if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
        echo "SKIP completed seed=$eval_seed chunk=$chunk_size task=$task"
        slot=$((slot + 1))
        continue
      fi

      (
        cd "$OGBENCH_ROOT/impls"
        TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$slot]} \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
        "$PYTHON_BIN" eval_lewm_4tasks.py \
          --task="$task" --controller=lewm_cem --policy-guidance=mode \
          --policy-guidance-action-adapter=prefix_zero_pad_truncate \
          --use-subgoal --guidance-goal-mode=final \
          --guidance-population-size=0 --guidance-temperature=1.0 \
          --guidance-elite-size=8 \
          --data-root="$LEWM_DATA_ROOT" \
          --lewm-checkpoint="${lewm_checkpoints[$task_index]}" \
          --policy-checkpoint-dir="$policy_dir" \
          --policy-checkpoint-step="$POLICY_STEPS" \
          --latent-subgoal-checkpoint="${subgoal_checkpoints[$task_index]}" \
          --flow-sampling-steps=16 --num-samples=1 \
          --num-eval="$NUM_EVAL" --seed="$eval_seed" \
          --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
          --cem-horizon=5 --cem-receding-horizon=1 \
          --action-block="$LEWM_ACTION_BLOCK" \
          --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
          --cem-var-scale=1.0 --cem-cost-mode=moh \
          --output="$result_file" >"$output_dir/eval.log" 2>&1
      ) &
      pids+=("$!")
      echo "LAUNCHED seed=$eval_seed chunk=$chunk_size task=$task gpu=${gpus[$slot]} pid=${pids[-1]}"
      slot=$((slot + 1))
    done
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "Evaluation seed $eval_seed failed; refusing to start later seeds." >&2
    return 1
  fi
  echo "COMPLETED eval_seed=$eval_seed"
}

wait_for_gpus
for eval_seed in "${eval_seeds[@]}"; do
  run_seed "$eval_seed"
done
echo "ALL ACTION-CHUNK H25 EVALUATIONS COMPLETED"
