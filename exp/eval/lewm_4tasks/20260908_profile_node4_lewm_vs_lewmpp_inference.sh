#!/usr/bin/env bash
set -euo pipefail

# A800 node4：在空闲独占 GPU 上分析 LeWM 与 LeWM++ 的稳态推理时间。
# 四任务使用 canonical mixed LeWM checkpoint；LeWM 使用论文基线
# CEM300x30、H5/RH1/J5、MoH，LeWM++ 使用 shared-all AWR seed777、
# LatentPathFlow Euler16、CEM300x5、H2/RH1/J5、MoH。H25 强制使用
# goalmax25 generator，H50 强制使用 general_uniform_future generator。
# 每个任务并行评估 NUM_EVAL 个环境，计时同步 GPU 并单独剔除 JIT 冷启动。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

GPU_IDS=${GPU_IDS:-"3 7"}
METHODS=${METHODS:-"lewm lewmpp"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50"}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EVAL=${NUM_EVAL:-16}
POLICY_SEED=777
POLICY_STEPS=100000
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
GOALMAX25_ROOT=${GOALMAX25_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
GENERAL_ROOT=${GENERAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
PROFILE_ROOT=${PROFILE_ROOT:-$OGBENCH_ROOT/profile_output/20260908_lewm_vs_lewmpp_inference}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260908-inference-profile}

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
read -r -a methods <<< "$METHODS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
if (( ${#gpus[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU ID." >&2
  exit 2
fi
for method in "${methods[@]}"; do
  if [[ "$method" != lewm && "$method" != lewmpp && "$method" != lewmpp_last ]]; then
    echo "METHODS only accepts lewm, lewmpp, and lewmpp_last; got $method." >&2
    exit 2
  fi
done
for horizon in "${goal_offsets[@]}"; do
  if [[ "$horizon" != 25 && "$horizon" != 50 ]]; then
    echo "GOAL_OFFSETS only accepts 25 and 50 in this family-control profile." >&2
    exit 2
  fi
done
for value in "$NUM_EVAL" "$EVAL_SEED" "${gpus[@]}"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "NUM_EVAL, EVAL_SEED, and GPU IDs must be non-negative integers." >&2
    exit 2
  fi
done
if (( NUM_EVAL == 0 )); then
  echo "NUM_EVAL must be positive." >&2
  exit 2
fi

for gpu in "${gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  if (( used >= 500 )); then
    echo "GPU $gpu is not idle enough for latency profiling: ${used} MiB used." >&2
    exit 2
  fi
done

# Read and validate every generator config before launching any task.
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
            raise SystemExit(f'wrong flow_sampling_steps at {checkpoint.parent}')
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

mkdir -p "$PROFILE_ROOT" "$TMP_ROOT"
exec > >(tee -a "$PROFILE_ROOT/launcher.log") 2>&1
echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "gpu_ids=$GPU_IDS methods=$METHODS horizons=$GOAL_OFFSETS"
echo "num_eval=$NUM_EVAL eval_seed=$EVAL_SEED"

run_task() {
  local gpu=$1
  local method=$2
  local goal_offset=$3
  local task_index=$4
  local task=${tasks[$task_index]}
  local eval_budget=$((goal_offset * 2))
  local generator_family=none
  local subgoal_checkpoint=
  if [[ "$method" != lewm ]]; then
    if (( goal_offset == 25 )); then
      generator_family=goalmax25
      subgoal_checkpoint=${goalmax25_checkpoints[$task_index]}
    else
      generator_family=general_uniform_future
      subgoal_checkpoint=${general_checkpoints[$task_index]}
    fi
  fi
  local output_dir="$PROFILE_ROOT/$method/$generator_family/H${goal_offset}/$task"
  local output="$output_dir/result.json"
  local task_tmp="$TMP_ROOT/$method/$generator_family/H${goal_offset}/$task"
  if [[ "$SKIP_COMPLETED" == 1 && -s "$output" ]]; then
    echo "SKIP method=$method family=$generator_family H=$goal_offset task=$task"
    return
  fi
  mkdir -p "$output_dir" "$task_tmp"

  local cost_mode=moh
  if [[ "$method" == lewmpp_last ]]; then cost_mode=last; fi
  local -a command=(
    "$PYTHON_BIN" eval_lewm_4tasks.py
    --task="$task" --controller=lewm_cem
    --data-root="$LEWM_DATA_ROOT"
    --lewm-checkpoint="${lewm_checkpoints[$task_index]}"
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED"
    --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget"
    --cem-receding-horizon=1 --action-block=5
    --cem-num-samples=300 --cem-topk=30 --cem-var-scale=1.0
    --cem-cost-mode="$cost_mode" --profile-inference --output="$output"
  )
  if [[ "$method" == lewm ]]; then
    command+=(
      --policy-guidance=none --cem-horizon=5 --cem-iterations=30
    )
  else
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    command+=(
      --policy-guidance=mode --guidance-goal-mode=final --use-subgoal
      --policy-checkpoint-dir="$policy_dir"
      --policy-checkpoint-step="$POLICY_STEPS"
      --latent-subgoal-checkpoint="$subgoal_checkpoint"
      --flow-sampling-steps=16 --num-samples=1
      --cem-horizon=2 --cem-iterations=5
    )
  fi

  echo "RUN gpu=$gpu method=$method family=$generator_family H=$goal_offset task=$task"
  (
    cd "$OGBENCH_ROOT/impls"
    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "${command[@]}" >"$output_dir/eval.log" 2>&1
  )
}

failed=0
for method in "${methods[@]}"; do
  for goal_offset in "${goal_offsets[@]}"; do
    for (( base=0; base<${#tasks[@]}; base+=${#gpus[@]} )); do
      pids=()
      for (( offset=0; offset<${#gpus[@]} && base+offset<${#tasks[@]}; offset++ )); do
        run_task "${gpus[$offset]}" "$method" "$goal_offset" "$((base + offset))" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
      done
    done
  done
done

echo "finished_at=$(date --iso-8601=seconds) failed=$failed"
if (( failed == 0 )); then
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/analyze_inference_timing.py" \
    --input-root="$PROFILE_ROOT" --output-dir="$PROFILE_ROOT"
  touch "$PROFILE_ROOT/DONE"
else
  touch "$PROFILE_ROOT/FAILED"
fi
exit "$failed"
