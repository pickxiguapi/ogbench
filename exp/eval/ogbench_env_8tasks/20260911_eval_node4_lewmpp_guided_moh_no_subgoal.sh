#!/usr/bin/env bash
set -euo pipefail

# A800 node4：评测 OGBench 视觉 8 Tasks。每个任务使用各自的 node3 seed3072/epoch10 LeWM，
# 使用 seed0、500K 的 GCIQL-Chunk-AWR policy mode 初始化 CEM，目标始终为最终任务目标，
# 禁用 subgoal generator，使用 MoH、CEM300x5、H5/RH1、action block 5；默认每任务 20 episodes。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-run}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5"}
TASK_INDICES=${TASK_INDICES:-"0 1 2 3 4 5 6 7"}
POLICY_SEED=${POLICY_SEED:-0}
POLICY_STEPS=${POLICY_STEPS:-500000}
NUM_EVAL=${NUM_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-42}
CEM_HORIZON=${CEM_HORIZON:-5}
CEM_RECEDING_HORIZON=${CEM_RECEDING_HORIZON:-1}
ACTION_BLOCK=${ACTION_BLOCK:-5}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_ITERATIONS=${CEM_ITERATIONS:-5}
CEM_TOPK=${CEM_TOPK:-30}
LEWM_ROOT=${LEWM_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8-node3-evaluated-mirror}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-visual-policy-runs/gciql-chunk-awr-500k-3seeds}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260911-ogbench8-guided-no-subgoal}

envs=(
  visual-cube-single-play-v0 visual-cube-double-play-v0
  visual-cube-triple-play-v0 visual-scene-play-v0
  visual-cube-single-noisy-v0 visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0 visual-scene-noisy-v0
)
datasets=(
  visual-cube-single-play visual-cube-double-play visual-cube-triple-play
  visual-scene-play visual-cube-single-noisy visual-cube-double-noisy
  visual-cube-triple-noisy visual-scene-noisy
)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)

read -r -a gpus <<< "$GPU_IDS"
read -r -a task_indices <<< "$TASK_INDICES"

output_root="$EVAL_ROOT/20260911_lewmpp_no_subgoal_policytrain${POLICY_SEED}_mode_finalgoal_moh_cem${CEM_NUM_SAMPLES}x${CEM_ITERATIONS}_h${CEM_HORIZON}_rh${CEM_RECEDING_HORIZON}_ep${NUM_EVAL}_evalseed${EVAL_SEED}"

lewm_checkpoint() {
  local tag=$1
  printf '%s/lewm_ogbench8_%s_e10_bs128_s3072/weights_epoch_10.msgpack' "$LEWM_ROOT" "$tag"
}

validate() {
  if (( ${#gpus[@]} < 1 )); then
    echo "GPU_IDS must contain at least one GPU ID." >&2
    exit 2
  fi
  for index in "${task_indices[@]}"; do
    if (( index < 0 || index >= ${#envs[@]} )); then
      echo "Invalid TASK_INDICES entry: $index" >&2
      exit 2
    fi
    local policy_dir="$POLICY_ROOT/seed-$POLICY_SEED/${datasets[$index]}"
    for path in \
      "$(lewm_checkpoint "${tags[$index]}")" \
      "$policy_dir/flags.json" \
      "$policy_dir/params_${POLICY_STEPS}.pkl" \
      "$OGBENCH_DATA_DIR/${envs[$index]}.npz"; do
      if [[ ! -s "$path" ]]; then
        echo "Missing required artifact: $path" >&2
        exit 2
      fi
    done
  done
  echo "VALIDATION_OK tasks=${#task_indices[@]} policy_seed=$POLICY_SEED policy_steps=$POLICY_STEPS no_subgoal=1 cost=moh"
  echo "OUTPUT_ROOT=$output_root"
}

run_one() {
  local gpu=$1
  local index=$2
  local tag=${tags[$index]}
  local dataset=${datasets[$index]}
  local output_dir="$output_root/$tag"
  local output="$output_dir/result.json"
  local policy_dir="$POLICY_ROOT/seed-$POLICY_SEED/$dataset"
  local task_tmp="$TMP_ROOT/policy${POLICY_SEED}/eval${EVAL_SEED}/$tag"
  if [[ -s "$output" ]]; then
    echo "Skipping complete result: $output"
    return 0
  fi
  mkdir -p "$output_dir" "$task_tmp"
  (
    cd "$OGBENCH_ROOT/impls"
    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_ogbench_env_8tasks.py \
      --env-name="${envs[$index]}" \
      --dataset-path="$OGBENCH_DATA_DIR/${envs[$index]}.npz" \
      --controller=lewm_cem \
      --policy-guidance=mode \
      --guidance-goal-mode=final \
      --lewm-checkpoint="$(lewm_checkpoint "$tag")" \
      --policy-checkpoint-dir="$policy_dir" \
      --policy-checkpoint-step="$POLICY_STEPS" \
      --policy-action-space=environment \
      --num-eval="$NUM_EVAL" \
      --seed="$EVAL_SEED" \
      --cem-horizon="$CEM_HORIZON" \
      --cem-receding-horizon="$CEM_RECEDING_HORIZON" \
      --action-block="$ACTION_BLOCK" \
      --cem-num-samples="$CEM_NUM_SAMPLES" \
      --cem-iterations="$CEM_ITERATIONS" \
      --cem-topk="$CEM_TOPK" \
      --cem-var-scale=1.0 \
      --cem-cost-mode=moh \
      --output="$output" >"$output_dir/eval.log" 2>&1
  )
}

status() {
  echo "OUTPUT_ROOT=$output_root"
  for index in "${task_indices[@]}"; do
    local result="$output_root/${tags[$index]}/result.json"
    local log="$output_root/${tags[$index]}/eval.log"
    if [[ -s "$result" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], "DONE", "overall_success=", d["overall_success"], "seconds=", round(d["evaluation_time"], 1), "use_subgoal=", d["use_subgoal"])' "$result" "${tags[$index]}"
    elif [[ -s "$log" ]]; then
      echo "${tags[$index]} RUNNING_OR_FAILED log=$log"
    else
      echo "${tags[$index]} PENDING"
    fi
  done
}

case "$MODE" in
  validate)
    validate
    ;;
  status)
    status
    ;;
  run)
    validate
    failed=0
    for (( base=0; base<${#task_indices[@]}; base+=${#gpus[@]} )); do
      pids=()
      for (( slot=0; slot<${#gpus[@]} && base+slot<${#task_indices[@]}; slot++ )); do
        index=${task_indices[$((base + slot))]}
        run_one "${gpus[$slot]}" "$index" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
      done
    done
    status
    exit "$failed"
    ;;
  *)
    echo "MODE must be validate, run, or status" >&2
    exit 2
    ;;
esac
