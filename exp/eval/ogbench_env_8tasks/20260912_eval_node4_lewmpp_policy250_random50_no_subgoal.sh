#!/usr/bin/env bash
set -euo pipefail

# A800 node4: IQL-TD-MPC-inspired OGBench visual 8-task evaluation.
# Every one of five CEM iterations contains 250 policy-guided candidates and
# 50 local Gaussian candidates.  At least 25/30 elites are policy-guided, the
# first-block refit is anchored halfway to policy mode, and execution uses an
# actually scored final-pool trajectory.  No subgoal generator is used.
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
POLICY_POPULATION=${POLICY_POPULATION:-250}
POLICY_TEMPERATURE=${POLICY_TEMPERATURE:-0.05}
RANDOM_FIRST_BLOCK_STD=${RANDOM_FIRST_BLOCK_STD:-0.05}
RANDOM_ELITE_CAP=${RANDOM_ELITE_CAP:-5}
MEAN_RESIDUAL_WEIGHT=${MEAN_RESIDUAL_WEIGHT:-0.5}
LEWM_ROOT=${LEWM_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8-node3-evaluated-mirror}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-visual-policy-runs/gciql-chunk-awr-500k-3seeds}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260912-policy250-random50}
WAIT_REQUIRED_GPUS=${WAIT_REQUIRED_GPUS:-6}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}
FREE_GPU_MEMORY_MIB=${FREE_GPU_MEMORY_MIB:-500}
FREE_STABLE_CHECKS=${FREE_STABLE_CHECKS:-3}
FREE_STABLE_INTERVAL_SECONDS=${FREE_STABLE_INTERVAL_SECONDS:-10}

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
random_population=$((CEM_NUM_SAMPLES - POLICY_POPULATION))

output_root="$EVAL_ROOT/20260912_lewmpp_no_subgoal_policytrain${POLICY_SEED}_policy${POLICY_POPULATION}_random${random_population}_eachiter_temp005_randelitecap${RANDOM_ELITE_CAP}_residual05_finalcandidate_finalgoal_moh_cem${CEM_NUM_SAMPLES}x${CEM_ITERATIONS}_h${CEM_HORIZON}_rh${CEM_RECEDING_HORIZON}_ep${NUM_EVAL}_evalseed${EVAL_SEED}"

lewm_checkpoint() {
  local tag=$1
  printf '%s/lewm_ogbench8_%s_e10_bs128_s3072/weights_epoch_10.msgpack' "$LEWM_ROOT" "$tag"
}

validate() {
  if (( ${#gpus[@]} < 1 )); then
    echo "GPU_IDS must contain at least one GPU ID." >&2
    exit 2
  fi
  if (( CEM_NUM_SAMPLES != 300 || CEM_ITERATIONS != 5 )); then
    echo "This experiment is fixed to CEM300x5." >&2
    exit 2
  fi
  if (( POLICY_POPULATION != 250 || random_population != 50 )); then
    echo "This experiment requires exactly 250 policy + 50 random candidates." >&2
    exit 2
  fi
  if (( CEM_TOPK != 30 || RANDOM_ELITE_CAP != 5 )); then
    echo "This experiment requires topk=30 and random_elite_cap=5." >&2
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
  echo "VALIDATION_OK tasks=${#task_indices[@]} no_subgoal=1 cost=moh cem=300x5 policy=250 random=50 per_iteration=1 topk=30 random_elite_cap=5"
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
      --policy-guidance=policy_random_mixture \
      --guidance-population-size="$POLICY_POPULATION" \
      --guidance-temperature="$POLICY_TEMPERATURE" \
      --guidance-first-block-std="$RANDOM_FIRST_BLOCK_STD" \
      --guidance-random-elite-cap="$RANDOM_ELITE_CAP" \
      --guidance-mean-residual-weight="$MEAN_RESIDUAL_WEIGHT" \
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
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["policy_guidance_config"]; print(sys.argv[2], "DONE", "overall_success=", d["overall_success"], "seconds=", round(d["evaluation_time"], 1), "use_subgoal=", d["use_subgoal"], "mix=", (c["population_size"], c["random_size"]), "per_iteration=", c["refreshes_policy_population_each_iteration"])' "$result" "${tags[$index]}"
    elif [[ -s "$log" ]]; then
      echo "${tags[$index]} RUNNING_OR_FAILED log=$log"
    else
      echo "${tags[$index]} PENDING"
    fi
  done
}

free_gpu_ids() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="$FREE_GPU_MEMORY_MIB" \
      '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if (($2 + 0) < (limit + 0)) print $1}'
}

wait_for_gpu_count() {
  local required=$1
  local -a available=()
  local -a selected=()
  local -a current=()
  local check gpu found stable
  while true; do
    mapfile -t available < <(free_gpu_ids)
    if (( ${#available[@]} >= required )); then
      selected=("${available[@]:0:required}")
      stable=1
      for (( check=1; check<FREE_STABLE_CHECKS; check++ )); do
        sleep "$FREE_STABLE_INTERVAL_SECONDS"
        mapfile -t current < <(free_gpu_ids)
        for gpu in "${selected[@]}"; do
          found=0
          for current_gpu in "${current[@]}"; do
            if [[ "$current_gpu" == "$gpu" ]]; then found=1; break; fi
          done
          if (( found == 0 )); then stable=0; break; fi
        done
        if (( stable == 0 )); then break; fi
      done
      if (( stable == 1 )); then
        printf '%s\n' "${selected[@]}"
        return 0
      fi
      echo "GPU_FREE_CHECK_UNSTABLE candidates=${selected[*]}" >&2
    fi
    echo "WAITING_FOR_GPUS required=$required available=${#available[@]} threshold_mib=$FREE_GPU_MEMORY_MIB" >&2
    sleep "$WAIT_POLL_SECONDS"
  done
}

wait_launch() {
  validate
  if (( WAIT_REQUIRED_GPUS < 1 )); then
    echo "WAIT_REQUIRED_GPUS must be positive." >&2
    exit 2
  fi
  if (( FREE_STABLE_CHECKS < 1 || FREE_STABLE_INTERVAL_SECONDS < 1 )); then
    echo "Stable GPU checks and their interval must be positive." >&2
    exit 2
  fi
  local -a smoke_gpu=()
  mapfile -t smoke_gpu < <(wait_for_gpu_count 1)
  echo "SMOKE_START gpu=${smoke_gpu[0]}"
  MODE=run TASK_INDICES=0 GPU_IDS="${smoke_gpu[0]}" \
    NUM_EVAL=1 EVAL_SEED=42012 bash "$BASH_SOURCE"
  echo "SMOKE_OK gpu=${smoke_gpu[0]}"

  local -a launch_gpus=()
  mapfile -t launch_gpus < <(wait_for_gpu_count "$WAIT_REQUIRED_GPUS")
  local launch_gpu_string="${launch_gpus[*]}"
  echo "FULL_RUN_START gpus=$launch_gpu_string episodes_per_task=$NUM_EVAL"
  MODE=run GPU_IDS="$launch_gpu_string" bash "$BASH_SOURCE"
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
  wait-launch)
    wait_launch
    ;;
  *)
    echo "MODE must be validate, run, wait-launch, or status" >&2
    exit 2
    ;;
esac
