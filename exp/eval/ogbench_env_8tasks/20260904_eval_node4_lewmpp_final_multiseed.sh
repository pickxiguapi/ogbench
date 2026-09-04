#!/usr/bin/env bash
set -euo pipefail

# A800 node4: final LeWM++ evaluation on all eight visual OGBench environments.
# Frozen node3-evaluated LeWM + seed-0 K10 LatentPathFlow + final-goal
# GCIQL-Chunk-AWR mode initialization; ns1, MoH, H2/RH1, CEM300x30.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_IDS=${GPU_IDS:-"4 5 6 7"}
TASK_INDICES=${TASK_INDICES:-"0 1 2 3 4 5 6 7"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
POLICY_SEED=${POLICY_SEED:-0}
POLICY_STEPS=${POLICY_STEPS:-500000}
NUM_EVAL=${NUM_EVAL:-50}
LEWM_ROOT=${LEWM_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8-node3-evaluated-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-ogbench8-k10}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-visual-policy-runs/gciql-chunk-awr-500k-3seeds}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260904-ogbench8-lewmpp-final}

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
read -r -a eval_seeds <<< "$EVAL_SEEDS"
if (( ${#gpus[@]} < 1 )); then
  echo "GPU_IDS must contain at least one GPU ID." >&2
  exit 2
fi

lewm_checkpoint() {
  local tag=$1
  printf '%s/lewm_ogbench8_%s_e10_bs128_s3072/weights_epoch_10.msgpack' "$LEWM_ROOT" "$tag"
}

subgoal_checkpoint() {
  local tag=$1
  printf '%s/latent_pathflow_ogbench8_%s_node3lewm3072e10_hist3_sg10_ab5_uniform_ns1_n200000_b1024_s0/checkpoint_200000.msgpack' "$SUBGOAL_ROOT" "$tag"
}

for index in "${task_indices[@]}"; do
  if (( index < 0 || index >= ${#envs[@]} )); then
    echo "Invalid TASK_INDICES entry: $index" >&2
    exit 2
  fi
  policy_dir="$POLICY_ROOT/seed-$POLICY_SEED/${datasets[$index]}"
  for path in \
    "$(lewm_checkpoint "${tags[$index]}")" \
    "$(subgoal_checkpoint "${tags[$index]}")" \
    "$policy_dir/flags.json" \
    "$policy_dir/params_${POLICY_STEPS}.pkl" \
    "$OGBENCH_DATA_DIR/${envs[$index]}.npz"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing required artifact: $path" >&2
      exit 2
    fi
  done
done

jobs=()
for eval_seed in "${eval_seeds[@]}"; do
  for index in "${task_indices[@]}"; do
    jobs+=("$eval_seed:$index")
  done
done

run_one() {
  local gpu=$1
  local eval_seed=$2
  local index=$3
  local tag=${tags[$index]}
  local dataset=${datasets[$index]}
  local output_root="$EVAL_ROOT/20260904_lewmpp_final_policytrain${POLICY_SEED}_flowtrain0_ns1_mode_finalgoal_moh_cem300x30_h2_rh1_ep${NUM_EVAL}_evalseed${eval_seed}"
  local output_dir="$output_root/$tag"
  local output="$output_dir/result.json"
  local policy_dir="$POLICY_ROOT/seed-$POLICY_SEED/$dataset"
  local task_tmp="$TMP_ROOT/policy${POLICY_SEED}/eval${eval_seed}/$tag"
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
      --controller=lewm_cem --policy-guidance=mode \
      --guidance-goal-mode=final --use-subgoal \
      --lewm-checkpoint="$(lewm_checkpoint "$tag")" \
      --latent-subgoal-checkpoint="$(subgoal_checkpoint "$tag")" \
      --num-samples=1 \
      --policy-checkpoint-dir="$policy_dir" \
      --policy-checkpoint-step="$POLICY_STEPS" \
      --policy-action-space=environment \
      --num-eval="$NUM_EVAL" --seed="$eval_seed" \
      --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-iterations=30 --cem-topk=30 \
      --cem-var-scale=1.0 --cem-cost-mode=moh \
      --output="$output" >"$output_dir/eval.log" 2>&1
  )
}

failed=0
for (( base=0; base<${#jobs[@]}; base+=${#gpus[@]} )); do
  pids=()
  for (( slot=0; slot<${#gpus[@]} && base+slot<${#jobs[@]}; slot++ )); do
    IFS=: read -r eval_seed index <<< "${jobs[$((base + slot))]}"
    run_one "${gpus[$slot]}" "$eval_seed" "$index" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
