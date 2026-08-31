#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4–7：新 K10 LatentPathFlow 的远距离矩阵。
# 保持 last/fixed-K10 H2/RH1、每次 5 步重新生成 subgoal、8-sample、
# CEM300x30、50 episodes，依次评测 goal/budget=50/100 和 75/150。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_IDS=${GPU_IDS:-"4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
path_flow_root=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10
eval_root=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
tmp_root=/data-training/yyf/ogbench-lewm-policy-runs/tmp

tasks=(cube pusht reacher tworoom)
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != ${#tasks[@]} )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$path_flow_root/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$path_flow_root/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$path_flow_root/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$path_flow_root/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

cd "$OGBENCH_ROOT/impls"
if "$PYTHON_BIN" eval_lewm_4tasks.py --help 2>&1 | grep -q -- '--controller'; then
  eval_api=controller
else
  eval_api=legacy_mode
fi
echo "Detected eval API: $eval_api"

run_setting() {
  local goal_offset=$1
  local eval_budget=$2
  local setting_tag="g${goal_offset}_b${eval_budget}"
  local output_root="$eval_root/20260901_latent_path_flow_hist3_k10_last_cem300x30_h2_rh1_${setting_tag}_ep50_seed42"
  local pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$output_root/$task"
    local task_tmp="$tmp_root/lewm-latent-path-flow-hist3-k10-last-${setting_tag}-eval/$task"
    local api_args=()
    mkdir -p "$output_dir" "$task_tmp"

    if [[ "$eval_api" == controller ]]; then
      api_args+=(
        --controller=lewm_cem
        --policy-guidance=none
        --use-subgoal
        --cem-iterations=30
        --cem-cost-mode=last
      )
    else
      api_args+=(
        --mode=subgoal_lewm
        --latent-subgoal-refresh-steps=5
        --cem-steps=30
        --cem-cost-mode=fixed_subgoal_horizon
      )
    fi

    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_4tasks.py \
      --task="$task" --data-root="$LEWM_DATA_ROOT" \
      --lewm-checkpoint="${lewm_checkpoints[$i]}" \
      --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
      --num-samples=8 \
      --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
      --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
      --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-topk=30 --cem-var-scale=1.0 \
      "${api_args[@]}" \
      --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "Evaluation failed for setting $setting_tag; inspect $output_root." >&2
    return 1
  fi
}

run_setting 50 100
run_setting 75 150
