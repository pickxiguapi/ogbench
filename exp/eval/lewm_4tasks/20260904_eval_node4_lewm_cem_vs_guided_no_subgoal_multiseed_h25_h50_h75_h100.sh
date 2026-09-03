#!/usr/bin/env bash
set -euo pipefail

# A800 node4：在 LeWM-4Tasks 上评测两个不使用 subgoal generator 的对照：
# （1）原生 LeWM-CEM，仅用 CEM 直接规划到 final goal；（2）Guided w/o
# Subgoal，保留 shared-all AWR seed777 policy mode 初始化，但 CEM 仍直接规划
# 到 final goal。Cube/Reacher/TwoRoom 固定 LeWM seed3072，PushT 固定 seed666；
# 统一 MoH、H5/RH1/J5、CEM300x5、budget=2H、50 episodes，覆盖
# H25/H50/H75/H100 与 evaluation seeds 0/1/42；每批用 8 卡并行两个四任务设置。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

METHODS=${METHODS:-"lewm_cem guided_no_subgoal"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50 75 100"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=777
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260904-lewm-cem-vs-guided-no-subgoal}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)

read -r -a methods <<< "$METHODS"
read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
read -r -a all_gpus <<< "$GPU_IDS"
if (( ${#all_gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi

variant_methods=()
variant_eval_seeds=()
variant_goal_offsets=()
for method in "${methods[@]}"; do
  if [[ "$method" != lewm_cem && "$method" != guided_no_subgoal ]]; then
    echo "Unknown method: $method" >&2
    exit 2
  fi
  for eval_seed in "${eval_seeds[@]}"; do
    for goal_offset in "${goal_offsets[@]}"; do
      variant_methods+=("$method")
      variant_eval_seeds+=("$eval_seed")
      variant_goal_offsets+=("$goal_offset")
    done
  done
done

run_setting() {
  local gpu_ids=$1
  local method=$2
  local eval_seed=$3
  local goal_offset=$4
  local eval_budget=$((goal_offset * 2))
  local output_root="$EVAL_ROOT/20260904_${method}_mixed_lewm_moh_cem300x5_h5_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${eval_seed}"
  local -a gpus
  local -a pids=()
  read -r -a gpus <<< "$gpu_ids"

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/$method/seed${eval_seed}/g${goal_offset}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    if [[ "$method" == lewm_cem ]]; then
      (
        cd "$OGBENCH_ROOT/impls"
        TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
        "$PYTHON_BIN" eval_lewm_4tasks.py \
          --task="$task" --controller=lewm_cem --policy-guidance=none \
          --data-root="$LEWM_DATA_ROOT" \
          --lewm-checkpoint="${lewm_checkpoints[$i]}" \
          --num-eval="$NUM_EVAL" --seed="$eval_seed" \
          --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
          --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
          --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
          --cem-cost-mode=moh \
          --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
      ) &
    else
      (
        cd "$OGBENCH_ROOT/impls"
        TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
        "$PYTHON_BIN" eval_lewm_4tasks.py \
          --task="$task" --controller=lewm_cem --policy-guidance=mode \
          --guidance-goal-mode=final \
          --data-root="$LEWM_DATA_ROOT" \
          --lewm-checkpoint="${lewm_checkpoints[$i]}" \
          --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
          --num-eval="$NUM_EVAL" --seed="$eval_seed" \
          --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
          --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
          --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
          --cem-cost-mode=moh \
          --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
      ) &
    fi
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for (( base=0; base<${#variant_methods[@]}; base+=2 )); do
  batch_pids=()
  run_setting "${all_gpus[*]:0:4}" \
    "${variant_methods[$base]}" \
    "${variant_eval_seeds[$base]}" \
    "${variant_goal_offsets[$base]}" &
  batch_pids+=("$!")

  if (( base + 1 < ${#variant_methods[@]} )); then
    run_setting "${all_gpus[*]:4:4}" \
      "${variant_methods[$((base + 1))]}" \
      "${variant_eval_seeds[$((base + 1))]}" \
      "${variant_goal_offsets[$((base + 1))]}" &
    batch_pids+=("$!")
  fi

  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
