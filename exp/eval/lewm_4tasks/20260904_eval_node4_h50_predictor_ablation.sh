#!/usr/bin/env bash
set -euo pipefail

# A800 node4：评测 H50 subgoal predictor 消融。比较 history3 MLP、单 K10
# EndpointFlow 与 K5/K10 LatentPathFlow 的 3 个训练 seed；统一只取 K10 作为
# online subgoal，固定 mixed LeWM（PushT seed666，其余 seed3072）、纯
# LeWM-CEM、无 policy guidance、ns1、MoH、H2/RH1/J5、CEM300x30、
# budget100、50 episodes 和 evaluation seeds 0/1/42。每批 8 卡并行两个设置。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
ARCHITECTURES=${ARCHITECTURES:-"history_mlp endpoint_flow latent_path_flow"}
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 42"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
NUM_EVAL=${NUM_EVAL:-50}
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-predictor-h50-ablation}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260904-h50-predictor-ablation}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_seeds=(3072 666 3072 3072)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)

read -r -a gpu_ids <<< "$GPU_IDS"
read -r -a architectures <<< "$ARCHITECTURES"
read -r -a train_seeds <<< "$TRAIN_SEEDS"
read -r -a eval_seeds <<< "$EVAL_SEEDS"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi

variant_architectures=()
variant_train_seeds=()
variant_eval_seeds=()
for architecture in "${architectures[@]}"; do
  if [[ "$architecture" != history_mlp && "$architecture" != endpoint_flow && "$architecture" != latent_path_flow ]]; then
    echo "Unknown architecture: $architecture" >&2
    exit 2
  fi
  for train_seed in "${train_seeds[@]}"; do
    for eval_seed in "${eval_seeds[@]}"; do
      variant_architectures+=("$architecture")
      variant_train_seeds+=("$train_seed")
      variant_eval_seeds+=("$eval_seed")
    done
  done
done

run_setting() {
  local setting_gpus=$1
  local architecture=$2
  local train_seed=$3
  local eval_seed=$4
  local output_root="$EVAL_ROOT/20260904_h50_${architecture}_train${train_seed}_eval${eval_seed}_ns1_lewm_cem_moh_cem300x30_h2_rh1_g50_b100_ep${NUM_EVAL}"
  local -a gpus
  local -a pids=()
  read -r -a gpus <<< "$setting_gpus"

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local exp_name="h50_${architecture}_${task}_lewm${lewm_seeds[$i]}_hist3_k10_pmatch18m_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${train_seed}"
    local subgoal_checkpoint="$RUNS_ROOT/$exp_name/checkpoint_${TRAIN_STEPS}.msgpack"
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/${architecture}/train${train_seed}/eval${eval_seed}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=none --use-subgoal \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --latent-subgoal-checkpoint="$subgoal_checkpoint" --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps=50 --eval-budget=100 \
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=30 --cem-topk=30 --cem-var-scale=1.0 \
        --cem-cost-mode=moh \
        --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
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
for (( base=0; base<${#variant_architectures[@]}; base+=2 )); do
  batch_pids=()
  run_setting "${gpu_ids[*]:0:4}" \
    "${variant_architectures[$base]}" \
    "${variant_train_seeds[$base]}" \
    "${variant_eval_seeds[$base]}" &
  batch_pids+=("$!")

  if (( base + 1 < ${#variant_architectures[@]} )); then
    run_setting "${gpu_ids[*]:4:4}" \
      "${variant_architectures[$((base + 1))]}" \
      "${variant_train_seeds[$((base + 1))]}" \
      "${variant_eval_seeds[$((base + 1))]}" &
    batch_pids+=("$!")
  fi

  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
