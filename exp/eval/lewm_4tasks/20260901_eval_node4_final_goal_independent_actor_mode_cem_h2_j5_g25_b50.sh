#!/usr/bin/env bash
set -euo pipefail

# A800 node4：并行评测两套 independent pixel policy 与 final-goal LeWM-CEM 的 mode guidance。
# 四任务、50 episodes、evaluation seed42；MoH、H2、J5、300 samples、top-30、
# action_block=5、RH1。仅使用 actor mode，Q/V 不参与。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-actor-ablation}
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks

tasks=(cube pusht reacher tworoom)
variants=(ind_awr_alpha1_sd0 ind_ddpgbc_alpha1_sd777)
variant_gpus=("0 1 2 3" "4 5 6 7")
variant_policy_suffixes=(ind_awr_alpha1_n100000_b256_a0.5_sd0 ind_ddpgbc_alpha1_n100000_b256_a0.5_sd777)
variant_output_names=(
  20260901_final_goal_ind_awr_alpha1_sd0_actor_mode_moh_cem300x5_h2_rh1_g25_b50_ep50_seed42
  20260901_final_goal_ind_ddpgbc_alpha1_sd777_actor_mode_moh_cem300x5_h2_rh1_g25_b50_ep50_seed42
)

lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)

cd "$OGBENCH_ROOT/impls"
pids=()
for variant_idx in "${!variants[@]}"; do
  read -r -a gpus <<< "${variant_gpus[$variant_idx]}"
  output_root="$EVAL_ROOT/${variant_output_names[$variant_idx]}"
  tmp_root="/data-training/yyf/ogbench-lewm-policy-runs/tmp/${variants[$variant_idx]}-final-goal-actor-mode-h2-j5"

  for task_idx in "${!tasks[@]}"; do
    task=${tasks[$task_idx]}
    policy_dir="$POLICY_ROOT/gc4_${task}_${variant_policy_suffixes[$variant_idx]}"
    output_dir="$output_root/$task"
    task_tmp="$tmp_root/$task"
    mkdir -p "$output_dir" "$task_tmp"

    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$task_idx]} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_4tasks.py \
      --task="$task" --controller=lewm_cem --policy-guidance=mode \
      --data-root="$LEWM_DATA_ROOT" \
      --lewm-checkpoint="${lewm_checkpoints[$task_idx]}" \
      --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
      --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
      --goal-offset-steps=25 --eval-budget=50 \
      --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=moh \
      --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
