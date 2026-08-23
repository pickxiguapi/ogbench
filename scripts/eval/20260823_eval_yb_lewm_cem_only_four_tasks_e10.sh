#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU 4–7 并行评测 LeWM 四任务的 CEM-only 对照；seed3072 epoch10 世界模型、CEM300×5、H5/RH1/action-block5、min-over-horizon、50 episodes、seed42。
CLIENT_ID=yb
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
gpus=(4 5 6 7)
output_root="$CLIENT_ROOT/lewm-gciql-chunk-shared-evals/20260823_cem_only_j5_h5_rh1_ep50_seed42"

pids=()
for i in "${!tasks[@]}"; do
  output_dir="$output_root/${tasks[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_jax_cem.py \
    --task="${tasks[$i]}" \
    --checkpoint="$CLIENT_ROOT/lewm-jax-seed3072-s23/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
    --data-root="$LEWM_DATA_ROOT" \
    --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
    --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=min_over_horizon --paired-plan-keys \
    --output="$output_dir/${tasks[$i]}.json" \
    >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
