#!/usr/bin/env bash
set -euo pipefail

# Server 23：四任务评测 GCIQL-Chunk mode 初始化的 LeWM-MPPI + native twin-Q 软加权。
# 每轮先由 min-over-horizon LeWM cost 选 top30，再以标准化 min(Q1,Q2) 调整官方 MPPI
# logits；对比 beta=0.25/0.5。其余固定为 samples300、J5、temperature0.5、H5、
# RH1、action-block5、goal25/budget50、50 episodes、seed42、paired planner keys。
CLIENT_ID=23
source /home/dzb/ogbench-shared-q/scripts/client_env.sh
OGBENCH_ROOT=/home/dzb/ogbench-shared-q
PYTHON_BIN=/home/dzb/ogbench/.venv/bin/python
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
gpus=(0 3 4 5)
output_root=/data/dzb/stablewm-data/lewm-jax-mppi-q-evals/20260821_official_mppi_gciql_init_native_q_soft_top30_seed42

for beta in 0.25 0.5; do
  pids=()
  beta_tag=${beta/./p}
  for i in "${!tasks[@]}"; do
    output_dir="$output_root/beta$beta_tag/${tasks[$i]}"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_jax_cem.py \
      --planner=mppi --mppi-temperature=0.5 --mppi-native-q-beta="$beta" \
      --task="${tasks[$i]}" \
      --checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
      --data-root="$LEWM_DATA_ROOT" \
      --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
      --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=min_over_horizon --paired-plan-keys \
      --proposal-method=gciql_chunk \
      --proposal-checkpoint-dir="/data/dzb/stablewm-data/gciql-chunk-proposals-s11/${tasks[$i]}" \
      --proposal-checkpoint-step=100000 --proposal-temperature=0.0 \
      --proposal-num-samples=1 --proposal-selection=mode \
      --output="$output_dir/${tasks[$i]}.json" \
      >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
