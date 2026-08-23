#!/usr/bin/env bash
set -euo pipefail

# Server 23：检验长距离下 native Q 是否比 deterministic mode 更重要。
# 在 goal/budget=50/100、75/150 上，GCIQL-Chunk actor 每次采样64个首块（包含
# deterministic mode、temperature0.1），由 min(Q1,Q2) 选一个作为单一 CEM 初始中心；
# 后续完全使用 LeWM min-over-horizon CEM。固定 samples300/topk30/J5/H5/RH1、
# action-block5、每任务50 episodes、seed42、paired planner keys。
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
output_root=/data/dzb/stablewm-data/lewm-jax-q-proposal-distance-evals/20260821_native_q_select64_temp01_mincost_j5_h5_rh1_ab5_seed42

for distance in 50:100 75:150; do
  goal_offset=${distance%%:*}
  eval_budget=${distance##*:}
  pids=()
  for i in "${!tasks[@]}"; do
    output_dir="$output_root/g${goal_offset}_b${eval_budget}/${tasks[$i]}"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_jax_cem.py \
      --planner=cem \
      --task="${tasks[$i]}" \
      --checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
      --data-root="$LEWM_DATA_ROOT" \
      --num-eval=50 --seed=42 \
      --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
      --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=min_over_horizon --paired-plan-keys \
      --proposal-method=gciql_chunk \
      --proposal-checkpoint-dir="/data/dzb/stablewm-data/gciql-chunk-proposals-s11/${tasks[$i]}" \
      --proposal-checkpoint-step=100000 --proposal-temperature=0.1 \
      --proposal-num-samples=64 --proposal-selection=native_q \
      --output="$output_dir/${tasks[$i]}.json" \
      >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
