#!/usr/bin/env bash
set -euo pipefail

# Server 23：分析历史96.5% LeWM + GCIQL-Chunk mode + min-over-horizon基线。
# 对 CEM planning horizon H=1/2/3/4/5 做因果消融；action-block5、RH1，因此每次
# 闭环始终只执行5个环境步，但分别想象5/10/15/20/25步。每次重规划额外记录最终
# 执行轨迹的 latent goal distance 在哪个预测 checkpoint 达到最小值，并按成功/失败
# episode 汇总。其余严格固定 CEM samples300/topk30/J5、goal25/budget50、
# GCIQL-Chunk deterministic mode、四任务各50 episodes、seed42、paired planner keys。
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
output_root=/data/dzb/stablewm-data/lewm-jax-horizon-diagnostics/20260821_reproduce965_mincost_j5_rh1_ab5_seed42

for horizon in 1 2 3 4 5; do
  pids=()
  for i in "${!tasks[@]}"; do
    output_dir="$output_root/h$horizon/${tasks[$i]}"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_jax_cem.py \
      --planner=cem \
      --task="${tasks[$i]}" \
      --checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
      --data-root="$LEWM_DATA_ROOT" \
      --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
      --cem-horizon="$horizon" --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=min_over_horizon --paired-plan-keys --diagnose-min-horizon \
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
