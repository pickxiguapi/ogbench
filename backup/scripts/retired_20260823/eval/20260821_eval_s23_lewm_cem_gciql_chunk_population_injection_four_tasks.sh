#!/usr/bin/env bash
set -euo pipefail

# Server 23：四任务评测 multi-sample GCIQL-Chunk population injection + LeWM-only CEM。
# 对比 K=16/32；每次重规划保留 actor mode，并在 CEM 第0轮把 K 个 policy chunks
# 注入前 K 条候选的首个 block；不使用 Q，全部候选仅由 min-over-horizon LeWM cost
# 排序。固定 CEM samples300/topk30/J5/H5/RH1/action-block5、actor temperature0.1、
# goal25/budget50、每任务50 episodes、seed42、paired planner keys。
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
output_root=/data/dzb/stablewm-data/lewm-jax-cem-population-evals/20260821_gciql_chunk_temp01_mincost_j5_h5_rh1_ab5_seed42

for population_size in 16 32; do
  pids=()
  for i in "${!tasks[@]}"; do
    output_dir="$output_root/k$population_size/${tasks[$i]}"
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
      --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=min_over_horizon --paired-plan-keys \
      --proposal-method=gciql_chunk \
      --proposal-checkpoint-dir="/data/dzb/stablewm-data/gciql-chunk-proposals-s11/${tasks[$i]}" \
      --proposal-checkpoint-step=100000 --proposal-temperature=0.1 \
      --proposal-population-size="$population_size" \
      --output="$output_dir/${tasks[$i]}.json" \
      >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
