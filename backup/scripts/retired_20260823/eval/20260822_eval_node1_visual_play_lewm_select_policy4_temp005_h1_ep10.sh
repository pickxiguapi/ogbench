#!/usr/bin/env bash
set -euo pipefail

# A800 node1：四卡并行测试低干预 LeWM 选择；从4个近 mode 的真实 actor chunks（temperature 0.05，含确定性 mode）中按 H1/min cost 选一个，无高斯 CEM，每内部任务10 episodes、seed42。
CLIENT_ID=node1
source /home/yyf/ogbench-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0)
tags=(single double triple scene)
lewm_exps=(
  LeWMJAX_ogbench_visual_cube_single_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23
  LeWMJAX_ogbench_visual_cube_double_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23
  LeWMJAX_ogbench_visual_cube_triple_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23
  LeWMJAX_ogbench_visual_scene_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23
)
gpus=(4 5 6 7)
output_root="$VISUAL_EVAL_ROOT/20260822_lewm_select_policy4_temp005_h1_mincost_ep10_seed42"

mkdir -p "$output_root"
pids=()
for i in "${!envs[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_jax_ogbench.py \
    --env-name="${envs[$i]}" --dataset-path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --checkpoint="$VISUAL_EVAL_ASSET_ROOT/lewm-jax-runs/${lewm_exps[$i]}/weights_step_500000.msgpack" \
    --num-eval=10 --seed=42 \
    --cem-horizon=1 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-steps=0 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=min_over_horizon --paired-plan-keys \
    --proposal-method=gciql_chunk \
    --proposal-checkpoint-dir="$VISUAL_EVAL_ASSET_ROOT/gciql-chunk-awr-visual/${tags[$i]}" \
    --proposal-checkpoint-step=500000 --proposal-temperature=0.05 \
    --proposal-action-space=environment \
    --proposal-num-samples=4 --proposal-selection=lewm \
    --output="$output_root/${tags[$i]}.json" \
    >"$output_root/${tags[$i]}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
