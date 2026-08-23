#!/usr/bin/env bash
set -euo pipefail

# 英博云：GPU 0–3 并行评测 LeWM 四任务的 GCIQL-Chunk AWR shared-π-only s100k checkpoint；确定性 mode、完整执行5步 chunk、50 episodes、seed42、goal25/budget50。
CLIENT_ID=yb
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
checkpoint_root="$CLIENT_ROOT/lewm-gciql-chunk-shared-runs"
output_root="$CLIENT_ROOT/lewm-gciql-chunk-shared-evals/20260823_shared_pi_only_policy_s100k_ep50_seed42"
tmp_root="$CLIENT_ROOT/tmp"
mkdir -p "$tmp_root"

pids=()
for i in "${!tasks[@]}"; do
  checkpoint_dir="$checkpoint_root/2026-08-22_yb_LeWM_with_GCIQL_Chunk_AWR_shared_pi_only_${tags[$i]}_k5_bs256_s100k_s0"
  output_dir="$output_root/${tags[$i]}"
  mkdir -p "$output_dir"
  TMPDIR="$tmp_root" CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_ogbench_agent_lewm_envs.py \
    --task="${tasks[$i]}" --method=gciql_chunk_lewm \
    --checkpoint-dir="$checkpoint_dir" --checkpoint-step=100000 \
    --data-root="$LEWM_DATA_ROOT" \
    --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
    --output="$output_dir/${tags[$i]}.json" \
    >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
