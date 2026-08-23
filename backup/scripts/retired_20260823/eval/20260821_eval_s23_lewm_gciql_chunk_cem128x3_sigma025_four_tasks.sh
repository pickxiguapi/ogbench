#!/usr/bin/env bash
set -euo pipefail

# Server 23：四任务评测低计算量 LeWM + GCIQL-Chunk mode 初始化 CEM；
# 初始第一 action block 均值为 actor deterministic mode，后续使用 shifted warm start，
# sigma0=0.25、population128、topk13（保持约10% elite ratio）、迭代3次；固定
# min-over-horizon、H5/RH1/action-block5、goal25/budget50、每任务50 episodes、
# seed42、paired planner keys，用于对照96.5%的 CEM300x5/sigma1.0 主基线。
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
output_root=/data/dzb/stablewm-data/lewm-jax-efficient-cem-evals/20260821_mode_mincost_k128_i3_sigma025_top13_h5_rh1_ab5_seed42

pids=()
for i in "${!tasks[@]}"; do
  output_dir="$output_root/${tasks[$i]}"
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
    --cem-num-samples=128 --cem-steps=3 --cem-topk=13 --cem-var-scale=0.25 \
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
