#!/usr/bin/env bash
set -euo pipefail

# LeWM++ local-subgoal-horizon ablation for the already trained K=15 general
# LatentPathFlow generators. Per the explicit experiment request, both H25 and
# H50 use the general uniform-future family. Each invocation owns one GPU and
# evaluates the whitespace-separated TASKS sequentially for seeds 0/1/42.

CLIENT_ID=${CLIENT_ID:-node3}
GPU_ID=${GPU_ID:?Set GPU_ID}
TASKS=${TASKS:?Set TASKS to a whitespace-separated task list}
WAIT_FOR_SCREEN=${WAIT_FOR_SCREEN:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

PYTHON_BIN=${PYTHON_BIN_OVERRIDE:-$PYTHON_BIN}
LEWM_DATA_ROOT=${LEWM_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k15}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260907-lewmpp-general-k15-h25-h50}
LOG_ROOT=${LOG_ROOT:-$EVAL_ROOT/20260907_lewmpp_general_k15_h25_h50_launchers}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50"}
NUM_EVAL=${NUM_EVAL:-50}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
POLICY_SEED=777
POLICY_STEPS=100000
SUBGOAL_STEPS=15

if [[ -n "$WAIT_FOR_SCREEN" ]]; then
  echo "WAIT $(date --iso-8601=seconds) screen=$WAIT_FOR_SCREEN"
  while screen -ls 2>/dev/null | grep -q "[.]${WAIT_FOR_SCREEN}[[:space:]]"; do
    sleep 30
  done
fi

checkpoint_for_task() {
  local task=$1
  case "$task" in
    tworoom)
      printf '%s|%s\n' \
        /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack \
        "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg15_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
      ;;
    pusht)
      printf '%s|%s\n' \
        /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack \
        "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg15_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
      ;;
    cube)
      printf '%s|%s\n' \
        /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack \
        "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg15_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
      ;;
    reacher)
      printf '%s|%s\n' \
        /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack \
        "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg15_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
      ;;
    *)
      echo "Unsupported task: $task" >&2
      return 2
      ;;
  esac
}

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
if (( ${#eval_seeds[@]} != 3 )); then
  echo "EVAL_SEEDS must contain exactly three values." >&2
  exit 2
fi
if [[ " ${goal_offsets[*]} " != " 25 50 " ]]; then
  echo "This launcher is restricted to GOAL_OFFSETS='25 50'." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$TMP_ROOT"

for task in $TASKS; do
  pair=$(checkpoint_for_task "$task")
  IFS='|' read -r lewm_checkpoint subgoal_checkpoint <<< "$pair"
  policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  for path in \
    "$lewm_checkpoint" \
    "$subgoal_checkpoint" \
    "$policy_dir/flags.json" \
    "$policy_dir/params_${POLICY_STEPS}.pkl"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing required artifact: $path" >&2
      exit 2
    fi
  done

  "$PYTHON_BIN" - "$subgoal_checkpoint" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
config_path = checkpoint.parent / 'config.json'
config = json.loads(config_path.read_text())
expected = ('hiql_uniform_future_same_trajectory', None, 15, 5, 16)
actual = (
    config.get('goal_sampling'),
    config.get('max_goal_steps'),
    int(config.get('subgoal_steps', -1)),
    int(config.get('action_block', -1)),
    int(config.get('flow_sampling_steps', -1)),
)
if actual != expected:
    raise SystemExit(f'K15 general config mismatch at {config_path}: {actual!r} != {expected!r}')
print(f'verified general_uniform_future_k15: {checkpoint.parent.name}')
PY

  for eval_seed in "${eval_seeds[@]}"; do
    for goal_offset in "${goal_offsets[@]}"; do
      eval_budget=$((goal_offset * 2))
      output_root="$EVAL_ROOT/20260907_subgoal_horizon_general_uniform_future_k15_ns1_policy_mode_sd777_moh_cem300x5_floweuler16_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${eval_seed}"
      output_dir="$output_root/$task"
      result_file="$output_dir/result.json"
      task_tmp="$TMP_ROOT/$task/seed${eval_seed}/g${goal_offset}"
      if [[ "$SKIP_COMPLETED" == 1 && -s "$result_file" ]]; then
        echo "SKIP task=$task seed=$eval_seed H=$goal_offset"
        continue
      fi

      mkdir -p "$output_dir" "$task_tmp"
      echo "RUN $(date --iso-8601=seconds) task=$task seed=$eval_seed H=$goal_offset gpu=$GPU_ID"
      (
        cd "$OGBENCH_ROOT/impls"
        TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES="$GPU_ID" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
        "$PYTHON_BIN" eval_lewm_4tasks.py \
          --task="$task" --controller=lewm_cem --policy-guidance=mode \
          --use-subgoal --guidance-goal-mode=final \
          --guidance-population-size=0 --guidance-temperature=1.0 \
          --guidance-elite-size=8 \
          --data-root="$LEWM_DATA_ROOT" \
          --lewm-checkpoint="$lewm_checkpoint" \
          --policy-checkpoint-dir="$policy_dir" \
          --policy-checkpoint-step="$POLICY_STEPS" \
          --latent-subgoal-checkpoint="$subgoal_checkpoint" \
          --flow-sampling-steps=16 --num-samples=1 \
          --num-eval="$NUM_EVAL" --seed="$eval_seed" \
          --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
          --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
          --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
          --cem-var-scale=1.0 --cem-cost-mode=moh \
          --output="$result_file" >"$output_dir/eval.log" 2>&1
      )
      echo "DONE $(date --iso-8601=seconds) task=$task seed=$eval_seed H=$goal_offset result=$result_file"
    done
  done
done

touch "$LOG_ROOT/DONE_gpu${GPU_ID}"
echo "ALL_DONE $(date --iso-8601=seconds) gpu=$GPU_ID tasks=$TASKS"
