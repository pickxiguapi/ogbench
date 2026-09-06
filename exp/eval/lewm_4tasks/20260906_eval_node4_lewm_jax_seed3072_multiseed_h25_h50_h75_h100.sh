#!/usr/bin/env bash
set -euo pipefail

# A800 node4：评测纯 LeWM-JAX/CEM 的长距离分数。Cube、PushT、Reacher、
# TwoRoom 四个任务全部固定使用 training seed 3072 的 epoch-10 checkpoint；
# 不加载 policy 或 subgoal generator。统一 MoH、H5/RH1/J5、CEM300x30、
# budget=2H、50 episodes，覆盖 H25/H50/H75/H100 与 evaluation seeds 0/1/42。
# 每批用 8 卡并行两个 horizon/seed 设置，结果与混合 checkpoint 实验隔离。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GOAL_OFFSETS=${GOAL_OFFSETS:-"25 50 75 100"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
RUN_ROOT=${RUN_ROOT:-$EVAL_ROOT/20260906_lewm_jax_allseed3072_generator_none_moh_cem300x30_h5_rh1_ep${NUM_EVAL}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-lewm-jax-allseed3072}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a goal_offsets <<< "$GOAL_OFFSETS"
read -r -a all_gpus <<< "$GPU_IDS"
if (( ${#all_gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi
if (( ${#eval_seeds[@]} != 3 )); then
  echo "EVAL_SEEDS must contain exactly three whitespace-separated seeds." >&2
  exit 2
fi

for value in "$NUM_EVAL" "${eval_seeds[@]}" "${goal_offsets[@]}"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "NUM_EVAL, EVAL_SEEDS, and GOAL_OFFSETS must be non-negative integers." >&2
    exit 2
  fi
done
for checkpoint in "${lewm_checkpoints[@]}"; do
  if [[ "$checkpoint" != *lewm-jax-seed3072* || "$checkpoint" != *seed3072* ]]; then
    echo "Checkpoint is not explicitly bound to training seed 3072: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing LeWM-JAX seed-3072 checkpoint: $checkpoint" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" "$TMP_ROOT"
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "training_seed=3072"
echo "evaluation_seeds=$EVAL_SEEDS"
echo "goal_offsets=$GOAL_OFFSETS"
echo "num_eval=$NUM_EVAL"
echo "generator_family=none"
echo "protocol=LeWM-JAX,CEM300x30,MoH,H5/RH1/J5,budget=2H"

jobs=()
for eval_seed in "${eval_seeds[@]}"; do
  for goal_offset in "${goal_offsets[@]}"; do
    jobs+=("$eval_seed:$goal_offset")
  done
done

run_setting() {
  local gpu_ids=$1
  local eval_seed=$2
  local goal_offset=$3
  local eval_budget=$((goal_offset * 2))
  local setting_root="$RUN_ROOT/h${goal_offset}_b${eval_budget}_seed${eval_seed}"
  local -a gpus
  local -a pids=()
  read -r -a gpus <<< "$gpu_ids"

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$setting_root/$task"
    local output="$output_dir/result.json"
    local task_tmp="$TMP_ROOT/seed${eval_seed}/h${goal_offset}/$task"
    if [[ -s "$output" ]]; then
      echo "Skipping complete result: $output"
      continue
    fi
    mkdir -p "$output_dir" "$task_tmp"
    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=none \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
        --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=30 --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode=moh \
        --output="$output" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

failed=0
for (( base=0; base<${#jobs[@]}; base+=2 )); do
  batch_pids=()
  IFS=: read -r eval_seed goal_offset <<< "${jobs[$base]}"
  run_setting "${all_gpus[*]:0:4}" "$eval_seed" "$goal_offset" &
  batch_pids+=("$!")

  if (( base + 1 < ${#jobs[@]} )); then
    IFS=: read -r eval_seed goal_offset <<< "${jobs[$((base + 1))]}"
    run_setting "${all_gpus[*]:4:4}" "$eval_seed" "$goal_offset" &
    batch_pids+=("$!")
  fi

  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

if (( failed != 0 )); then
  echo "One or more evaluation cells failed; see the per-task eval.log files." >&2
  touch "$RUN_ROOT/FAILED"
  exit "$failed"
fi

"$PYTHON_BIN" - "$RUN_ROOT" "$EVAL_SEEDS" "$GOAL_OFFSETS" "$NUM_EVAL" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
eval_seeds = [int(value) for value in sys.argv[2].split()]
goal_offsets = [int(value) for value in sys.argv[3].split()]
num_eval = int(sys.argv[4])
tasks = ['cube', 'pusht', 'reacher', 'tworoom']
rows = []
for seed in eval_seeds:
    for horizon in goal_offsets:
        budget = 2 * horizon
        for task in tasks:
            result_path = run_root / f'h{horizon}_b{budget}_seed{seed}' / task / 'result.json'
            result = json.loads(result_path.read_text())
            checkpoint = result['lewm_checkpoint']
            if 'seed3072' not in checkpoint or result.get('use_subgoal'):
                raise RuntimeError(f'invalid protocol in {result_path}')
            rows.append({
                'seed': seed,
                'task': task,
                'horizon': horizon,
                'eval_budget': budget,
                'success_rate': float(result['success_rate']),
                'successes': int(round(float(result['success_rate']) * num_eval / 100.0)),
                'num_eval': num_eval,
                'checkpoint': checkpoint,
            })

with (run_root / 'per_seed.tsv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

aggregate_rows = []
for task in tasks:
    for horizon in goal_offsets:
        values = [row['success_rate'] for row in rows if row['task'] == task and row['horizon'] == horizon]
        aggregate_rows.append({
            'task': task,
            'horizon': horizon,
            'eval_budget': 2 * horizon,
            'mean_success_rate': statistics.mean(values),
            'sample_std': statistics.stdev(values),
            'num_seeds': len(values),
            'eval_seeds': ','.join(map(str, eval_seeds)),
            'episodes_per_seed': num_eval,
        })

with (run_root / 'aggregate.tsv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=aggregate_rows[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(aggregate_rows)
PY

echo "finished_at=$(date --iso-8601=seconds)"
touch "$RUN_ROOT/DONE"
