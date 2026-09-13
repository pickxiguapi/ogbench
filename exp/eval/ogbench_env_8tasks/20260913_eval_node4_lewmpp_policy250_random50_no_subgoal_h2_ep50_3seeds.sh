#!/usr/bin/env bash
set -euo pipefail

# A800 node4: paired no-subgoal ablation for the formal OGBench visual
# 8-task LeWM++ evaluation.  Keep policy seed0, 250 policy + 50 random
# candidates per CEM iteration, final-goal policy guidance, MoH, CEM300x5,
# H2/RH1, action block 5, and evaluation seeds 0/1/42 unchanged.  The only
# method change is disabling the Subgoal Generator.  Run 50 episodes per
# official task and seed, then report three-seed sample mean +/- std.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-run}
RUN_DATE=${RUN_DATE:-20260913}
NUM_EVAL=50
POLICY_SEED=0
SEEDS=(0 1 42)
GPU_GROUPS=("0 1" "2 3" "4 5")
WAIT_GPU_IDS=(0 1 2 3 4 5)
BASE_SCRIPT="$SCRIPT_DIR/20260912_eval_node4_lewmpp_policy250_random50_no_subgoal.sh"
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
LAUNCHER_ROOT=${LAUNCHER_ROOT:-$EVAL_ROOT/20260913_policy250_random50_no_subgoal_h2_ep50_3seeds}
SUMMARY_FILE=${SUMMARY_FILE:-$EVAL_ROOT/20260913_lewmpp_no_subgoal_h2_policytrain0_policy250_random50_3evalseeds_ep50_summary.json}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}
FREE_GPU_MEMORY_MIB=${FREE_GPU_MEMORY_MIB:-500}
FREE_STABLE_CHECKS=${FREE_STABLE_CHECKS:-3}
FREE_STABLE_INTERVAL_SECONDS=${FREE_STABLE_INTERVAL_SECONDS:-10}

run_seed() {
  local eval_seed=$1
  local gpu_ids=$2
  local launcher_log="$LAUNCHER_ROOT/seed${eval_seed}.log"
  mkdir -p "$LAUNCHER_ROOT"
  RUN_DATE="$RUN_DATE" MODE=run NUM_EVAL="$NUM_EVAL" EVAL_SEED="$eval_seed" \
  POLICY_SEED="$POLICY_SEED" GPU_IDS="$gpu_ids" USE_SUBGOAL=0 \
  CEM_HORIZON=2 CEM_RECEDING_HORIZON=1 \
    bash "$BASE_SCRIPT" >"$launcher_log" 2>&1
}

validate_all() {
  for i in "${!SEEDS[@]}"; do
    RUN_DATE="$RUN_DATE" MODE=validate NUM_EVAL="$NUM_EVAL" \
    EVAL_SEED="${SEEDS[$i]}" POLICY_SEED="$POLICY_SEED" \
    GPU_IDS="${GPU_GROUPS[$i]}" USE_SUBGOAL=0 \
    CEM_HORIZON=2 CEM_RECEDING_HORIZON=1 bash "$BASE_SCRIPT"
  done
  echo "MULTISEED_VALIDATION_OK seeds=${SEEDS[*]} episodes_per_official_task=$NUM_EVAL no_subgoal=1 cem_horizon=2 gpu_groups=${GPU_GROUPS[*]}"
}

status_all() {
  for i in "${!SEEDS[@]}"; do
    RUN_DATE="$RUN_DATE" MODE=status NUM_EVAL="$NUM_EVAL" \
    EVAL_SEED="${SEEDS[$i]}" POLICY_SEED="$POLICY_SEED" \
    GPU_IDS="${GPU_GROUPS[$i]}" USE_SUBGOAL=0 \
    CEM_HORIZON=2 CEM_RECEDING_HORIZON=1 bash "$BASE_SCRIPT"
  done
}

requested_gpus_are_free() {
  local gpu used
  local table
  table=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for gpu in "${WAIT_GPU_IDS[@]}"; do
    used=$(awk -F, -v wanted="$gpu" '
      {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2)}
      $1 == wanted {print $2}
    ' <<< "$table")
    if [[ -z "$used" ]] || (( used >= FREE_GPU_MEMORY_MIB )); then
      return 1
    fi
  done
  return 0
}

wait_for_requested_gpus() {
  local check stable
  while true; do
    if requested_gpus_are_free; then
      stable=1
      for (( check=1; check<FREE_STABLE_CHECKS; check++ )); do
        sleep "$FREE_STABLE_INTERVAL_SECONDS"
        if ! requested_gpus_are_free; then
          stable=0
          break
        fi
      done
      if (( stable == 1 )); then
        echo "GPU_WAIT_OK ids=${WAIT_GPU_IDS[*]} threshold_mib=$FREE_GPU_MEMORY_MIB"
        return 0
      fi
      echo "GPU_FREE_CHECK_UNSTABLE ids=${WAIT_GPU_IDS[*]}" >&2
    fi
    echo "WAITING_FOR_GPUS ids=${WAIT_GPU_IDS[*]} threshold_mib=$FREE_GPU_MEMORY_MIB" >&2
    sleep "$WAIT_POLL_SECONDS"
  done
}

summarize() {
  "$PYTHON_BIN" - "$EVAL_ROOT" "$SUMMARY_FILE" "$RUN_DATE" "${SEEDS[@]}" <<'PY'
import json
import pathlib
import statistics
import sys

eval_root = pathlib.Path(sys.argv[1])
summary_file = pathlib.Path(sys.argv[2])
run_date = sys.argv[3]
seeds = [int(value) for value in sys.argv[4:]]
tags = (
    'cs_play', 'cd_play', 'ct_play', 'scene_play',
    'cs_noisy', 'cd_noisy', 'ct_noisy', 'scene_noisy',
)

values = {tag: [] for tag in tags}
for seed in seeds:
    run_name = (
        f'{run_date}_lewmpp_no_subgoal_policytrain0_policy250_random50_'
        'eachiter_temp005_randelitecap5_residual05_finalcandidate_'
        'finalgoal_moh_cem300x5_h2_rh1_'
        f'ep50_evalseed{seed}'
    )
    run_dir = eval_root / run_name
    for tag in tags:
        result_path = run_dir / tag / 'result.json'
        if not result_path.is_file():
            raise SystemExit(f'Missing result: {result_path}')
        result = json.loads(result_path.read_text())
        guidance = result['policy_guidance_config']
        cem = result['cem']
        expected = {
            'use_subgoal': result['use_subgoal'] is False,
            'latent_subgoal': result['latent_subgoal'] is None,
            'policy_guidance': result['policy_guidance'] == 'policy_random_mixture',
            'guidance_goal': result['guidance_goal_mode'] == 'final',
            'population': guidance['population_size'] == 250,
            'random': guidance['random_size'] == 50,
            'per_iteration': guidance['refreshes_policy_population_each_iteration'] is True,
            'random_elite_cap': guidance['random_elite_cap'] == 5,
            'mean_residual_weight': guidance['mean_residual_weight'] == 0.5,
            'cem': cem == {
                'horizon': 2,
                'receding_horizon': 1,
                'action_block': 5,
                'num_samples': 300,
                'iterations': 5,
                'topk': 30,
                'var_scale': 1.0,
                'cost_mode': 'moh',
            },
            'episodes': result['episodes_per_task'] == 50,
            'seed': result['seed'] == seed,
            'policy_step': result['policy_checkpoint_step'] == 500000,
        }
        failed = [name for name, passed in expected.items() if not passed]
        if failed:
            raise SystemExit(f'Config mismatch in {result_path}: {failed}')
        values[tag].append(float(result['overall_success']))

by_environment = {}
for tag in tags:
    rates = values[tag]
    by_environment[tag] = {
        'per_seed': dict(zip(map(str, seeds), rates)),
        'mean': statistics.mean(rates),
        'sample_std': statistics.stdev(rates),
    }

macro_per_seed = [
    statistics.mean(values[tag][index] for tag in tags)
    for index in range(len(seeds))
]
summary = {
    'evaluation_seeds': seeds,
    'episodes_per_official_task': 50,
    'ablation': 'LeWM++ without Subgoal Generator; all other formal settings fixed',
    'std_definition': 'sample standard deviation across evaluation seeds (ddof=1)',
    'by_environment': by_environment,
    'macro': {
        'per_seed': dict(zip(map(str, seeds), macro_per_seed)),
        'mean': statistics.mean(macro_per_seed),
        'sample_std': statistics.stdev(macro_per_seed),
    },
}
summary_file.write_text(json.dumps(summary, indent=2) + '\n')
for tag in tags:
    row = by_environment[tag]
    print(f'{tag}: {100 * row["mean"]:.2f} +/- {100 * row["sample_std"]:.2f}')
macro = summary['macro']
print(f'macro: {100 * macro["mean"]:.2f} +/- {100 * macro["sample_std"]:.2f}')
print(f'SUMMARY_FILE={summary_file}')
PY
}

run_all() {
  validate_all
  pids=()
  for i in "${!SEEDS[@]}"; do
    run_seed "${SEEDS[$i]}" "${GPU_GROUPS[$i]}" &
    pids+=("$!")
    echo "SEED_START seed=${SEEDS[$i]} gpus=${GPU_GROUPS[$i]} log=$LAUNCHER_ROOT/seed${SEEDS[$i]}.log"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    status_all
    exit "$failed"
  fi
  summarize
}

case "$MODE" in
  validate)
    validate_all
    ;;
  status)
    status_all
    ;;
  summary)
    summarize
    ;;
  run)
    run_all
    ;;
  wait-launch)
    validate_all
    wait_for_requested_gpus
    run_all
    ;;
  *)
    echo "MODE must be validate, run, wait-launch, status, or summary." >&2
    exit 2
    ;;
esac
