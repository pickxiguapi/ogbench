#!/usr/bin/env bash
set -euo pipefail

# A800 node4：正式评测 OGBench visual 8-task LeWM++。固定 policy training
# seed0、general-uniform-future K10 single-sample Subgoal Generator、final-goal
# policy proposal、每轮 250 policy + 50 random、MoH、CEM300x5、有效 H2/RH1。
# evaluation seeds 0/1/42 三组并行，每个官方 task 50 episodes；全部结束后
# 自动计算八环境及 macro 的三 seed sample mean ± std。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-run}
RUN_DATE=20260913
NUM_EVAL=50
POLICY_SEED=0
SEEDS=(0 1 42)
GPU_GROUPS=("0 1" "2 3" "4 5")
BASE_SCRIPT="$SCRIPT_DIR/20260912_eval_node4_lewmpp_policy250_random50_general_subgoal_k10.sh"
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
LAUNCHER_ROOT=${LAUNCHER_ROOT:-$EVAL_ROOT/20260913_policy250_random50_general_subgoal_k10_ep50_3seeds}
SUMMARY_FILE=${SUMMARY_FILE:-$EVAL_ROOT/20260913_lewmpp_subgoal_general_uniform_future_k10_ns1_policytrain0_policy250_random50_3evalseeds_ep50_summary.json}

run_seed() {
  local eval_seed=$1
  local gpu_ids=$2
  local launcher_log="$LAUNCHER_ROOT/seed${eval_seed}.log"
  mkdir -p "$LAUNCHER_ROOT"
  RUN_DATE="$RUN_DATE" MODE=run NUM_EVAL="$NUM_EVAL" EVAL_SEED="$eval_seed" \
  POLICY_SEED="$POLICY_SEED" GPU_IDS="$gpu_ids" \
    bash "$BASE_SCRIPT" >"$launcher_log" 2>&1
}

validate_all() {
  for i in "${!SEEDS[@]}"; do
    RUN_DATE="$RUN_DATE" MODE=validate NUM_EVAL="$NUM_EVAL" \
    EVAL_SEED="${SEEDS[$i]}" POLICY_SEED="$POLICY_SEED" \
    GPU_IDS="${GPU_GROUPS[$i]}" bash "$BASE_SCRIPT"
  done
  echo "MULTISEED_VALIDATION_OK seeds=${SEEDS[*]} episodes_per_official_task=$NUM_EVAL gpu_groups=${GPU_GROUPS[*]}"
}

status_all() {
  for i in "${!SEEDS[@]}"; do
    RUN_DATE="$RUN_DATE" MODE=status NUM_EVAL="$NUM_EVAL" \
    EVAL_SEED="${SEEDS[$i]}" POLICY_SEED="$POLICY_SEED" \
    GPU_IDS="${GPU_GROUPS[$i]}" bash "$BASE_SCRIPT"
  done
}

summarize() {
  "$PYTHON_BIN" - "$EVAL_ROOT" "$SUMMARY_FILE" "$RUN_DATE" "${SEEDS[@]}" <<'PY'
import hashlib
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
        f'{run_date}_lewmpp_subgoal_general_uniform_future_k10_ns1_'
        'policytrain0_policy250_random50_eachiter_temp005_randelitecap5_'
        'residual05_finalcandidate_finalgoal_moh_cem300x5_h2_rh1_'
        f'ep50_evalseed{seed}'
    )
    run_dir = eval_root / run_name
    for tag in tags:
        result_path = run_dir / tag / 'result.json'
        if not result_path.is_file():
            raise SystemExit(f'Missing result: {result_path}')
        result = json.loads(result_path.read_text())
        guidance = result['policy_guidance_config']
        subgoal = result['latent_subgoal']
        cem = result['cem']
        expected = {
            'use_subgoal': result['use_subgoal'] is True,
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
            'subgoal_sample': subgoal['num_samples'] == 1,
            'subgoal_step': subgoal['selected_waypoint_step'] == 10,
            'subgoal_action_block': subgoal['training_action_block'] == 5,
        }
        config_path = pathlib.Path(subgoal['checkpoint']).parent / 'config.json'
        config = json.loads(config_path.read_text())
        lewm_hash = hashlib.sha256(
            pathlib.Path(result['lewm_checkpoint']).read_bytes()
        ).hexdigest()
        expected['generator_family'] = (
            config['goal_sampling'] == 'hiql_uniform_future_same_trajectory'
            and config['max_goal_steps'] is None
        )
        expected['lewm_sha'] = config['lewm_checkpoint_sha256'] == lewm_hash
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
    print(f'{tag}: {100 * row["mean"]:.2f} ± {100 * row["sample_std"]:.2f}')
macro = summary['macro']
print(f'macro: {100 * macro["mean"]:.2f} ± {100 * macro["sample_std"]:.2f}')
print(f'SUMMARY_FILE={summary_file}')
PY
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
    ;;
  *)
    echo "MODE must be validate, run, status, or summary." >&2
    exit 2
    ;;
esac
