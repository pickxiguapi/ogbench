#!/usr/bin/env bash
set -euo pipefail

# A800 node2：等待当前 Visual GCIQL state-dependent std 八任务评测结束，再并行启动 π-only 与 Q/V/π shared 两套 LeWM 四任务策略训练，共 8 卡。
while kill -0 1030424 2>/dev/null; do sleep 30; done
while [[ ! -f /data-training/yyf/datasets/latent-geometry/.four_lewm_policy_datasets_ready ]]; do sleep 30; done

bash /data-training/yyf/ogbench-visual-policy-runs/code/ogbench-shared-policy/scripts/train/20260822_train_node2_lewm_gciql_chunk_pi_shared_four_tasks_s100k.sh &
pi_pid=$!
bash /data-training/yyf/ogbench-visual-policy-runs/code/ogbench-shared-policy/scripts/train/20260822_train_node2_lewm_gciql_chunk_qvpi_shared_four_tasks_s100k.sh &
all_pid=$!

wait "$pi_pid"
wait "$all_pid"
