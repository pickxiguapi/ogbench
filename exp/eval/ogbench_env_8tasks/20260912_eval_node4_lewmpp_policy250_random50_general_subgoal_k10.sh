#!/usr/bin/env bash
set -euo pipefail

# A800 node4：在刚完成的 OGBench 8-task policy250/random50 LeWM++ 协议上，
# 加入各任务 checkpoint-matched general-uniform-future K5/K10
# LatentPathFlow。Policy proposal 仍以 final goal 为条件；CEM 以 predicted
# K10 subgoal 做 H2 MoH，300 samples × 5 iterations；每个官方 task 评测
# 20 episodes，seed42，single-sample generator。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

USE_SUBGOAL=1 \
SUBGOAL_FAMILY=general_uniform_future \
SUBGOAL_NUM_SAMPLES=${SUBGOAL_NUM_SAMPLES:-1} \
NUM_EVAL=${NUM_EVAL:-20} \
EVAL_SEED=${EVAL_SEED:-42} \
GPU_IDS=${GPU_IDS:-"1 2 3 4 5 7"} \
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260912-policy250-random50-general-subgoal-k10} \
bash "$SCRIPT_DIR/20260912_eval_node4_lewmpp_policy250_random50_no_subgoal.sh"
