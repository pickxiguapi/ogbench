#!/usr/bin/env bash
set -euo pipefail

# A800 node4：完整执行 H50 subgoal predictor 消融。先训练 history3 MLP、
# EndpointFlow、LatentPathFlow 的四任务 × seeds 0/1/42（200k、bs1024），
# 再用固定 mixed LeWM 运行纯 CEM、ns1、MoH、H2/RH1/J5、CEM300x30、
# evaluation seeds 0/1/42 和每任务 50 episodes 的正式评测。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

bash "$SCRIPT_DIR/20260904_train_node4_h50_predictor_ablation.sh"
bash "$OGBENCH_ROOT/exp/eval/lewm_4tasks/20260904_eval_node4_h50_predictor_ablation.sh"
