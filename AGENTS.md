# OGBench Local Rules

本文件作用于 `ogbench/` 整棵目录；与上层 `AGENTS.md` 同时生效。若有冲突，以这里更具体的规则为准。

## 实验脚本：一次一稿，显式留档

- 训练、评测、复现、消融等实验只能通过仓库 `scripts/` 中的 Bash 启动，并继续遵守上层规则：真实启动必须由 `experiment-dashboard/scripts/recorded_run.sh` 包裹。
- 训练与启动脚本放在 `scripts/train/`，评测脚本放在 `scripts/eval/`。
- 新实验 Bash 必须使用 `YYYYMMDD_<type>_<description>.sh` 命名，例如 `scripts/eval/20260811_eval_lewm_reacher_gciql_s100k.sh`。日期是脚本首次实际运行日期；禁止只写 `0811` 这类缺少年份的日期。
- 一个 Bash 只对应一个确定的实验或一次固定、有限的运行。它是该次实验的配置快照，运行后保留，不改造成通用启动器。配置变化时复制成一个新脚本并重新命名；同日可追加 `_v2`、`_seed1` 等明确后缀。
- Bash 必须零参数直接运行：禁止 `$1`、`getopts`、`Usage`、`case` 任务分发、`--worker` 模式，以及用命令行参数切换 task、method、checkpoint、seed 等实验配置。
- 禁止用 task/checkpoint/seed 参数矩阵、通用循环、数组拼装参数或多层函数把许多实验塞进同一个 Bash。多个实验分别写多个带日期的脚本；少量不可分割且顺序固定的步骤可以在同一脚本中逐条显式写出。
- 关键配置必须直接写在脚本里，包括服务器路径、GPU、环境、算法、checkpoint 路径与 step、seed、评测次数、关键超参、`exp_name`、输出目录和日志路径。禁止使用 `${VAR:-default}` 让外部环境静默覆盖实验配置。
- Python 命令及其 flags 要在脚本中完整展开，方便只看这一个文件就复现实验。不要引入“几十个可选 Bash 参数”的抽象层。
- 脚本在前台完成真实任务；禁止在脚本内部再用 `tmux`、`screen`、`nohup` 或后台 `&` 派生无法随记录器追踪的实验进程。
- 每个脚本使用 `#!/usr/bin/env bash` 和 `set -euo pipefail`，创建独立输出目录，并把 stdout/stderr、最终产物和退出状态留在该实验目录。运行前做必要的路径、checkpoint 和数据集存在性检查。
- 新增或修改后至少运行 `bash -n scripts/train/<script>.sh` 或 `bash -n scripts/eval/<script>.sh`。历史上不符合本规则、但为追溯实验而保留的脚本必须使用 `YYYYMMDD_legacy_*.sh` 命名；不得执行或复制 `legacy` 脚本来发起新实验。

## `scripts/` 命名边界

- 所有新建的训练/评测/复现/消融 Bash 都必须带八位日期前缀。
- `README.md`、纯工具代码（例如数据转换 `.py`）不属于实验 Bash，可以不带日期；如果某个工具 Bash 会直接启动实验，则仍按实验 Bash 规则命名。
- 不再新增 `eval.sh`、`launch.sh`、`run_all.sh`、`train_<task>.sh` 这类无日期、可复用但无法对应具体实验的名字。

## Github

固定拉最新的https://github.com/pickxiguapi/ogbench
