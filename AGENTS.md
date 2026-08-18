# OGBench Local Rules

本文件作用于 `ogbench/` 整棵目录；与上层 `AGENTS.md` 同时生效。若有冲突，以这里更具体的规则为准。

## 实验 Bash：极简、清晰、可追溯

- 训练、评测、复现、消融等实验只能通过仓库 `scripts/` 中的 Bash 启动，并继续遵守上层规则：真实启动必须由 `experiment-dashboard/scripts/recorded_run.sh` 包裹。
- 训练与启动脚本放在 `scripts/train/`，评测脚本放在 `scripts/eval/`。
- 新实验 Bash 必须使用 `YYYYMMDD_<type>_<description>.sh` 命名，例如 `scripts/eval/20260811_eval_lewm_reacher_gciql_s100k.sh`。日期是脚本首次实际运行日期；禁止只写 `0811` 这类缺少年份的日期。
- 每个 Bash 都必须保持极简，让人能快速看懂真实命令。禁止冗余的路径检查、重复变量、过度封装、多层函数以及仅为“看起来健壮”而增加的样板代码；命令本身能给出明确错误时，不再重复预检。
- 每个 Bash 开头必须用中文注释写清楚：运行服务器、实验目的、任务范围、算法、训练量和关键特殊设置。注释应短而具体，不能只写“训练脚本”之类无信息描述。
- 服务器专用 Bash 的文件名必须显式包含服务器标签，例如 `yb`、`s23`、`s11`、`server7002`；禁止仅靠硬编码路径猜运行位置。推荐格式为 `YYYYMMDD_<type>_<server>_<domain>_<algorithm>_<task-or-scope>_<budget>.sh`。
- 同一服务器、同一算法、同一套关键超参，若脚本之间只差 task、环境名、数据集和 GPU，则必须合并为一个带 task 参数的极简脚本；例如四个 LeWM GCIQL 任务合并为 `YYYYMMDD_train_s23_lewm_gciql_task_s100k.sh`。任务映射应集中写在一个简短 `case` 中。
- 不同服务器、不同算法、不同训练量、不同关键超参、不同 checkpoint/seed 或不同实验目的不得为了少写文件而强行合并。此时一个 Bash 仍对应一个确定的实验配置；配置变化时新建带日期脚本，同日可追加 `_v2`、`_seed1` 等明确后缀。
- 参数化仅用于合并上述“同配置多任务”重复脚本。禁止用 task/checkpoint/seed 参数矩阵、复杂循环、数组拼装或多层函数把不同实验塞进一个 Bash。
- 关键实验配置必须直接写在脚本中，包括服务器路径、算法、训练量、seed、关键超参和输出根目录。仅允许由 `recorded_run.sh` 注入 `EXPERIMENT_EXP_NAME`、`EXPERIMENT_RUN_ID`，以及由外层 launcher 分配 `CUDA_VISIBLE_DEVICES`；不得用大量 `${VAR:-default}` 隐藏实验设置。
- Python 命令及其 flags 要在脚本中完整展开，方便只看这一个文件就复现实验。不要引入“几十个可选 Bash 参数”的抽象层。
- 脚本在前台完成真实任务；禁止在脚本内部再用 `tmux`、`screen`、`nohup` 或后台 `&` 派生无法随记录器追踪的实验进程。
- 每个脚本使用 `#!/usr/bin/env bash` 和 `set -euo pipefail`。stdout/stderr 和退出状态统一交给 `recorded_run.sh`，脚本内不要再重复 `tee` 同一份日志；只创建真实命令必需的输出目录。
- 新增或修改后至少运行 `bash -n scripts/train/<script>.sh` 或 `bash -n scripts/eval/<script>.sh`。仍留在活跃目录、但仅为兼容历史调用的旧脚本必须使用 `YYYYMMDD_legacy_*.sh` 命名；移入 backup 的脚本保留原名。不得执行或复制 legacy/backup 脚本来发起新实验。

## Bash 清理与归档

- `scripts/train/`、`scripts/eval/` 和 `scripts/setup/` 只保留当前仍有明确用途的入口；重复、过时、异机误放或已被统一脚本取代的 Bash 移入 `scripts/backup/train/`、`scripts/backup/eval/` 或 `scripts/backup/setup/`，不要直接删除。
- backup 中保留原文件名和完整内容，取消可执行权限，并用 `README.md` 记录原服务器、实验用途、归档原因和替代入口；backup 脚本仅供历史追溯，不得用于启动新实验。
- 清理前必须逐个核对脚本内容、调用关系、实验看板记录及服务器上的活动进程；正在运行、排队或仍被 launcher/eval 引用的脚本不得移动。
- Bash 整理先在本地基于 GitHub `main` 完成并单独提交，再推送 GitHub，最后让实验服务器从该 `main` 快进同步；禁止只在服务器上直接改出一个独立版本。

## 服务器公共路径

- 各服务器反复使用的稳定硬编码路径统一记录在 `scripts/client_env.sh`；实验 Bash 应先 source 该文件，再使用其中的路径变量，禁止在多个活跃脚本中重复散落同一客户端路径。
- `CLIENT_ID` 只允许使用四个固定短 ID：英博云为 `yb`、Server 23 为 `23`、Server 7002 为 `7002`、Server 11 为 `11`；禁止使用 hostname、IP、长别名或其他拼写。
- `client_env.sh` 除用于按 `CLIENT_ID` 选择服务器的极简 `case` 外，只能包含客户端身份和纯路径赋值；不得包含 GPU、task、环境名、算法、超参、`EXP_NAME`、日志名、路径检查、`mkdir`、`cd` 或任何训练/评测命令。
- 只记录当前有效的公共路径。旧 checkout、一次性 release-audit 和废弃输出目录留在 backup 历史脚本中，不得作为当前客户端默认值。
- 新增服务器时使用独立且清晰的服务器小节，并保持同一语义的变量名一致；服务器专属差异集中在 `client_env.sh`，实验配置仍留在各实验 Bash 中。

## `scripts/` 命名边界

- 所有新建的训练/评测/复现/消融 Bash 都必须带八位日期前缀。
- `README.md`、纯工具代码（例如数据转换 `.py`）不属于实验 Bash，可以不带日期；如果某个工具 Bash 会直接启动实验，则仍按实验 Bash 规则命名。
- 不再新增 `eval.sh`、`launch.sh`、`run_all.sh`、`train_<task>.sh` 这类无日期、可复用但无法对应具体实验的名字。

## Github

- 正式开发与发布分支固定为 `main`，服务器、本地和 GitHub 均以 `https://github.com/pickxiguapi/ogbench` 的 `main` 为准。
- `master` 仅用于镜像和对照上游 `https://github.com/seohongpark/ogbench` 的 `master`；禁止向 `master` 提交本地功能、实验脚本或修复。
- 日常同步先更新 `origin/master`，再把确认需要的上游变更合入 `main`；不得在 `main` 与 `master` 双线开发。
- 发布或同步服务器时必须明确使用 `main`，禁止依赖远端默认分支的隐式选择。
