# OGBench Local Rules

本文件作用于 `ogbench/`；与上层 `AGENTS.md` 同时生效。

## 最终方法边界

- LeWM世界模型训练入口只有 `impls/train_lewm_jax.py`。
- GCIQL-Chunk 训练入口只有 `impls/train_gciql_chunk.py`。
- GCIQL-Chunk 表征模式固定为 `independent`、`pi`、`qv`、`all`。
- `pi`、`qv`、`all` 必须加载已经训练好的 LeWM-JAX，并冻结 encoder、projector、predictor 与 batch statistics；只训练策略侧仍可训练的 encoder 和 downstream heads。
- 正式共享消融默认 `p_aug=0.0`。如启用增强，必须先增强同一批像素，再同时送入 pixel encoder 和 frozen LeWM encoder，禁止只增强其中一个分支，注意，如果使用`pi`、`qv`、`all`增强，前提肯定是LeWM本身也被增强训练。
- 执行阶段，也有三种方式：1️⃣ 纯Planning，就是调用LeWM来决策 2️⃣ 纯Policy，就是直接使用Chunk Level的action来决策，跟LeWM没关系 3️⃣ Guidance：GCIQL-Chunk proposal 初始化第一个 action block，LeWM min-over-horizon CEM 完成后续优化。

## 正式评测入口

- `impls/eval_lewm_4tasks.py`：LeWM-4Tasks 的 policy、LeWM、guided、native-Q 四种评测模式。
- `impls/eval_ogbench_env_8tasks.py`：OGBench-Env-8Tasks 的 policy、LeWM、guided、native-Q 四种评测模式。
- `impls/gciql_chunk_policy.py`：两套评测共用的 checkpoint 加载和 policy/native-Q adapter，不是命令入口。
- `impls/lewm_jax/planner.py` 是两套评测共用的 planner 实现，不是正式命令入口。
- shared policy 的 Q 只能通过公开 `score_actions()` 接口调用。`qv`/`all` 的 shared-Q 策略与 planner 必须引用同一个规范化 LeWM checkpoint 路径；

## 实验 Bash

- 所有新训练与评测只能通过 `exp/` 下的 Bash 发起；不得直接运行 Python 命令。
- `exp/train/` 保存正式训练 Bash；`exp/eval/lewm_4tasks/` 与 `exp/eval/ogbench_env_8tasks/` 分别保存两套评测协议。
- Bash 顶部必须用中文说明服务器、任务范围、算法、训练量和特殊设置。
- 可调参数集中放在 Bash 开头；真实 Python 命令及关键 flags 必须完整展开。
- 修改 Bash 后至少运行 `bash -n`。
- 历史实验只能放在仓库一级 `backup/`，不得从 backup 发起新实验。
- 在bash里面不要设置冗余的判断

## 当前正式 Bash

| Bash | 用途 |
|---|---|
| `exp/train/20260823_train_yb_lewm_4tasks.sh` | 英博云训练 LeWM-4Tasks canonical LeWM-JAX |
| `exp/train/20260823_train_yb_gciql_chunk_4tasks.sh` | 英博云训练 LeWM-4Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/train/20260823_train_node2_lewm_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks canonical LeWM-JAX |
| `exp/train/20260823_train_node2_gciql_chunk_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/eval/lewm_4tasks/20260823_eval_yb_lewm_4tasks.sh` | LeWM-4Tasks 主评测 |
| `exp/eval/ogbench_env_8tasks/20260823_eval_node2_ogbench_env_8tasks.sh` | OGBench-Env-8Tasks 主评测 |
| `exp/train/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 顺序复现 LeWM-4Tasks 四种训练设计 |
| `exp/train/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 顺序复现 OGBench-Env-8Tasks 四种训练设计 |
| `exp/eval/lewm_4tasks/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 复现 LeWM-4Tasks 主评测矩阵 |
| `exp/eval/ogbench_env_8tasks/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 复现 OGBench-Env-8Tasks 主评测矩阵 |

## 服务器与 GitHub

- 公共服务器路径仍统一记录在 `scripts/client_env.sh`；实验 Bash 先设置 `CLIENT_ID` 再 source。
- 远程 GitHub `https://github.com/pickxiguapi/ogbench` 的 `main` 是唯一代码基准。
- 服务器无法 pull 时，先在本地对齐远程 `main`，再将本地代码 rsync 到服务器。
- `master` 只用于跟踪上游 `https://github.com/seohongpark/ogbench`，不得提交本地方法代码。
