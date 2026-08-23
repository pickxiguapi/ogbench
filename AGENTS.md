# OGBench Local Rules

本文件作用于 `ogbench/`；与上层 `AGENTS.md` 同时生效。

## 最终方法边界

- LeWM世界模型训练入口只有 `impls/train_lewm_jax.py`。
- GCIQL-Chunk 训练入口只有 `impls/train_gciql_chunk.py`。
- GCIQL-Chunk 表征模式固定为 `independent`、`pi`、`qv`、`all`。
- `pi`、`qv`、`all` 必须加载已经训练好的 LeWM-JAX，并冻结 encoder、projector、predictor 与 batch statistics；只训练策略侧仍可训练的 encoder 和 downstream heads。
- 正式共享消融默认 `p_aug=0.0`。如启用增强，必须先增强同一批像素，再同时送入 pixel encoder 和 frozen LeWM encoder，禁止只增强其中一个分支，注意，如果使用`pi`、`qv`、`all`增强，前提肯定是LeWM本身也被增强训练。
- 执行阶段，也有三种方式：1️⃣ 纯Planning，就是调用LeWM来决策 2️⃣ 纯Policy，就是直接使用Chunk Level的action来决策，跟LeWM没关系 3️⃣ Guidance：GCIQL-Chunk proposal 初始化第一个 action block，LeWM min-over-horizon CEM 完成后续优化。

## LeWM-JAX 4Tasks 三训练 Seed Checkpoint

以下为 LeWM-JAX、IMPALA-Small、10 epoch、batch size 128、frameskip 5、history 3、SigReg 0.09 的主表 checkpoint。表中 SR 是当前实验主表记录值；`Seed 0/42` 位于英博云，`Seed 3072` 的原始训练产物位于 Server 23。

| Training Seed | Task | 主表 SR | 服务器 | Checkpoint |
|---:|---|---:|---|---|
| 0 | PushT | 88 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_pusht_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | OGB Cube | 60 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_cube_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | Reacher | 80 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_reacher_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | TwoRoom | 90 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_tworoom_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | PushT | 90 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_pusht_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | OGB Cube | 68 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_cube_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | Reacher | 80 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_reacher_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | TwoRoom | 88 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_tworoom_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 3072 | PushT | 90 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | OGB Cube | 70 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | Reacher | 86 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | TwoRoom | 86 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |

Seed 3072 四个 checkpoint 有以下镜像；已对英博云、Server 23、A800 node1 和 node2 的文件做 SHA-256 校验，逐任务一致：

| 服务器 | 镜像根目录 |
|---|---|
| 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-seed3072-s23/` |
| A800 node1 | `/data-training/yyf/lewm-jax-seed3072/` |
| A800 node2、node3、node4 | `/data-training/yyf/models/lewm-jax-seed3072/` |

注意：Seed 3072 的主表 SR 与 Server 23 的原始 `eval_cem300x30_seed42/` JSON 精确对应。Seed 0/42 的 PushT、Reacher、TwoRoom 可与英博云相邻评测 JSON 对应，但两个 Cube checkpoint 当前相邻的 `eval_cem300x30_seed42_20260822/cube.json` 分别记录 40 和 42，而非主表的 60 和 68；使用 Cube 主表数字前必须先核对评测轮次，不能把 training seed 与其他评测结果混用。

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
| `exp/train/20260823_train_s23_lewm_cube_pusht_seeds42_777.sh` | Server23 GPU2–5 训练 Cube/PushT × seed42/777 LeWM-JAX |
| `exp/train/20260823_train_yb_gciql_chunk_4tasks.sh` | 英博云训练 LeWM-4Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/train/20260823_train_node2_lewm_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks canonical LeWM-JAX |
| `exp/train/20260823_train_node2_gciql_chunk_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/eval/lewm_4tasks/20260823_eval_yb_lewm_4tasks.sh` | LeWM-4Tasks 主评测 |
| `exp/eval/lewm_4tasks/20260823_eval_s23_lewm_seed666.sh` | Server23 四卡评测 seed 666 的 LeWM-4Tasks checkpoint |
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
