# 2026-08-23 LeWM + GCIQL-Chunk shared-π 三组评测

## 实验设置

英博云四个 `shared_pi_only` checkpoint 均完整训练到 100k；仅 GCIQL-Chunk actor 使用冻结的 Server23 seed3072 LeWM post-projector 表征，Q/V 保留独立 IMPALA Small。评测统一使用 50 episodes、evaluation seed42、goal offset25、budget50，action chunk 长度5。

| 方法 | 配置 |
|---|---|
| π-only | actor deterministic mode，完整执行5步 chunk |
| CEM-only | seed3072 LeWM epoch10，CEM300×5，H5/RH1，topk30，min-over-horizon |
| CEM+π | 与 CEM-only 完全相同，仅用 shared-π mode 初始化首个 action block，paired planner keys |

## 结果

| 方法 | PushT | Cube | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| π-only | 64.0 | 92.0 | 56.0 | 100.0 | 78.0 |
| CEM-only | 84.0 | 76.0 | 100.0 | 94.0 | 88.5 |
| **CEM+π** | **90.0** | **90.0** | **100.0** | **100.0** | **95.0** |

CEM+π 相对 CEM-only 在 PushT/Cube/Reacher/TwoRoom 上分别改变 `+6/+14/0/+6`，宏平均提午6.5个百分点；相对 π-only 提午17个百分点。shared-π 对 Cube 的强先验弥补了 CEM-only 的短板，CEM 则显著修复了 π-only 的 PushT 与 Reacher。这组结果支持“π 提供数据支持内初值，LeWM-CEM 负责长程优化”的互补结构。

## 复现与输出

- π-only Bash：`scripts/eval/20260823_eval_yb_lewm_gciql_chunk_pi_shared_policy_four_tasks_s100k.sh`
- CEM-only Bash：`scripts/eval/20260823_eval_yb_lewm_cem_only_four_tasks_e10.sh`
- CEM+π Bash：`scripts/eval/20260823_eval_yb_lewm_cem_with_gciql_chunk_pi_shared_four_tasks_s100k.sh`
- π-only 输出：`/root/data/yyf/lewm-gciql-chunk-shared-evals/20260823_shared_pi_only_policy_s100k_ep50_seed42/`
- CEM-only 输出：`/root/data/yyf/lewm-gciql-chunk-shared-evals/20260823_cem_only_j5_h5_rh1_ep50_seed42/`
- CEM+π 输出：`/root/data/yyf/lewm-gciql-chunk-shared-evals/20260823_cem_with_shared_pi_j5_h5_rh1_ep50_seed42/`

首轮启动时英博云 overlay `/tmp` 满，导致 LanceDB 打开失败和一个 CEM 任务无法创建临时目录。三个正式 Bash 均已将 `TMPDIR` 指向数据盘 `/root/data/yyf/tmp`，之后完整重跑；上表只使用重跑后完整落盘的12个 JSON。
