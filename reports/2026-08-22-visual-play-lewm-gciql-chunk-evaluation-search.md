# 2026-08-22 Visual Play LeWM + GCIQL-Chunk 评估搜索

## 目标

在已经完整训练到 500k 的 Visual Play 四个 LeWM 与 GCIQL-Chunk-AWR policy 上，系统搜索不会破坏强 policy 的 LeWM 结合方式。所有筛选使用相同四环境、相同内部任务、seed42，并先以每内部任务10 episodes筛选，再对最高分配置扩大到50 episodes。

## 已知基线

| 方法 | Single | Double | Triple | Scene | 平均 |
|---|---:|---:|---:|---:|---:|
| Vanilla LeWM，旧 J30/terminal | 0.0 | 0.0 | 0.4 | 0.0 | 0.1 |
| 旧 GCIQL-Chunk proposal + LeWM，J30/terminal | 0.0 | 0.0 | 0.8 | 1.2 | 0.5 |
| GCIQL-Chunk policy-only，500k | 68.0 | 29.2 | 20.0 | 47.6 | 41.2 |

说明：用户给出的八任务 policy-only 平均为46.25%；上表只计算四个 Play 环境，平均为41.2%。

## 首轮审计发现

1. 旧 Visual hybrid 仍使用30轮 CEM 和 terminal cost，没有迁移 LeWM 四任务高分方案的 J5/min-over-horizon。
2. 旧 hybrid 把 GCIQL-Chunk 的环境动作直接当作 LeWM 标准化动作。四个 Play 数据集的动作标准差约为0.25–0.65，第五维均值约为0.08–0.17；因此旧路径会对 actor proposal 再做一次错误的缩放和平移。
3. 新 evaluator 显式支持 `proposal_action_space=environment`，先用数据 scaler 的 `transform` 进入 planner，执行前再 `inverse_transform`，从而保持 actor 动作不变。

## 分阶段搜索

| 阶段 | 配置 | 目的 | 状态 |
|---|---|---|---|
| A | J0，环境动作校准 | 验证新 evaluator 与 policy-only 等价 | 完成：60/32/20/48，平均40.0 |
| A | J5/min/σ1 | 直接迁移 LeWM 四任务最高分配置 | 完成：10/0/0/8，平均4.5 |
| B | J1/min/σ1 | 测试最保守的一轮 world-model 修正 | 完成：50/6/8/36，平均25.0 |
| B | J5/terminal/σ1 | 分离 min-over-horizon 的贡献 | 待运行 |
| B | 32个真实 policy chunks + LeWM H1选择 | 只在 actor 支持集内筛选，避免 CEM elite 均值与模型外推 | 完成：22/18/16/44，平均25.0 |
| B | 32个真实 policy chunks + 原生Q选择 | 区分 LeWM 排序误差与 actor 采样本身 | 完成：28/34/18/46，平均31.5 |
| C | LeWM选择32个真实 chunks，逐原子动作重规划 | 降低误选 chunk 的持续伤害，并增加闭环反馈 | 完成：18/8/8/22，平均14.0 |
| C | Policy mode，逐原子动作重规划 | 判断高频反馈本身是否有益 | 完成：46/28/12/26，平均28.0 |
| C | Mode init + CEM300×1，σ0.1，H1/min | 只允许 world model 做局部残差修正 | 完成：50/34/18/52，平均38.5 |
| C | Mode init + CEM300×1，σ0.05，H1/min | 进一步收紧残差以保住 policy | 完成：54/28/20/54，平均39.0 |
| C | 4个近-mode真实 chunks + LeWM H1选择 | 降低 actor 支持集内的选择过优化 | 完成：56/34/18/58，平均41.5 |
| D | 当前最佳 K4/temp0.05 扩大到50 ep/task | 检验10-episode筛选收益是否稳定 | 运行中 |
| D | K2/temp0.05 | 进一步降低选择干预率 | 运行中 |

后续只根据前两轮证据增加 σ、J、policy population、执行频率或安全门控消融，避免无信息的全排列搜索。

## 已完成结果

| 配置 | Single | Double | Triple | Scene | 平均 | 输出目录 |
|---|---:|---:|---:|---:|---:|---|
| GCIQL-Chunk mode，J0，动作空间已校准，10 ep/task | 60.0 | 32.0 | 20.0 | 48.0 | 40.0 | `20260822_policy_equivalence_j0_envscale_h5_rh1_ab5_ep10_seed42` |
| GCIQL-Chunk init + CEM300×1，σ1，H5/min，10 ep/task | 50.0 | 6.0 | 8.0 | 36.0 | 25.0 | `20260822_gciqlchunk_envscale_mincost_cem300x1_sigma1_h5_rh1_ab5_ep10_seed42` |
| GCIQL-Chunk init + CEM300×5，σ1，H5/min，10 ep/task | 10.0 | 0.0 | 0.0 | 8.0 | 4.5 | `20260822_gciqlchunk_envscale_mincost_cem300x5_sigma1_h5_rh1_ab5_ep10_seed42` |
| LeWM H1/min 从32个真实 policy chunks 中选择，temperature0.1，10 ep/task | 22.0 | 18.0 | 16.0 | 44.0 | 25.0 | `20260822_lewm_select_policy32_temp01_h1_mincost_ep10_seed42` |
| 原生保守Q从32个真实 policy chunks 中选择，temperature0.1，10 ep/task | 28.0 | 34.0 | 18.0 | 46.0 | 31.5 | `20260822_nativeq_select_policy32_temp01_h1_ep10_seed42` |
| LeWM H1/min 从32个真实 policy chunks 中选择，逐原子动作重规划，10 ep/task | 18.0 | 8.0 | 8.0 | 22.0 | 14.0 | `20260822_lewm_select_policy32_temp01_h1_atomic_replan1_ep10_seed42` |
| GCIQL-Chunk mode，逐原子动作重规划，10 ep/task | 46.0 | 28.0 | 12.0 | 26.0 | 28.0 | `20260822_policy_mode_atomic_replan1_ep10_seed42` |
| GCIQL-Chunk mode + CEM300×1，σ0.1，H1/min，10 ep/task | 50.0 | 34.0 | 18.0 | 52.0 | 38.5 | `20260822_gciqlchunk_envscale_mincost_cem300x1_sigma01_h1_rh1_ab5_ep10_seed42` |
| GCIQL-Chunk mode + CEM300×1，σ0.05，H1/min，10 ep/task | 54.0 | 28.0 | 20.0 | 54.0 | 39.0 | `20260822_gciqlchunk_envscale_mincost_cem300x1_sigma005_h1_rh1_ab5_ep10_seed42` |
| LeWM H1/min 从4个近-mode真实 policy chunks 中选择，temperature0.05，10 ep/task | 56.0 | 34.0 | 18.0 | 58.0 | 41.5 | `20260822_lewm_select_policy4_temp005_h1_mincost_ep10_seed42` |

J0 与完整500k policy-only 的 68/29.2/20/47.6（平均41.2）一致到10-episode抽样误差范围，证明新的动作空间转换恢复了真实 actor 行为。从 J0 到 J1 再到 J5，平均分为40.0、25.0、4.5，呈随 CEM 修正轮数增加而恶化的清晰趋势；剩余主因是 CEM 对强 policy 的离分布修正，而不是 checkpoint、数据或 evaluator 对齐失败。
