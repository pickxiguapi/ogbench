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
| A | J0，环境动作校准 | 验证新 evaluator 与 policy-only 等价 | 待运行 |
| A | J5/min/σ1 | 直接迁移 LeWM 四任务最高分配置 | 待运行 |
| B | J1/min/σ1 | 测试最保守的一轮 world-model 修正 | 待运行 |
| B | J5/terminal/σ1 | 分离 min-over-horizon 的贡献 | 待运行 |

后续只根据前两轮证据增加 σ、J、policy population、执行频率或安全门控消融，避免无信息的全排列搜索。
