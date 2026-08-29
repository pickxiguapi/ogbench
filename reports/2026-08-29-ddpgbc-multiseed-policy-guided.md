# 2026-08-29 DDPG+BC Chunk Policy 多 Seed 评测

## 结论

Independent GCIQL-Chunk DDPG+BC 在三个 policy training seeds 和三个 evaluation seeds 上没有出现灾难性失败。Policy-only 的四任务宏平均为 90.39%，Guided 为 92.39%；Guidance 平均提高 2.00 个百分点，并将 training-seed 宏平均标准差从 1.08 降到 0.35。

Guidance 的作用存在稳定的任务差异：PushT、Reacher 分别平均提高 8.22、8.44 个百分点；Cube、TwoRoom 分别平均下降 6.00、2.67 个百分点。九个 seed 组合中，Cube 和 TwoRoom 没有一次因 Guidance 提升，Reacher 每次均提升。因此 DDPG+BC 是稳定的 policy prior，但当前 LeWM Guidance 不应无条件覆盖所有任务。

## 设置

- 服务器：A800 node4
- GitHub main commit：`327678b`
- Policy：independent GCIQL-Chunk DDPG+BC，chunk 5，alpha 1，batch 256，100k steps，`p_aug=0.5`
- Policy training seeds：`0, 42, 777`
- Evaluation seeds：`0, 1, 42`
- 每格：50 episodes，goal offset 25，budget 50
- Guided：seed3072 LeWM，CEM300x5，H5/RH1，top-k 30，min-over-horizon
- Policy 与 Guided 使用相同 evaluation seed；planner 使用 paired keys

正式 Bash：`exp/eval/lewm_4tasks/20260829_eval_node4_ddpgbc_multiseed_policy_guided.sh`

结果根目录：

`/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-actor-ablation/evals/2026-08-29_ddpgbc_multiseed_policy_guided/`

## 每个 seed 组合的四任务宏平均

| 模式 | Policy seed | Eval seed 0 | Eval seed 1 | Eval seed 42 | 三个 Eval seeds 均值 |
|---|---:|---:|---:|---:|---:|
| Policy-only | 0 | 90.5 | 88.5 | 89.0 | 89.33 |
| Policy-only | 42 | 92.5 | 90.0 | 88.5 | 90.33 |
| Policy-only | 777 | 91.5 | 91.5 | 91.5 | 91.50 |
| Guided | 0 | 93.5 | 90.5 | 93.5 | 92.50 |
| Guided | 42 | 94.0 | 90.5 | 93.5 | 92.67 |
| Guided | 777 | 91.5 | 91.5 | 93.0 | 92.00 |

## 跨 policy/evaluation seeds 的逐任务结果

| 模式 | Cube | PushT | Reacher | TwoRoom | 四任务宏平均 |
|---|---:|---:|---:|---:|---:|
| Policy-only | 94.44 | 75.56 | 91.56 | 100.00 | 90.39 |
| Guided | 88.44 | 83.78 | 100.00 | 97.33 | 92.39 |
| Guided - Policy | -6.00 | +8.22 | +8.44 | -2.67 | +2.00 |

## 按 policy training seed 汇总

| 模式 | Seed 0 | Seed 42 | Seed 777 | Training-seed mean ± std |
|---|---:|---:|---:|---:|
| Policy-only | 89.33 | 90.33 | 91.50 | 90.39 ± 1.08 |
| Guided | 92.50 | 92.67 | 92.00 | 92.39 ± 0.35 |

三个 policy seeds 的平均 Guidance 增益分别为 +3.17、+2.33、+0.50 个百分点。Guidance 对 seed777 的总体收益已经接近零，原因是 policy 本身较强，而 planner 对 Cube/TwoRoom 的损伤抵消了 PushT/Reacher 的收益。

## 与现有 AWR 结果的关系

- 旧的 alpha1 单 seed 对照中，AWR 因 Reacher=0% 使四任务平均只有 69%；本次 DDPG+BC 三个 seeds 均无类似塌缩。
- 现有 AWR alpha3 seeds 131/132 在 eval seed42 下，Policy-only 平均87.5%、Guided平均94.5%。本次 DDPG+BC 在 eval seed42 下，三个 policy seeds 的 Policy-only 平均89.67%、Guided平均93.33%。
- 两组 AWR 使用不同 alpha、policy seeds，不能把 94.5% vs 93.33% 解释成严格 actor-loss 差异；可以确认的是 DDPG+BC policy 更稳定，而当前 Guided 最佳点仍没有明显超过好 seed 的 AWR。

## 下一步

1. 将 DDPG+BC 作为稳定 policy prior，AWR 保留为性能消融。
2. 不再无条件对四任务全部启用 Guidance。优先验证一个 task-agnostic safety gate，例如当 CEM 第一块动作偏离 policy mode 超过阈值时回退 policy。
3. gate 阈值只用 eval seed0 选择，eval seeds1/42作为锁定测试，避免按测试任务结果后验选择 Policy/Guided。

