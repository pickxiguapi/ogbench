# K10 subgoal 与 final goal 的 staged planning 实验

## 结论

在当前 LeWM-4Tasks、policy seed 777、evaluation seed 42、每任务 50 episodes
的设定下，最高分组合不是始终使用 final goal 或始终使用 K10 subgoal，而是：

- Goal offset 为 25：直接使用 final-goal guided CEM（H5）。
- Goal offset 大于 25：先使用 K10 8-sample path-medoid subgoal guided CEM
  （H2），当名义剩余距离为 10 步时切换到 final-goal guided CEM（H5）。
- 两个阶段都使用 shared-all GCIQL-Chunk seed777 policy mode 初始化、MoH、
  CEM 300 samples × 30 iterations、RH1、action block 5。

推荐组合的结果为：

| Goal/Budget | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| 25/50（始终 final goal） | 82 | 94 | 100 | 100 | 94.0 |
| 50/100（K10 → final，window=10） | 82 | 86 | 100 | 100 | 92.0 |
| 75/150（K10 → final，window=10） | 78 | 86 | 100 | 100 | 91.0 |

三个距离的宏平均为 **92.33**。

## Apples-to-apples 基线

所有结果使用同一版 evaluator、相同 policy/LeWM checkpoint、CEM 300×30、
MoH 和 evaluation seed 42。

| 方法 | Goal/Budget | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---|---:|---:|---:|---:|---:|
| 始终 final goal，H5 | 25/50 | 82 | 94 | 100 | 100 | 94.0 |
| 始终 final goal，H5 | 50/100 | 68 | 56 | 100 | 100 | 81.0 |
| 始终 final goal，H5 | 75/150 | 78 | 36 | 98 | 100 | 78.0 |
| 始终 K10，单样本，H2 | 25/50 | 74 | 94 | 82 | 100 | 87.5 |
| 始终 K10，单样本，H2 | 50/100 | 76 | 74 | 94 | 100 | 86.0 |
| 始终 K10，单样本，H2 | 75/150 | 72 | 70 | 100 | 100 | 85.5 |
| 始终 K10，8-sample medoid，H2 | 25/50 | 76 | 90 | 98 | 100 | 91.0 |
| 始终 K10，8-sample medoid，H2 | 50/100 | 72 | 78 | 100 | 100 | 87.5 |
| 始终 K10，8-sample medoid，H2 | 75/150 | 72 | 70 | 98 | 100 | 85.0 |

8-sample path medoid 主要修复了 Reacher 的单样本方差，但不保证每个任务都提升；
因此它不能独立解决长距离控制。真正显著的增益来自 episode 内 target/horizon
切换：远段用容易到达的 local latent target，近段用无预测偏差的 final latent target。

## Final-goal window 消融

`final-goal window = w` 表示在第 `goal_offset - w` 个环境步，从 K10/H2
切换到 final-goal/H5。

| Final window | 50/100 | 75/150 |
|---:|---:|---:|
| 5 | 89.5 | 90.5 |
| **10** | **92.0** | **91.0** |
| 15 | 91.5 | 89.0 |
| 25 | 88.0 | 88.5 |

切换过早会让 PushT 再次暴露在远距离 final-goal planning 下；切换过晚则缺少
足够的 final-goal refinement。当前 action block 和 subgoal horizon 都以 5 为粒度，
window=10 恰好保留两个名义 action block 的 final-goal refinement 区间。

## 实现

- `impls/latent_subgoal_runtime.py`：恢复 configurable multi-sample inference；
  LatentPathFlow 先对完整 `[z_{t+5}, z_{t+10}]` 路径选 medoid，再取 K10 waypoint。
- `impls/lewm_jax/planner.py`：增加 `StagedLeWMCEMPolicy`，在 action-block
  边界从 local planner 切换到 final planner。
- `impls/eval_lewm_4tasks.py`：增加 `--num-samples` 和
  `--final-goal-switch-steps`，并在 JSON 中记录 staged 配置。
- `exp/eval/lewm_4tasks/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_ns8_staged_guided_long_distance.sh`：
  node4 八卡正式评测 Bash，默认 final window 为 10。

## 局限

当前窗口消融和最终分数都来自同一个 evaluation seed 42。它已经证明 staged
组合在该固定正式 benchmark 上优于单一目标，但若要报告统计显著性，还应增加多个
evaluation seed，并固定 window=10，不再基于新增 seed 调参。
