# 固定 200k checkpoint 的纯 LeWM + CEM 搜索结论

## 结论

固定 `2026-08-22_node2_LeWMJAX_*_s200k_s3072_fs5_h3_sigreg009` 的 200k checkpoint 时，纯 LeWM + CEM 无法在 Visual Cube Single Play 上获得可测成功率。新做的约 390 个 episode 全部失败；即使加入训练一致的 history3、合法动作边界、真实离线动作块、状态/目标条件轨迹检索以及物体位置 latent probe，方块仍几乎完全不动。

因此不再把当前纯规划配置扩展到八环境正式评测。代表环境已经连续给出零信号，全量评测不能区分方案，只会增加计算量。

## 已排除的简单原因

| 方向 | 配置 | Cube Single Play | Scene Noisy | 结论 |
|---|---|---:|---:|---|
| 降低 rollout 长度 | H1，1024×3，sigma0.5 | 0/25 | 0/25 | 不是单纯的 H5 累积误差 |
| 原子步重规划 | H1，execution=1 | 0/25 | 0/25 | 更慢且无收益 |
| 增大探索 | H1，sigma1.0 | 0/25 | 0/25 | 无收益 |
| 中等 horizon | H2 | 0/25 | 0/25 | 无收益 |
| 合法动作边界 | planning 前按环境 bounds 裁剪 | 0 | 0 | 修复了评测一致性，但不能解决控制问题 |
| 时间相关动作 | 块内恒定动作 | 0/25 | — | 降维仍无收益 |
| 真实动作块初始化 | 4096 empirical chunks | 0/25 | — | 单块行为支持不足 |
| 训练一致历史 | history3，H1 | 0/50 | — | 修复单帧评测失配后仍为 0 |
| history3 长 rollout | H3/H5，块内恒定 | 0/50 | — | 长预测也无收益 |

上述前两组代表环境筛选共 200 episodes；后续 Cube Single 消融和诊断共约 190 episodes，均为 0。

## 关键诊断

### 1. CEM 在移动机械臂，但没有操作物体

history3/H1 诊断中：

- 平均绝对动作约 0.179，动作不接近零。
- 动作边界饱和率为 0。
- 每个 episode 末端执行器累计移动约 0.28–0.54 m。
- 五个任务的方块位移都只有约 `3.93e-5 m`，等于数值噪声。

视频显示机械臂在方块上方摆动，但不会形成“接近—抓取—搬运”的连续行为。整图 latent goal cost 可以通过改变机器人外观下降，却没有带来物体进展。

### 2. 物体位置可从 frozen latent 读出，但 rollout 会被搜索利用

用 20k 个离线样本对冻结 LeWM latent 拟合方块 xyz 线性读出器，测试集 R² 为：

- x: 0.9894
- y: 0.9931
- z: 0.9909

说明 encoder latent 确实包含精确物体位置。将 CEM cost 改成预测方块位置到目标位置的标准化距离后，机械臂动作更大、路径更长，但真实方块仍是零位移。这表明 CEM 找到了“模型预测方块会动、真实环境中不接触”的动作，即典型 model exploitation。

### 3. 更强的行为支持仍未恢复接触操作

依次测试：

1. 从离线数据随机采样完整 H5（25 步）真实动作计划，World Model 选最佳原始计划；
2. 在 frozen encoder latent 中检索当前状态最近的 32768 个离线状态，再选真实 H5 计划；
3. 扩大到 131072 状态，并加入强近邻秩正则；
4. 同时匹配当前图像和同轨迹 100 步后的目标图像；
5. 将选中计划从每 5 步重规划改成完整执行 25 步；
6. 使用 H20 真实计划、目标偏移 200 步并完整执行 100 步。

这些方案总体成功率仍为 0。执行 25 步的版本仅在一个 task 中出现约 0.73 mm 方块位移，远低于任务成功所需位移；执行 100 步后该信号也没有复现。

## 对当前 checkpoint 的判断

该 checkpoint 的训练目标是单步 `num_preds=1` latent prediction，训练数据时间尺度为 action-block5/history3。它可以编码物体位置，也能在离线分布上获得较低 prediction loss，但没有提供足够可靠的接触动力学供在线优化器反复求最优。CEM 的 optimizer's curse 会优先选择预测误差最大的候选，而不是真实可执行的抓取轨迹。

旧的全八环境纯 CEM 评测在 2000 episodes 中只有 Scene Noisy 的 task1_open 成功 4 次，总体 0.2%；本轮所有更严格配置都没有复现正成功信号。因此旧的 4 次成功更像偶发行为，而不是可调成高成功率的稳定基线。

## 真正可能提升成功率的下一步

如果 checkpoint 必须保持不变：

1. 使用 GCIQL-Chunk/HIQL-Chunk policy proposal 或 Q/value 约束候选。这不再是“纯 World Model”，但现有证据显示这是最现实的高成功率路径。
2. 使用显式任务阶段控制器或 privileged state shaping。这也不再是纯视觉 LeWM+CEM。

如果坚持纯 World Model + planning：

1. 需要重新训练 World Model，而不是继续调 CEM。
2. 训练目标至少加入 multi-step rollout、object/contact-aware prediction 和 planner-aware uncertainty/OOD penalty。
3. 最好把空间物体 token 与机器人 token 分开，使 cost 能直接针对被操作物体，而不是整图单一 latent。
4. 用模型 ensemble 或不确定性惩罚抑制 CEM 利用单模型盲区。

在这些改变之前，不建议继续对同一 200k checkpoint 扫更多纯 CEM 超参数。
