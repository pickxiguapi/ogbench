# Frozen LeWM Latent Subgoal GCBC（K=10）定稿方案

## 目标与方法边界

本方法在每个任务已经训练并冻结的 LeWM 表征空间中，训练一个纯监督的高层 subgoal generator。它不训练或微调 LeWM，不使用 Q、value、reward、advantage weighting，也不从 subgoal latent 解码图像。

四个任务分别使用各自的 LeWM checkpoint 和独立的 generator。虽然四个 latent 都是 192 维，但它们不属于共享坐标系。

## 数据

每个源数据 row 已离线编码为：

\[
z_t=\phi(s_t)\in\mathbb R^{192}.
\]

正式 latent cache 为 float32 HDF5，包含 `z`、`episode_idx`、`step_idx`、`ep_offset`、`ep_len` 和 checkpoint SHA-256。训练开始时把完整 `z` 和 trajectory 索引加载到内存并放入单张 GPU，训练阶段不再解码图像或调用 LeWM encoder。

训练/验证按照 episode 固定划分为 95%/5%，`split_seed=0`。禁止按照 row 随机划分，以免相邻 latent 泄漏到验证集。

## HIQL 风格上层 Goal Sampling

固定 subgoal horizon：

\[
K=10.
\]

从训练 episode 的所有非终止 row 中均匀采样当前索引 \(t\)，令该 episode 的最后一个 row 为 \(T\)。按照 HIQL 上层默认设置，从同轨迹未来均匀采样最终目标：

\[
g\sim\operatorname{Uniform}\{t+1,\ldots,T\}.
\]

配置语义为：

```text
actor_p_trajgoal   = 1.0
actor_p_randomgoal = 0.0
actor_p_curgoal    = 0.0
actor_geom_sample  = false
subgoal_steps      = 10
```

监督目标索引为：

\[
u=\min(t+10,g).
\]

因此训练映射为：

\[
(z_t,z_g)\longrightarrow z_{\min(t+10,g)}.
\]

当最终目标在未来 10 步内时，target 直接等于 \(z_g\)；当最终目标更远时，target 等于同一轨迹上的 \(z_{t+10}\)。第一版不采样其他轨迹的 random goal，因为纯 GCBC 没有 HIQL 的 Q/advantage weighting，无法为跨轨迹目标判断当前轨迹 target 是否合理。

## 网络

generator 直接预测完整 subgoal latent，不使用 residual prediction：

\[
\hat z_{\mathrm{sg}}=G_\theta(z_t,z_g)\in\mathbb R^{192}.
\]

网络输入只是 \([z_t,z_g]\)，不加入时间距离、手工差分特征或 task embedding：

```text
concat(z_t, z_g)       384
Linear                 512
LayerNorm + SiLU
Linear                 512
LayerNorm + SiLU
Linear                 512
LayerNorm + SiLU
Linear                 192
```

输出不做 L2 normalization。每个任务单独训练一套网络。

## 损失

唯一优化目标是 raw LeWM latent 上的均方误差：

\[
\mathcal L_{\mathrm{MSE}}
=\frac{1}{192}\left\|G_\theta(z_t,z_g)-z_{\min(t+10,g)}\right\|_2^2.
\]

不加入 cosine loss。cosine similarity 只作为验证指标记录，因为 LeWM planner 使用原始 latent 的平方欧氏距离，cosine 无法约束预测 latent 的模长。

## 正式训练参数

| 参数 | 数值 |
|---|---:|
| train steps | 100,000 |
| batch size | 1,024 |
| hidden dimensions | 512, 512, 512 |
| optimizer | AdamW |
| peak learning rate | 3e-4 |
| weight decay | 1e-4（只作用于二维及以上权重） |
| warmup | 2,000 steps |
| schedule | cosine decay |
| final learning rate | 3e-5 |
| gradient clipping | 1.0 |
| dropout | 0 |
| precision | float32 |
| train/validation split | 95%/5% episodes |
| split seed | 0 |
| fixed validation pairs | 50,000 |
| log interval | 1,000 steps |
| validation interval | 5,000 steps |
| checkpoint interval | 25,000 steps |

第一轮四任务都使用 training seed 0。确认训练和闭环接口后，正式统计补 seed 42 和 777。主结果固定报告 step 100,000，不根据测试成功率挑 checkpoint。

## 验证指标

验证三元组在训练开始时固定生成，此后所有 checkpoint 使用同一组样本。记录：

```text
val/mse
val/mse_near       1 <= g - t <= 10
val/mse_medium     11 <= g - t <= 25
val/mse_far        g - t > 25
val/l2
val/cosine_similarity
val/prediction_norm
val/target_norm
```

如果 MSE 很低但闭环效果差，优先诊断多模态平均和 off-manifold prediction，而不是把主损失替换成 cosine。

## 闭环执行

上层每 10 个 atomic steps 生成一次 subgoal。低层仍可每 5 步重新规划，但在前后两个 5-step block 中必须追逐同一个固定 subgoal：

```text
t=0    G(z_t, z_g) -> z_subgoal
t=0-4  低层执行第一个 action block
t=5    重新观察，低层继续以同一个 z_subgoal 规划
t=5-9  执行第二个 action block
t=10   上层生成下一个 z_subgoal
```

不能在 t+5 重新生成一个新的 \(z_{t+15}\)，否则会重新引入滚动时域目标不断后移的问题。若环境在 10 步内提前达到最终目标，则直接终止。

## 必要对照

1. Global goal：低层直接追逐 \(z_g\)。
2. Oracle subgoal：使用真实 \(z_{\min(t+10,g)}\)。
3. Predicted subgoal：使用本方法输出。
4. Goal-only generator：只输入 \(z_g\)。
5. No-goal generator：只输入 \(z_t\)。

Oracle 与 predicted 的差距用于区分 generator 误差和低层控制误差。

## 首轮 predicted-subgoal CEM 闭环协议

第一轮只检验 predicted subgoal 能否改善纯 CEM，不加入 policy、Q、value 或 action proposal。CEM 搜索分布、warm start 和 LeWM rollout 完全保持原方法，只把 latent cost 的目标从最终目标 \(z_g\) 换成当前固定的 \(\hat z_{\mathrm{sub}}\)。

| 参数 | 数值 |
|---|---:|
| episodes | 50 |
| evaluation seed | 42 |
| goal offset | 25 atomic steps |
| evaluation budget | 50 atomic steps |
| CEM population | 300 |
| CEM iterations | 30 |
| top-k | 30 |
| planning horizon | 5 action blocks = 25 atomic steps |
| receding horizon | 1 action block = 5 atomic steps |
| cost | min over horizon latent L2 |
| subgoal refresh | 10 atomic steps |

每个 predicted subgoal 必须固定服务两个连续的 5-step replan；在第 10 个 atomic step 执行完以后，才从新的真实 observation 生成下一个 subgoal。LeWM checkpoint 与 generator 训练配置中记录的 SHA-256 必须精确匹配。

正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_yb_lewm_latent_subgoal_moh.sh`。

严格配对的 global-goal CEM 基线为 `exp/eval/lewm_4tasks/20260830_eval_yb_lewm_mixed_ckpts_moh.sh`，当前 seed42/50-episode 结果是 PushT 88、Cube 74、Reacher 100、TwoRoom 98。两种方法必须共享 dataset starts 和 paired planner keys，并额外报告逐 episode success flip。

## 2026-08-31 首轮闭环结果

正式 predicted-subgoal CEM 评测已在英博云完成。结果目录：

`/root/data/yyf/lewm-final/evals/lewm-4tasks/20260831_latent_subgoal_k10_moh_cem300x30_h5_rh1_ep50_seed42/`

| 任务 | Global-goal CEM | Predicted-subgoal CEM | 变化 | Subgoal-only success | Baseline-only success |
|---|---:|---:|---:|---:|---:|
| PushT | 88 | 90 | +2 | 5 | 4 |
| Cube | 74 | 64 | -10 | 1 | 6 |
| Reacher | 100 | 82 | -18 | 0 | 9 |
| TwoRoom | 98 | 84 | -14 | 0 | 7 |
| 宏平均 | 90 | 80 | -10 | 6 | 26 |

四任务总成功数从 `180/200` 降为 `160/200`。两组结果的 dataset seeds 已逐项验证完全一致，CEM 使用 paired plan keys；因此差值不是由换了 evaluation starts 造成的。

首轮结论：纯 CEM 能执行 predicted latent subgoal，且 PushT 有小幅收益，但“完全用 predicted subgoal 替换 global goal cost”不能作为通用方法。失败 episode 的 generator 都执行到 5 次更新，即用满 50-step budget；主要问题是闭环 current latent 离开 expert trajectory 后的分布偏移、确定性 MSE 输出可能处于多条路径之间，以及反复追逐局部 latent 时丢失 global-goal 约束。后续若继续该路线，应优先测试同时保留 global goal 的双项 cost，而不是增加 generator 训练步数。

为区分 generator 误差与 CEM 的 subgoal 可利用性，增加 oracle 对照：evaluation 已知 sampled dataset episode 和 start，因此在 goal offset25、K10 时依次使用同一 demonstration 的 `start+10`、`start+20`、`start+25` 图像，经同一个 frozen LeWM 编码成 GT latent waypoint。环境成功条件仍然只使用 `start+25` 的最终目标，不能把到达中间 waypoint 计为成功。正式入口为 `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_oracle_subgoal_moh.sh`。
