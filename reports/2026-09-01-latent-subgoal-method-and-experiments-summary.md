# 2026-09-01 Latent Subgoal：方法、实验与论文写作总结

> 本文汇总 2026-09-01 完成的 LatentPathFlow、LeWM-CEM、GCIQL-Chunk
> policy guidance、goal sampling 和长距离控制实验。除特别注明外，成功率均为
> `evaluation seed=42`、每任务 50 episodes，因此最小分辨率为 2 个百分点。
> 文中严格区分 `CEM 300×30` 与 `CEM 300×5` 两套协议，不应跨协议直接宣称
> 方法优劣。

## 1. 今日结论

1. **Subgoal predictor 是有价值的，但主要价值出现在远距离目标。** 在纯
   CEM、相同 H2/MoH 协议下，global goal 的四任务平均从 `25/50` 的 83.5
   降到 `50/100` 的 61.0 和 `75/150` 的 58.0；predicted K10 则为
   82.0、79.0、82.0。K10 在长距离分别提升 18 和 24 个百分点。
2. **时间对齐是必要条件。** K10 subgoal 必须用 H2 规划到 $t+10$，并在
   RH1 的每次重规划（每 5 个原子步）重新预测。曾经的 H5 terminal 和
   refresh=10 都属于错误协议，不能用于判断 idea 是否有效。
3. **短距离 `25/50` 不需要无条件使用 subgoal。** 强 actor-guided final-goal
   CEM 已达到 96.5；不过把 predictor 专门训练在 25 步内，并让 policy 看
   final goal、CEM 看 K10 subgoal，可达到今天最高的 97.0。
4. **goal-conditioned policy 与 CEM target 最好解耦。** 当前最稳定的组合是：
   policy 使用最终目标 $z_g$ 提供全局方向，CEM 使用预测 K10 subgoal 提供
   局部可达性。100/200 的 uniform-future predictor 上，该组合为 90.5；
   policy 也看 subgoal 时为 89.0。
5. **goal sampling 决定 predictor 的适用距离。** `goalmax25` 是很强的
   `25/50` 专用模型，但对 100-step condition goal 严重分布外，尤其 PushT
   从 uniform-future 的 80 降到 40。长距离必须使用覆盖完整 future range
   的 predictor，或显式训练多距离模型。
6. **K15 不如 K10。** 在纯 CEM/terminal 实验中，K15 的平均成功率为
   70.5、70.5、72.5，显著低于对应 K10 的 74.5、79.5、80.5。预测更远的
   waypoint 增加了误差，未换来更好的控制。
7. **预测路径的两个 waypoint 直接取 mean cost 没有增益。** K5/K10 对齐
   `path_mean` 为 84.0，低于只追踪 K10 的 MoH 87.5。当前不能直接把两个
   waypoint 等权平均作为主方法。
8. **增加 actor proposal 数量或复杂选择规则收益很小。** 短距离 H2/J5 下，
   policy mode 为 87.5；最好的人口变体为 population64、temperature0.3，
   仅到 88.5。Q/V 因而不进入当前主方法。
9. **最强的长距离 300×30 方案仍是 staged planning。** 远段使用 K10/H2，
   最后 10 步切换到 final-goal/H5，50/100 和 75/150 分别达到 92.0 和 91.0。

## 2. 方法定义

### 2.1 Frozen LeWM latent space

每个任务先训练 LeWM，然后完全冻结 encoder：

\[
z_t=\phi(s_t)\in\mathbb R^{192},\qquad z_g=\phi(g)\in\mathbb R^{192}.
\]

四个任务的 latent 都是 192 维，但来自不同 checkpoint，不应认为它们处于共享
坐标系。图像数据已提前编码成 float32 HDF5；训练 subgoal generator 时直接把
latent cache 与 episode 索引加载到内存/GPU，不再解码图像或调用 LeWM encoder。

| Task | Latent rows | Episodes | Train rows | Validation rows |
|---|---:|---:|---:|---:|
| Cube | 2,010,000 | 10,000 | 1,900,000 | 100,000 |
| PushT | 2,336,736 | 18,685 | 2,201,385 | 116,666 |
| Reacher | 2,010,000 | 10,000 | 1,900,000 | 100,000 |
| TwoRoom | 920,809 | 10,000 | 865,091 | 45,718 |

训练/验证按 episode 做 95/5 固定划分，`split_seed=0`，避免相邻 transition
泄漏。history 不足 3 帧时重复 episode 第一帧补齐。

### 2.2 LatentPathFlow target

正式设置为：

\[
K=10,\qquad C=\text{action block}=5,
\]

所以网络预测与决策边界对齐的两点路径：

\[
Z^*=\begin{bmatrix}
z_{\min(t+5,g)}\\
z_{\min(t+10,g)}
\end{bmatrix}\in\mathbb R^{2\times192}.
\]

条件输入为最近三帧 latent history 和最终目标 latent：

\[
H_t=[z_{t-2},z_{t-1},z_t],\qquad z_g.
\]

这里直接预测 latent path，不先预测像素状态再编码，不使用 residual prediction，
也不训练 inverse dynamics。

### 2.3 Conditional flow matching

训练只使用 flow matching loss：

\[
\epsilon\sim\mathcal N(0,I),\qquad \tau\sim U(0,1),
\]

\[
X_\tau=(1-\tau)\epsilon+\tau Z^*,\qquad v^*=Z^*-\epsilon,
\]

\[
\mathcal L_{\mathrm{flow}}=
\mathbb E\left[
\left\|v_\theta(X_\tau,\tau\mid H_t,z_g)-(Z^*-\epsilon)\right\|_2^2
\right].
\]

没有 $\mathcal L_{\mathrm{ID}}$、LeWM consistency loss、cosine loss 或 latent
mean/std 标准化。cosine similarity 只作为验证指标。

### 2.4 网络结构

当前网络采用 LeFlow 风格的 conditional Transformer encoder：

- noisy K5/K10 path 各自作为一个 token；
- latent 维度 `192 → 512`，加入可学习 waypoint position embedding；
- 3 帧 history 展平后单独投影，goal 和 flow time 分别投影；
- 三种 condition 经非线性 MLP 融合；
- 4 个 AdaLN Transformer encoder blocks；
- 8 heads，FFN 维度 2048，hidden width 512；
- sinusoidal time embedding 维度 64，直接编码 $\tau\in[0,1]$，不乘 1000；
- LayerNorm 后由 `512 → 192` velocity head 输出两个 waypoint 的速度。

### 2.5 训练与推理参数

| 参数 | 正式值 |
|---|---:|
| Train steps | 200,000 |
| Batch size | 1,024 |
| Optimizer | AdamW |
| Peak / final LR | `1e-4 / 1e-5` |
| Warmup | 5,000 |
| Weight decay | `1e-4` |
| Global grad clip | 1.0 |
| EMA | 0.9999 |
| Validation pairs | 10,000 |
| Flow solver | Euler |
| Flow integration steps | 16 |
| Training seed | 0 |
| History size | 3 |
| Validation `num_samples` | 8 |

四任务在 node4 四卡并行时，每个 200k run 约 23.6–26.5 分钟。由于 batch
是有放回随机采样，严格来说没有 epoch；200k×1024 相当于抽取 2.048 亿个训练
样本，对不同任务约为 93–237 次 nominal dataset pass。

推理可设置 `num_samples=N`。`N=1` 使用单条随机 flow trajectory；`N>1`
先生成 N 条完整 latent paths，再在展平的 K5/K10 path 上选择 medoid，最后由
planner 取完整路径或 K10 token。训练本身不会因为 `num_samples=8` 产生 8 个
监督目标；该参数只用于 validation/inference sampling。

## 3. Goal sampling 变体

### 3.1 Uniform future（完整范围）

先均匀采当前 transition $t$，再从同一 episode 的所有未来帧中均匀采 $g$：

\[
g\sim\mathrm{Uniform}\{t+1,\ldots,T\}.
\]

这是最接近 HIQL/HGCBC 上层 trajectory-goal sampling 的版本，覆盖完整 future
range，但 goal distance 的边缘分布不均衡。

### 3.2 Aligned future

保留相同 episode 内采样，但限制：

\[
(g-t)\bmod C=0,
\]

其中 $C=\text{action block}=5$，所以候选 goal 是

\[
t+5,t+10,t+15,\ldots.
\]

这解决了训练 goal 与 chunked controller 决策边界不一致的问题，但实验表明单独
加入对齐没有稳定提升。

### 3.3 Distance-balanced goalmax25

为 `25/50` 专门训练的版本先均匀采 goal distance：

\[
\delta\sim\mathrm{Uniform}\{5,10,15,20,25\},
\]

再从满足 $t+\delta<T_{\mathrm{episode}}$ 的合法 $t$ 中均匀采样，并令

\[
g=t+\delta.
\]

这既保证 action-block alignment，也保证五种距离获得相同训练权重。但它只覆盖
25 步内 condition goal，因此不应直接用于 50、75、100-step condition goal。

## 4. Planner 与 policy guidance

### 4.1 Global-goal LeWM-CEM

原始 planner 直接优化 LeWM rollout 与最终目标 latent 的距离：

\[
J_{\mathrm{global}}(a)=
\rho_h\left(\|\hat z_{t+h}(a)-z_g\|_2^2\right).
\]

`last` 只使用最后一个 rollout checkpoint；`MoH` 使用 min-over-horizon：

\[
J_{\mathrm{MoH}}(a)=
\min_h\|\hat z_{t+h}(a)-z_{\mathrm{target}}\|_2^2.
\]

### 4.2 K10 local-goal LeWM-CEM

Generator 在每次 RH1 replan 时，基于最新真实观测重新预测：

\[
\hat Z_t=[\hat z_{t+5},\hat z_{t+10}].
\]

K10-only planner 使用两 action blocks，即 H2：

\[
J_{K10}(a)=\rho_{h\in\{5,10\}}
\left(\|\hat z^{\mathrm{LeWM}}_{t+h}(a)-\hat z_{t+10}\|_2^2\right).
\]

RH1 表示只执行第一个 5-step action block，然后观察真实状态、重新预测 subgoal、
重新运行 CEM。

### 4.3 Path-aligned mean

今天还测试了把 LeWM 的两个 rollout checkpoint 分别与 K5/K10 对齐：

\[
J_{\mathrm{path}}(a)=\frac12\left(
\|\hat z^{\mathrm{LeWM}}_{t+5}-\hat z_{t+5}\|_2^2+
\|\hat z^{\mathrm{LeWM}}_{t+10}-\hat z_{t+10}\|_2^2
\right).
\]

该版本实现名为 `path_mean`，目前没有优于 K10+MoH。

### 4.4 Actor-guided CEM

Policy 使用 shared-all GCIQL-Chunk checkpoint，但当前只使用 actor，不使用 Q/V。
`mode` guidance 把 deterministic actor action chunk 放入 CEM 的 initial mean；其余
horizon 仍由 warm start/零均值初始化，随后完全由 LeWM cost 做 CEM 更新。

当前支持两个 policy goal：

- `guidance_goal_mode=subgoal`：policy 与 CEM 都看预测 K10 latent；
- `guidance_goal_mode=final`：policy 看最终 $z_g$，CEM 仍看预测 K10 latent。

第二种是今天新增的 decoupled global-policy/local-planner 形式：

\[
a_t^{\pi}=\pi(z_t,z_g),\qquad
a_t=\operatorname{CEMInit}\left(a_t^{\pi};\hat z_{t+10}\right).
\]

### 4.5 Staged planning

长距离 episode 的远段使用 K10/H2，接近目标后切换到无 predictor bias 的
final-goal/H5：

\[
\text{target}(t)=
\begin{cases}
\hat z_{t+10}, & \text{nominal remaining distance}>w,\\
z_g, & \text{nominal remaining distance}\le w.
\end{cases}
\]

当前最优 `final window` 为 $w=10$。

## 5. Predictor 离线结果

下表为 200k EMA checkpoint、validation `num_samples=8` path-medoid 的 joint
path MSE/cosine。不同 sampling 版本的 validation goal 分布不同，尤其 goalmax25
没有 far examples，因此不能把它的低 MSE 直接解释为全距离更准确。

| Task | Uniform future | Aligned future | Goalmax25 | K15 uniform |
|---|---:|---:|---:|---:|
| Cube | 0.0772 / 0.9591 | 0.0758 / 0.9597 | **0.0498 / 0.9735** | 0.1001 / 0.9470 |
| PushT | 0.0432 / 0.9771 | 0.0456 / 0.9758 | **0.0319 / 0.9830** | 0.0500 / 0.9735 |
| Reacher | 0.5410 / 0.7272 | 0.5650 / 0.7148 | **0.3944 / 0.8008** | 0.6371 / 0.6785 |
| TwoRoom | 0.9088 / 0.5028 | 0.9774 / 0.4687 | 0.8907 / 0.5077 | **0.8824 / 0.5214** |

Uniform K10 的 waypoint 指标进一步显示 K10 比 K5 更难：

| Task | K5 MSE / cosine | K10 MSE / cosine |
|---|---:|---:|
| Cube | 0.0471 / 0.9748 | 0.1073 / 0.9431 |
| PushT | 0.0319 / 0.9831 | 0.0546 / 0.9711 |
| Reacher | 0.4115 / 0.7926 | 0.6704 / 0.6618 |
| TwoRoom | 0.8959 / 0.5064 | 0.9216 / 0.4952 |

离线指标与闭环成功率并非单调对应：TwoRoom 的 cosine 最差但控制几乎总是 100，
说明任务容错、latent geometry 和 planner 可达性同样重要；不能只凭跨任务 MSE
判断 subgoal 方法好坏。

## 6. 纯 CEM 实验

### 6.1 时间对齐修正过程（诊断）

| K10 protocol | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| 错误：H5 terminal 对齐到约 $t+25$ | 62 | 14 | 24 | 58 | 39.5 |
| H2 terminal，但错误 refresh=10 | 76 | 70 | 28 | 96 | 67.5 |
| H2 terminal，正确 refresh=5 | 80 | 84 | 48 | 100 | 78.0 |
| H2 MoH，正确 refresh=5 | 76 | 90 | 62 | 100 | 82.0 |

这组结果说明早期低分主要由 planner/subgoal 时间错位造成，而不是 idea 本身必然
无效。修正后仍然偏低的 Reacher 才更可能由 predictor quality 或 policy/planner
分布偏移造成。

### 6.2 相同 H2/MoH 下的距离扩展（CEM 300×30）

| Goal/Budget | Target | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---|---:|---:|---:|---:|---:|
| 25/50 | Final goal | 76 | 76 | 96 | 86 | **83.5** |
| 25/50 | Predicted K10 | 76 | 90 | 62 | 100 | 82.0 |
| 50/100 | Final goal | 62 | 36 | 88 | 58 | 61.0 |
| 50/100 | Predicted K10 | 62 | 72 | 82 | 100 | **79.0** |
| 75/150 | Final goal | 76 | 4 | 88 | 64 | 58.0 |
| 75/150 | Predicted K10 | 62 | 72 | 94 | 100 | **82.0** |

核心趋势：final latent cost 随目标变远迅速失去局部指导性；K10 把每次规划目标
限制在当前状态附近，因此平均成功率对 goal distance 更稳定。收益主要来自 PushT
和 TwoRoom；Cube 是尚未稳定受益的例外。

### 6.3 最新 evaluator 的 cost/receding-horizon 消融（CEM 300×30，25/50）

| Target / planner | Last | MoH |
|---|---:|---:|
| Final goal，H5/RH1 | 59.0 | **89.5** |
| Final goal，H5/RH5 | 81.0 | **86.5** |
| Predicted K10，H2/RH1 | 74.5 | **85.5** |

MoH 对闭环 RH1 尤其重要。`last` 容易要求 rollout 在固定时间点精确落到目标，而
MoH 允许候选在 rollout 中更早到达目标。

### 6.4 K10 与 K15 terminal 对照（CEM 300×30）

| Predictor | 25/50 | 50/100 | 75/150 |
|---|---:|---:|---:|
| K10，H2/RH1 | **74.5** | **79.5** | **80.5** |
| K15，H3/RH1 | 70.5 | 70.5 | 72.5 |

因此当前保留 K10，不继续把固定 subgoal horizon 拉长。

### 6.5 Goal sampling：纯 CEM（CEM 300×5，H2/RH1/MoH）

| Sampling | 25/50 | 50/100 | 75/150 |
|---|---:|---:|---:|
| Uniform future | **84.0** | 81.5 | 81.0 |
| Aligned future | 83.0 | **84.5** | 81.0 |
| Goalmax25 | **91.5** | — | — |

Aligned future 没有稳定超过 uniform future；限制到 25 步并做 distance balance 对
25/50 明显有效，但不应将该结果外推到更长目标。

### 6.6 K5/K10 path mean

在 `25/50`、actor-guided、CEM 300×5 下：

| Cost | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| K10 + MoH | 76 | 90 | 84 | 100 | **87.5** |
| K5/K10 aligned mean | 78 | 84 | 74 | 100 | 84.0 |

等权 path mean 同时压低 PushT 与 Reacher。后续若利用完整 path，应考虑 waypoint
置信度加权、只约束首个 waypoint，或把 K5/K10 用作分阶段约束，而不是固定平均。

## 7. Policy-guided CEM 实验

### 7.1 Direct policy 与 policy checkpoint 选择

Shared-all GCIQL-Chunk direct policy 在 25/50 上：

| Policy seed | Cube | PushT | Reacher | TwoRoom | 平均 |
|---:|---:|---:|---:|---:|---:|
| 777 | 98 | 76 | 98 | 100 | **93.0** |
| 789 | 98 | 70 | 100 | 100 | 92.0 |

因此后续统一使用 seed777。相同 final-goal H2/J5 CEM 中，shared-all actor 为
90.5；independent AWR policy 为 89.0，independent DDPG+BC 为 88.0。shared-all
checkpoint 最适合当前 frozen LeWM latent bypass。

### 7.2 Final goal 与 K10：距离消融（CEM 300×5）

| Goal/Budget | Method | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---|---:|---:|---:|---:|---:|
| 25/50 | Final-goal actor CEM，H5 | 94 | 92 | 100 | 100 | **96.5** |
| 25/50 | K10 actor CEM，H2 | 76 | 90 | 84 | 100 | 87.5 |
| 50/100 | Final-goal actor CEM，H5 | 70 | 60 | 100 | 100 | 82.5 |
| 50/100 | K10 actor CEM，H2 | 76 | 76 | 98 | 100 | **87.5** |
| 75/150 | Final-goal actor CEM，H5 | 82 | 30 | 100 | 100 | 78.0 |
| 75/150 | K10 actor CEM，H2 | 74 | 76 | 98 | 100 | **87.0** |

这再次支持：短距离 final goal 足够；随着目标变远，K10 的局部可达 target 开始
获得明显优势。

### 7.3 `num_samples=1` 与 8-sample medoid（CEM 300×30）

| Goal/Budget | K10 single sample | K10 8-sample path medoid |
|---|---:|---:|
| 25/50 | 87.5 | **91.0** |
| 50/100 | 86.0 | **87.5** |
| 75/150 | **85.5** | 85.0 |

Multi-sample medoid 能减少部分任务的 flow 随机性，尤其修复短距离 Reacher，但
没有随距离稳定提升。它是可用的 inference-time robustness 技术，不是长距离问题的
根本解法。

### 7.4 Staged K10 → final（CEM 300×30）

| Goal/Budget | Always final | Always K10, N=8 | Staged, final window=10 |
|---|---:|---:|---:|
| 25/50 | **94.0** | 91.0 | 25/50 直接使用 final |
| 50/100 | 81.0 | 87.5 | **92.0** |
| 75/150 | 78.0 | 85.0 | **91.0** |

Final-window 消融：

| Final window | 50/100 | 75/150 |
|---:|---:|---:|
| 5 | 89.5 | 90.5 |
| **10** | **92.0** | **91.0** |
| 15 | 91.5 | 89.0 |
| 25 | 88.0 | 88.5 |

`window=10` 恰好保留两个 action blocks 的 final-goal refinement。切换过早会重新
暴露远距离 global-goal planning 问题；切换过晚则没有足够时间消除 predictor bias。

### 7.5 Actor-CEM 结合方式（25/50，CEM 300×5）

所有变体都不使用 Q/V：

| Guidance | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| Policy mode | 76 | 90 | 84 | 100 | 87.5 |
| Mode anchor | 74 | 90 | 78 | 100 | 85.5 |
| Mode + first-block std 0.3 | 78 | 84 | 80 | 100 | 85.5 |
| Population32, temp0.3 | 76 | 90 | 80 | 100 | 86.5 |
| Population64, temp0.3 | 82 | 90 | 82 | 100 | **88.5** |
| Population64, temp1.0 | 78 | 90 | 78 | 100 | 86.5 |
| Population128, temp0.3 | 78 | 90 | 82 | 100 | 87.5 |
| LeWM-select64 | 74 | 92 | 80 | 100 | 86.5 |
| LeWM-elite64, elite8 | 76 | 94 | 82 | 100 | 88.0 |

复杂 proposal 方法最多只提升 1 个平均百分点，因此论文主方法优先保留简单的
deterministic policy mode。

早期还测试过用 learned Q 从 policy proposals 中选择动作。该批次属于较早的评测
协议，只保留趋势，不与上表做逐项比较：

| Goal/Budget | Policy mode | Q-selected |
|---|---:|---:|
| 25/50 | **96.5** | 96.0 |
| 50/100 | 79.0 | **83.0** |
| 75/150 | **76.5** | 75.5 |

Q-selected 只在 50/100 提升，在另外两个距离不升反降，未表现出稳定收益。结合
actor-only CEM 已能取得更强结果，当前最终方法明确不使用 Q/V。

### 7.6 Goalmax25 专用 predictor（25/50）

| Controller | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| Pure CEM 看 K10 | 88 | 92 | 86 | 100 | 91.5 |
| Policy 与 CEM 都看 K10 | 96 | 96 | 86 | 100 | 94.5 |
| **Policy 看 final，CEM 看 K10** | **96** | **98** | **94** | **100** | **97.0** |

Reacher 单任务调参中，让 policy 更占主导最多达到 90；直接 policy 看预测 K10
为 92；最终采用 policy 看 $z_g$、CEM 看 K10 后达到 94。说明 Reacher 的关键
不是简单缩小 CEM 方差，而是保留准确的 global policy condition。

### 7.7 Decoupled guidance 的长距离结果（CEM 300×5）

Policy 始终使用最终 $z_g$，CEM 始终追踪相应 predictor 生成的 K10：

| Goal/Budget | Generator | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---|---:|---:|---:|---:|---:|
| 25/50 | Goalmax25 | 96 | 98 | 94 | 100 | **97.0** |
| 50/100 | Goalmax25 | 80 | 84 | 100 | 100 | **91.0** |
| 50/100 | Uniform future | 72 | 78 | 98 | 100 | 87.0 |
| 75/150 | Goalmax25 | 84 | 56 | 98 | 100 | 84.5 |
| 75/150 | Uniform future | 78 | 64 | 100 | 100 | **85.5** |
| 100/200 | Goalmax25 | 84 | 40 | 98 | 100 | 80.5 |
| 100/200 | Uniform future | 82 | 80 | 100 | 100 | **90.5** |

Goalmax25 在 50/100 仍有收益，但从 75 开始 PushT 明显退化，到 100-step condition
goal 时已经发生严重 OOD。Uniform future 在 100/200 保持 90.5，证明“换一个覆盖
长距离的 generator”确实能解决大部分退化，而不是 subgoal 机制本身失效。

不同 goal offsets 使用不同合法-window manifest，所以不应把 75/150 与 100/200
的非单调变化解释为距离越远越容易；严格 paired comparison 只在同一距离、同一
evaluation seed 内成立。

### 7.8 Policy 使用 final goal 还是 subgoal（100/200）

| Generator | Policy goal | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---|---:|---:|---:|---:|---:|
| Goalmax25 | Final $z_g$ | 84 | 40 | 98 | 100 | **80.5** |
| Goalmax25 | Predicted K10 | 82 | 38 | 98 | 100 | 79.5 |
| Uniform future | Final $z_g$ | 82 | 80 | 100 | 100 | **90.5** |
| Uniform future | Predicted K10 | 76 | 82 | 98 | 100 | 89.0 |

当 policy 与 CEM 都依赖同一个预测 subgoal 时，两者的误差高度相关；policy 看 final
goal 能提供独立的全局方向，CEM 再用 local target 修正可达性，因此平均更好。

## 8. 原始 LeWM 长距离参考

原始 LeWM 参数（last、H5/RH5、CEM 300×30）随距离明显退化：

| Goal/Budget | Cube | PushT | Reacher | TwoRoom | 平均 |
|---|---:|---:|---:|---:|---:|
| 50/100 | 54 | 50 | 98 | 50 | 63.0 |
| 75/150 | 62 | 10 | 98 | 50 | 55.0 |

这不是最强 global-goal baseline，因为 cost/receding-horizon 仍是原始协议；它的作用
是说明原始 LeWM-CEM 在远距离尤其容易在 PushT 和 TwoRoom 上失去指导信号。

## 9. 失败实验与已排除解释

### 9.1 已确认的实现错误

- 用 H5 的最后一个 LeWM checkpoint 与 K10 比较：预测时间和规划时间错位。
- `subgoal refresh=10` 与 RH1/5-step execution 绑定：在 $t+5$ 重规划时仍追旧的
  $z_{t+10}$，实际上让新 rollout 的 $t+15$ 追旧目标。
- 只改 target 为 K10、但仍“规划到第 25 步”：不能代表局部 subgoal 方法。

这些配置仅保留为诊断，不进入论文主表。

### 9.2 今天没有被支持的假设

- “aligned goal sampling 一定更好”：不成立，平均收益不稳定。
- “更多 flow samples 一定更好”：不成立，N=8 在 75/150 略低于 N=1。
- “K15 比 K10 更能处理远距离”：不成立。
- “K5/K10 两项直接求 mean 会更好”：不成立。
- “增加 policy proposals 或使用 LeWM proposal selection 会有大幅增益”：不成立。
- “所有任务的瓶颈都能由离线 MSE 排序解释”：不成立，TwoRoom 是明显反例。
- “goalmax25 可以作为通用长距离 generator”：不成立。

### 9.3 仍无法完全证明的因果关系

Reacher 在早期 K10-only 短距离实验中的明显下降，与较差的 K10 validation
MSE/cosine 一致，而且换回 final goal 能恢复成功率；这强烈支持 predictor quality
是主要瓶颈。但闭环 OOD、latent geometry、LeWM rollout error 和 actor 未针对 local
goal 训练仍可能共同贡献，不能只靠当前单 seed 结果宣称唯一因果。

## 10. 论文主线建议

### 10.1 推荐问题表述

当最终目标在 frozen latent space 中很远时，直接最小化

\[
\|\hat z_{t+h}-z_g\|_2^2
\]

未必提供局部可执行的控制方向。本文学习一个只依赖离线轨迹监督的 conditional
latent path generator，把远目标转换为动态更新的局部 waypoint，并用 actor 提供
全局方向、LeWM-CEM 保证局部可达性。

### 10.2 当前最能支持的贡献点

1. 在 frozen LeWM 空间中提出纯监督 LatentPathFlow，不依赖 reward、Q、V 或
   inverse dynamics。
2. 预测与 action chunk 对齐的 K5/K10 latent path，并在闭环中每个 action block
   重新生成。
3. 提出 global-policy/local-planner decoupling：actor condition 在 final goal，
   model-based cost 在 predicted local subgoal。
4. 实验证明 local latent targets 的收益随目标距离增加而显现，并揭示 training goal
   distribution 对跨距离泛化的关键影响。
5. Staged local-to-final planning 在长距离上进一步消除 predictor terminal bias。

### 10.3 建议论文主表

主表不要混合 CEM 300×30 与 300×5。建议先统一重跑后选择以下四类方法：

1. Original LeWM-CEM：final goal；
2. Actor-guided final-goal CEM；
3. Actor-guided K10 CEM；
4. Decoupled policy-final/CEM-K10；
5. 可选：staged K10→final。

距离统一报告 `25/50、50/100、75/150、100/200`，四任务分别报告成功率和宏平均。

### 10.4 必补实验

1. 固定最终协议后补 evaluation seeds，例如 42、131、132、777、789；报告均值和
   置信区间。
2. Predictor 至少补 3 个 training seeds，区分 flow sampling 方差和训练方差。
3. 在相同 evaluator/相同 CEM iteration 下重跑 staged 与 decoupled 方法。
4. 训练覆盖 `{5,10,...,100}` 的显式 distance-balanced predictor，与 uniform future
   比较，验证 SAGE-style distance balance 在长距离是否真正有效。
5. 做 GT K5/K10 oracle，与 predicted path 严格配对，量化 predictor error 和
   planner/model error 的上限差距。
6. 分 goal-distance 报告 predictor validation MSE/cosine，并测 closed-loop state
   到 expert-manifold 的偏移。
7. 对 path loss 尝试置信度/时间权重，而不是等权 mean。
8. 对 100/200 补 final-goal actor-CEM baseline，形成完整 paired table。

## 11. 固定 checkpoint 与结果位置

### 11.1 LeWM checkpoints

```bash
# PushT
/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack

# Cube
/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack

# Reacher
/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack

# TwoRoom
/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
```

### 11.2 Predictor roots

```text
Uniform K10:
/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10/

Aligned K10:
/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-aligned-future/

Goalmax25 K10:
/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25/

Uniform K15:
/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k15/
```

四任务正式 checkpoint 均为各自 run 目录下的 `checkpoint_200000.msgpack`。

### 11.3 Policy checkpoints

```text
/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror/
  gc4_cube_all_n100000_b256_a0.0_sd777/params_100000.pkl
  gc4_pusht_all_n100000_b256_a0.0_sd777/params_100000.pkl
  gc4_reacher_all_n100000_b256_a0.0_sd777/params_100000.pkl
  gc4_tworoom_all_n100000_b256_a0.0_sd777/params_100000.pkl
```

### 11.4 Evaluation roots

```text
/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/
```

每个正式 run 的 `result.json` 都记录 controller、goal/budget、CEM 配置、policy goal
mode、是否使用 Q/V、predictor checkpoint、history size、num_samples 和逐 episode
success，后续制表应直接读取 JSON，不从终端文本手抄。

## 12. 代码与 Bash 索引

核心实现：

- `impls/latent_subgoal.py`：LatentPathFlow、AdaLN Transformer、flow sampler、medoid；
- `impls/train_latent_subgoal_gcbc.py`：latent cache、episode split、goal sampling、训练和验证；
- `impls/latent_subgoal_runtime.py`：checkpoint 校验、history buffer、在线生成；
- `impls/lewm_jax/planner.py`：LeWM-CEM、MoH/path_mean、actor guidance、staged planner；
- `impls/eval_lewm_4tasks.py`：四任务统一 evaluator 与 JSON 记录；
- `ogbench/lewm_envs/evaluation.py`：dataset start/final-goal 采样和环境 rollout。

训练 Bash：

- `exp/train/latent_subgoal/20260831_run_node4_4tasks_latent_path_flow_k10_s0.sh`；
- `exp/train/latent_subgoal/20260901_run_node4_4tasks_latent_path_flow_k10_aligned_future_s0.sh`；
- `exp/train/latent_subgoal/20260901_run_node4_4tasks_latent_path_flow_k10_goalmax25_s0.sh`；
- `exp/train/latent_subgoal/20260901_run_node4_4tasks_latent_path_flow_k15_s0.sh`。

关键评测 Bash：

- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_uniform_future_ns1_lewm_cem_distance_matrix.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_aligned_future_ns1_lewm_cem_distance_matrix.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_goalmax25_ns1_cem_vs_policy_g25_b50.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_final_goal_actor_mode_cem_h5_j5_long_distance.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_gciql_chunk_all_latent_path_flow_hist3_k10_ns8_staged_guided_long_distance.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_goalmax25_policy_final_goal_long_distance.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_uniform_future_policy_final_goal_long_distance.sh`；
- `exp/eval/lewm_4tasks/20260901_eval_node4_k10_policy_final_goal_g100_b200_generators.sh`。

今天与本方法直接相关的主要提交范围为 `a2516fb` 至 `43fcbf9`。其中最终新增的
关键能力包括：validation sample count 与 inference 对齐、history3、action-block
aligned sampling、distance-balanced goalmax、path_mean、actor-only CEM variants、
staged planning，以及 policy-final/CEM-subgoal decoupling。

## 13. 其它并行工作

今天还准备了 node3 上 OGBench 8-task LeWM epoch10、BatchNorm latent geometry
诊断，以及 shared-all GCIQL-Chunk 三 seed/500k 训练 Bash。这些工作与本文四任务
LatentPathFlow 主表尚未形成统一结果，不应混入上述结论；对应入口为：

- `exp/train/20260901_train_node3_lewm_ogbench_env_8tasks_e10.sh`；
- `exp/eval/ogbench_env_8tasks/20260901_diagnose_node3_lewm_bn_geometry_epoch10.sh`；
- `exp/train/20260901_train_node3_gciql_chunk_all_ogbench8_3seeds_500k.sh`。

## 14. 当前推荐配置

如果现在继续做论文实验，建议固定为：

```text
Predictor: LatentPathFlow K10, history3, frozen z192
Sampling: long-range uniform future（下一步对比 distance-balanced 5..100）
Flow: Euler-16, EMA, inference num_samples=1 或预注册 N=8
Policy: shared-all GCIQL-Chunk seed777 actor only
Policy goal: final z_g
CEM target: predicted z_{t+10}
Cost: MoH
Action block: 5
Horizon / RH: H2 / RH1
CEM: 300 samples, top-30；统一固定 iterations 后再做主表
Q/V: disabled
Evaluation: 25/50, 50/100, 75/150, 100/200，多 evaluation seeds
```

现阶段最稳妥的论文叙述不是“subgoal 在所有设置都优于 final goal”，而是：

> Learned local latent paths address the loss of local control signal under distant
> goals. Their benefit emerges as goal distance increases, while short-horizon
> performance is best preserved by retaining global goal conditioning in the policy
> and using predicted subgoals only as locally reachable model-based planning targets.
