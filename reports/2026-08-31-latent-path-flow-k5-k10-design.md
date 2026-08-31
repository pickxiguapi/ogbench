# Frozen-LeWM LatentPathFlow（K5/K10）设计

## 目标

在每个任务已经冻结的 LeWM 表征空间中，用当前状态 latent `z_t` 和同轨迹未来目标 latent `z_g` 生成一条两点局部路径：

\[
Z^*=\left[z_{\min(t+5,g)},\;z_{\min(t+10,g)}\right]\in\mathbb R^{2\times192}.
\]

PushT 使用 seed666 LeWM，Cube、Reacher、TwoRoom 使用 seed3072 LeWM。四任务分别训练，不跨任务共享 latent 坐标系或 generator。

## 数据与 goal sampling

- 直接读取预计算的 frozen LeWM `float32` z192 HDF5，不加载图像和 LeWM。
- episode 级 95/5 train/validation split，split seed 0。
- 当前行 `t` 从训练 episode 的所有非终止行均匀采样。
- `g` 按 HIQL/HGCBC 风格从同一条轨迹的 `[t+1, episode_end]` 均匀采样。
- 两个 target 均在 `g` 截断；当目标不足 5 或 10 步时，相应 waypoint 就等于 `z_g`。

## 仅 Flow Matching 的训练目标

不训练 inverse dynamics，也不加 LeWM consistency loss：

\[
\mathcal L=\mathcal L_{\mathrm{flow}}.
\]

对整个两点路径采样同形状高斯噪声和一个共享 flow time：

\[
\epsilon\sim\mathcal N(0,I),\quad \tau\sim U(0,1),\quad
X_\tau=(1-\tau)\epsilon+\tau Z^*.
\]

网络输入 `(X_tau, tau, z_t, z_g)`，输出两个 waypoint 的速度，监督速度为 `Z* - epsilon`：

\[
\mathcal L_{\mathrm{flow}}=
\mathbb E\left[\left\|v_\theta(X_\tau,\tau\mid z_t,z_g)
-(Z^*-\epsilon)\right\|_2^2\right].
\]

## 网络（LeFlow 风格）

- noisy path 中的两个 waypoint 分别作为 Transformer token。
- latent token projection：192 -> 512。
- `z_t` 和 `z_g` 分别线性投影到 512，二者与 time condition 相加后广播到两个 path token。
- sinusoidal time embedding 64，MLP `64 -> 512 -> 512`，SiLU。
- 两个可学习的位置 embedding，区分 K5 与 K10。
- 4 个 pre-norm Transformer encoder blocks，8 heads，FFN 2048，GELU，无 dropout。
- output LayerNorm + linear `512 -> 192`，输出 `[B, 2, 192]`。
- 参数量约 10–20M，单元测试强制检查预算。

## 训练与采样参数

| 参数 | 值 |
|---|---:|
| train steps | 200,000 |
| batch size | 1,024 |
| optimizer | AdamW |
| peak / final LR | 1e-4 / 1e-5 |
| warmup | 5,000 |
| weight decay | 1e-4 |
| global grad clip | 1.0 |
| EMA | 0.9999 |
| checkpoint interval | 25,000 |
| validation interval | 10,000 |
| inference solver | Euler |
| inference steps | 16 |
| training seed | 0 |

验证同时记录整条路径以及 K5、K10 各自的 MSE 与 cosine similarity。模型选择与正式推理使用 EMA 参数。

## 与旧单点 Transformer-CFM 的隔离

新 architecture 标记为 `latent_path_flow_transformer_encoder`，loss 标记为 `conditional_path_flow_matching_mse`，输出目录为 `latent-path-flow-k10`。旧的 `latent_flow_transformer_encoder` 单点 checkpoint、加载逻辑和实验目录保持不变。

## 正式启动 Bash

- `exp/train/latent_subgoal/20260831_run_yb_pusht_latent_path_flow_k10_s0.sh`
- `exp/train/latent_subgoal/20260831_run_node2_cube_latent_path_flow_k10_s0.sh`
- `exp/train/latent_subgoal/20260831_run_node3_reacher_latent_path_flow_k10_s0.sh`
- `exp/train/latent_subgoal/20260831_run_node4_tworoom_latent_path_flow_k10_s0.sh`
- `exp/train/latent_subgoal/20260831_run_node4_4tasks_latent_path_flow_k10_s0.sh`（node4 GPU0–3 四任务并行；`TASKS` 可选择子集）

训练完成后的 CEM 使用方式另行做严格配对评测：LeWM 第一个 rollout checkpoint（t+5）匹配预测 K5，第二个 checkpoint（t+10）匹配预测 K10，并将两个 latent MSE 相加作为候选动作序列 cost。

## 200k 训练结果

四任务均在 node4 完成 200k steps，单任务约 20.4 分钟。最终单样本验证如下：

| Task | Joint MSE / cosine | K5 MSE / cosine | K10 MSE / cosine |
|---|---:|---:|---:|
| PushT | 0.0878 / 0.9538 | 0.0676 / 0.9645 | 0.1081 / 0.9429 |
| Cube | 0.2406 / 0.8760 | 0.1732 / 0.9091 | 0.3080 / 0.8428 |
| Reacher | 0.7509 / 0.6213 | 0.5885 / 0.7032 | 0.9133 / 0.5394 |
| TwoRoom | 1.0780 / 0.4356 | 1.0824 / 0.4289 | 1.0735 / 0.4396 |

## 只用 K10 替换 K25 global goal：H5 terminal CEM

正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_terminal.sh`。

协议为 50 episodes、evaluation seed42、dataset goal offset25、budget50、CEM300x30、H5/RH1、top-k30。Generator 仍以 `z_t,z_g(K25)` 为条件生成 `[K5,K10]`，planner 只选择 index1 的 K10 token，并每 10 atomic steps 刷新。CEM cost 仅比较第 5 个 LeWM rollout checkpoint（约 t+25）与 predicted K10；不使用 K5，也不使用 MoH。

结果目录：

`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_latent_path_flow_k10_only_terminal_cem300x30_h5_rh1_ep50_seed42/`

| Method | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| Single-point Transformer-CFM K10, H5 terminal | 58 | 8 | 24 | 50 | 35.0 |
| LatentPathFlow K10 token, H5 terminal | 62 | 14 | 24 | 58 | 39.5 |

结果 JSON 均记录 `selected_waypoint_index=1`、`selected_waypoint_step=10` 和 `cost_mode=terminal`。LatentPathFlow 相对旧单点 Flow 提升 4.5 个宏平均百分点，但总体仍很差。原因不是误用了 K5 或 MoH，而是 H5 terminal 要求约 t+25 的 LeWM rollout 回到一个应在 t+10 到达的局部 waypoint，时间位置不匹配。下一步若要检验 K10 本身的可利用性，应固定比较第 2 个 rollout checkpoint（t+10）；若要检验两点路径，则使用 K5/K10 两项对齐 cost。

### K10-only 局部规划：H2 terminal 与刷新频率修正

上一节 H5 terminal 是时间错位诊断，不是“K10 替代 K25 后只规划到 K10”的正确协议。正确目标为：

\[
J(a_{t:t+9})=
\left\|\hat z^{\mathrm{LeWM}}_{t+10}(a_{t:t+9})
-z^{\mathrm{pred}}_{t+10}\right\|_2^2.
\]

由于 action block 为 5，CEM horizon 固定为 H2；terminal 因而严格对应当前状态之后的 t+10。Generator 条件仍使用 dataset K25 global goal。其余保持 50 episodes、seed42、budget50、CEM300x30、RH1、top-k30；不使用 K5 或 MoH。

第一次 H2 运行仍错误地把 `prediction_horizon=10` 与 `refresh_interval=10` 绑定。RH1 每次只执行 5 步，因此在 t+5 replan 时仍持有旧的 z(t+10)，却用新的 H2 rollout（终点 t+15）追踪它。修正后两者解耦：generator 始终预测“当前状态之后 K10”，但每次 RH1 replan（5步）都重新生成。

正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_h2_terminal.sh`。

结果目录：

修正后结果目录：

`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_latent_path_flow_k10_only_refresh5_terminal_cem300x30_h2_rh1_ep50_seed42/`

| K10-only protocol | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| 错误时间对齐：H5 terminal=t+25 | 62 | 14 | 24 | 58 | 39.5 |
| H2 terminal，但错误 refresh=10 | 76 | 70 | 28 | 96 | 67.5 |
| H2 terminal，正确每次 replan refresh=5 | 80 | 84 | 48 | 100 | 78.0 |

从 H5 错位协议到完全修正协议，宏平均提高 38.5 个百分点；仅修正刷新频率就提高 10.5 个百分点。结果 JSON 明确记录 `training_subgoal_steps=10`、`selected_waypoint_step=10`、`refresh_steps=5`、`horizon=2`、`cost_mode=terminal`。Reacher 仍只有 48%，与它最差的 K10 离线预测指标（MSE 0.9133、cosine 0.5394）一致，当前剩余的主要瓶颈是该任务的 subgoal prediction quality。

### Refresh5 H2 MoH

在完全修正的 K10-only H2 协议上，只将 cost 从 terminal 改成：

\[
J_{\mathrm{MoH}}=
\min_{h\in\{5,10\}}
\left\|\hat z^{\mathrm{LeWM}}_{t+h}-z^{\mathrm{pred}}_{t+10}\right\|_2^2.
\]

正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_h2_moh.sh`。其余保持 K25 condition goal、refresh5、H2/RH1、CEM300x30、50 episodes、seed42，不使用预测的 K5 token。

结果目录：

`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_latent_path_flow_k10_only_refresh5_moh_cem300x30_h2_rh1_ep50_seed42/`

| Corrected K10-only cost | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| H2 terminal | 80 | 84 | 48 | 100 | 78.0 |
| H2 MoH over t+5/t+10 | 76 | 90 | 62 | 100 | 82.0 |
| Direct global-goal H5 MoH | 74 | 88 | 100 | 98 | 90.0 |

MoH 相对 terminal 的 paired flips（MoH-only / terminal-only）为 Cube `1/3`、PushT `6/3`、Reacher `12/5`、TwoRoom `0/0`，净增加 8 个成功 episode，宏平均提高 4 分。MoH 后 Cube、PushT、TwoRoom 均不低于 direct global-goal baseline；剩余 8 分宏平均差距全部由 Reacher 的 62 vs 100 造成。

### 与无 subgoal 的严格 H2 MoH 对照

为了排除上一表 H5 与 H2 规划视野不同的混杂因素，额外运行无 subgoal 的 global-goal 对照。除 planner 直接追踪 dataset K25 global goal 外，其余参数与 corrected K10 subgoal 完全相同：50 episodes、seed42、budget50、CEM300x30、H2/RH1、action block5、top-k30、MoH。

| H2 MoH target | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| K25 global goal（无 subgoal） | 76 | 76 | 96 | 86 | 83.5 |
| Predicted K10 subgoal | 76 | 90 | 62 | 100 | 82.0 |
| Predicted K10 - global | 0 | +14 | -34 | +14 | -1.5 |

同一批 episode 的 paired flips（subgoal-only / global-only）为 Cube `3/3`、PushT `10/3`、Reacher `1/18`、TwoRoom `7/0`。因此不能笼统地说 predicted subgoal 在四个任务上都不准确或都伤害规划：它在 PushT 与 TwoRoom 明显有益，Cube 持平。当前主要失败点集中在 Reacher；该任务的 K10 离线指标也是四个任务中第二差（MSE `0.9133`、cosine `0.5394`），且严格同视野下换回 global goal 即由 62% 恢复到 96%，支持“Reacher 的主要瓶颈是 subgoal prediction quality”。但这仍是强证据而非单独充分的因果证明；最终应以同协议的 GT K10 oracle 上限来区分 predictor error 与局部目标本身的可规划性。

### 远距离目标：H50/H75

在保持 planner 参数完全一致的前提下，将 dataset goal offset / environment budget 从 `25/50` 扩展为 `50/100` 和 `75/150`。每个距离都严格配对 predicted K10 subgoal 与 direct global goal：50 episodes、seed42、CEM300x30、H2/RH1、action block5、MoH；predicted K10 每5个 environment steps 重新生成。这里表中的 H 表示 dataset goal offset，而非 CEM horizon（CEM horizon 始终为2）。

| Goal offset / budget | Target | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|
| 25 / 50 | Global goal | 76 | 76 | 96 | 86 | 83.5 |
| 25 / 50 | Predicted K10 | 76 | 90 | 62 | 100 | 82.0 |
| 50 / 100 | Global goal | 62 | 36 | 88 | 58 | 61.0 |
| 50 / 100 | Predicted K10 | 62 | 72 | 82 | 100 | 79.0 |
| 75 / 150 | Global goal | 76 | 4 | 88 | 64 | 58.0 |
| 75 / 150 | Predicted K10 | 62 | 72 | 94 | 100 | 82.0 |

Predicted K10 相对 global goal 的宏平均差值随距离从 `-1.5` 变为 `+18.0`、`+24.0`。尤其 PushT 的差值为 `+14/+36/+68`，TwoRoom 为 `+14/+42/+36`；这两个任务直接体现了局部 target 在远目标下避免 global latent cost 失去局部指导性的价值。Cube 在 H50 持平、H75 落后14分，是当前没有获得远距离收益的例外。Reacher 从 H25 的 `-34` 变为 H50 的 `-6` 和 H75 的 `+6`，说明 H25 上的失败不能外推为“Reacher 的 subgoal generator 对所有 goal distance 都不可用”。

同一距离内的 paired flips（subgoal-only / global-only）如下：

| Goal offset | Cube | PushT | Reacher | TwoRoom |
|---:|---:|---:|---:|---:|
| 25 | 3 / 3 | 10 / 3 | 1 / 18 | 7 / 0 |
| 50 | 7 / 7 | 22 / 4 | 4 / 7 | 21 / 0 |
| 75 | 4 / 11 | 35 / 1 | 5 / 2 | 18 / 0 |

不同 goal offset 会由 evaluator 根据各自的合法窗口重新采样 manifest，因此跨距离的同一列不是 episode-level paired comparison；每个距离内部的两种 target 使用完全一致的 manifest，可以严格配对。总体结论是：当前 LatentPathFlow K10 在宏平均上随 goal distance 增大保持稳定，而直接 global-goal H2 MoH 明显退化，支持“subgoal decomposition 主要在远目标下发挥作用”。
