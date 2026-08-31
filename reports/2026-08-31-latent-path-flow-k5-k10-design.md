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

### 正确的 K10-only 局部规划：H2 terminal

上一节 H5 terminal 是时间错位诊断，不是“K10 替代 K25 后只规划到 K10”的正确协议。正确目标为：

\[
J(a_{t:t+9})=
\left\|\hat z^{\mathrm{LeWM}}_{t+10}(a_{t:t+9})
-z^{\mathrm{pred}}_{t+10}\right\|_2^2.
\]

由于 action block 为 5，CEM horizon 固定为 H2；terminal 因而严格对应 t+10。Generator 条件仍使用 dataset K25 global goal，K10 token 每 10 atomic steps 刷新。其余保持 50 episodes、seed42、budget50、CEM300x30、RH1、top-k30；不使用 K5 或 MoH。

正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_h2_terminal.sh`。

结果目录：

`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260831_latent_path_flow_k10_only_terminal_cem300x30_h2_rh1_ep50_seed42/`

| K10-only protocol | Cube | PushT | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| 错误时间对齐：H5 terminal=t+25 | 62 | 14 | 24 | 58 | 39.5 |
| 正确时间对齐：H2 terminal=t+10 | 76 | 70 | 28 | 96 | 67.5 |

严格对齐后宏平均提高 28 个百分点，验证了 K10 subgoal 确实能够被 CEM 利用；但 Reacher 仍只有 28%，与它最差的 K10 离线预测指标（MSE 0.9133、cosine 0.5394）一致，当前主要瓶颈仍是该任务的 subgoal prediction quality。
