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

训练完成后的 CEM 使用方式另行做严格配对评测：LeWM 第一个 rollout checkpoint（t+5）匹配预测 K5，第二个 checkpoint（t+10）匹配预测 K10，并将两个 latent MSE 相加作为候选动作序列 cost。
