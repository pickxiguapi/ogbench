# Frozen LeWM Latent Subgoal Transformer-CFM（K=10）

## 目标

旧版 MLP 用单点 MSE 学习条件均值，在多条合理的 K10 路径并存时可能产生不够可达的平均 latent。新版保持 frozen LeWM、latent cache、K10 target 和 HIQL-style future goal sampling 不变，只把 generator 改为可表达条件分布的 Transformer conditional flow matching（CFM）。

## 数据定义

对同一 episode 中的当前位置 `t`，均匀采样未来 goal index `g in [t+1,T]`：

\[
z_c=z_t,\qquad z_g=z_g,\qquad z^*=z_{\min(t+10,g)}.
\]

PushT 使用 seed666 LeWM latent cache；Cube、Reacher、TwoRoom 使用 seed3072 cache。LeWM 和全部 latent cache 在 generator 训练期间冻结。

## Conditional flow matching

采样：

\[
\epsilon\sim\mathcal N(0,I),\qquad \tau\sim U[0,1],
\]

构造直线路径和监督速度：

\[
z_\tau=(1-\tau)\epsilon+\tau z^*,\qquad
v^*=z^*-\epsilon.
\]

网络学习：

\[
v_\theta(z_\tau,\tau\mid z_c,z_g),\qquad
\mathcal L_{\mathrm{CFM}}=
\mathbb E\left[\lVert v_\theta-v^*\rVert_2^2\right].
\]

推理从 `N(0,I)` 采样初值，并积分 ODE `dz/dtau=v_theta`。正式配置使用 16-step Heun solver 和 EMA 参数。规划器中的噪声 key 由 evaluation seed、environment index、subgoal generation count 确定，因此 paired evaluation 可复现。

## Transformer Encoder

每个样本有四个 token：当前 latent、最终目标 latent、当前 flow latent、sinusoidal flow-time。四者投影到同一宽度并加 token-type embedding，再经过 pre-norm Transformer Encoder。仅 noisy/flow token 经 LayerNorm 和线性头输出 192 维速度。

| 参数 | 设置 |
|---|---:|
| LeWM latent dim | 192 |
| Transformer width | 384 |
| Encoder layers | 8 |
| Attention heads | 8 |
| FFN width | 1536 |
| Token count | 4 |
| 参数量 | 约 14.64M |

## 正式训练配置

| 参数 | 设置 |
|---|---:|
| K | 10 atomic steps |
| Training steps | 200,000 |
| Batch size | 1024 |
| Optimizer | AdamW |
| Peak / final LR | 1e-4 / 1e-5 |
| Warmup | 5,000 steps |
| Weight decay | 1e-4 |
| Gradient clip | 1.0 |
| EMA decay | 0.9999 |
| Episode split | 95% / 5% |
| Validation pairs | 10,000 fixed pairs |
| Checkpoint interval | 25,000 steps |

正式入口仍为 `impls/train_latent_subgoal_gcbc.py`，通过 `--architecture=transformer_flow` 选择新版。四任务只能从 `exp/train/latent_subgoal/20260831_run_*_flow_transformer_k10_s0.sh` 启动。

## 评估原则

训练 loss 是速度场 MSE；最终关心的 generator 指标仍是完整 ODE sample 相对 `z^*` 的 latent MSE、L2 和 cosine。因为模型表达的是条件分布，单样本 MSE 不必严格低于条件均值 MLP；最终判断标准是相同 CEM protocol 下的闭环成功率，并同时保留 GT-subgoal oracle 与 global-goal CEM 作为上下界参照。
