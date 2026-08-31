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

## 200k CEM 闭环结果

正式评测使用 50 episodes、evaluation seed42、goal offset25、budget50、CEM300x30、H5/RH1、top-k30、min-over-horizon，并与 MLP/global/oracle 共享起点和 CEM plan keys。正式 Bash：`exp/eval/lewm_4tasks/20260831_eval_yb_lewm_flow_transformer_subgoal_moh.sh`。

结果目录：

`/root/data/yyf/lewm-final/evals/lewm-4tasks/20260831_flow_transformer_subgoal_k10_moh_cem300x30_h5_rh1_ep50_seed42/`

| 方法 | Cube | PushT | Reacher | TwoRoom | 四任务平均 | 成功数 |
|---|---:|---:|---:|---:|---:|---:|
| Global-goal CEM | 74 | 88 | 100 | 98 | 90.0 | 180/200 |
| MLP predicted subgoal | 64 | 90 | 82 | 84 | 80.0 | 160/200 |
| Transformer-CFM predicted subgoal | 70 | 84 | 52 | 92 | 74.5 | 149/200 |
| GT subgoal oracle | 70 | 96 | 100 | 100 | 91.5 | 183/200 |

Transformer-CFM 相对 MLP 的 paired episode flips：Cube `4/1`、PushT `2/5`、Reacher `1/16`、TwoRoom `5/1`（前者为 Flow-only success，后者为 MLP-only success），合计新增 12、丢失 23，净少 11 个成功 episode。

结论：当前单样本 conditional-flow waypoint 没有解决 subgoal 精度瓶颈。它改善 Cube 和 TwoRoom，但 PushT 下降，Reacher 大幅下降 30 个百分点；总体也明显低于直接规划最终目标。结合离线单样本 MSE 同样变差，说明在当前数据与使用方式下，随机条件分布样本比 MSE 条件均值更容易生成对 CEM 不够精确或不可达的 waypoint。后续不应直接用单个 Flow sample 完全替换 global goal；更合理的方向是候选 subgoal 多采样后按可达性筛选，并保留 global-goal cost。
