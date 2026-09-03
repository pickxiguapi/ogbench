# OGBench 8 Tasks LatentPathFlow 主模型训练设计

## 目标

为 `visual-cube-{single,double,triple}-{play,noisy}` 与
`visual-scene-{play,noisy}` 八个数据集分别训练主 subgoal generator。这里只
训练 LeWM++ 主模型，不重复 MLP、EndpointFlow 或 policy-guidance 消融。

## Frozen LeWM 与数据绑定

每个数据集使用 A800 node3 上已经完成正式 OGBench-8 评测的独立
`seed3072 / epoch10 / bs128` LeWM checkpoint：

`/data-training/yyf/ogbench-lewm-policy-runs/lewm-ogbench8/lewm_ogbench8_<tag>_e10_bs128_s3072/weights_epoch_10.msgpack`

训练前从官方 train NPZ 的 `observations` 逐行执行
`LeWM.encode_pixels(..., train=False)`，生成独立的 float32 z192 HDF5。cache
保存原始 trajectory boundary，并同时记录 source NPZ 与 LeWM checkpoint
的 SHA-256；不同数据集的 latent 坐标系不得混用。

## 主模型参数

- 模型：LatentPathFlow，history size 3，K5/K10 两个 path token；
- 条件：最近 3 帧 frozen latent 与同轨迹 future final-goal latent；
- goal sampling：full-range same-trajectory uniform future；
- loss：conditional flow matching MSE；
- backbone：hidden 512、depth 4、8 heads、FFN 2048、time dim 64；
- optimizer：AdamW，200k updates，batch size 1024；
- LR：warmup 5k 到 `1e-4`，cosine decay 到 `1e-5`；
- weight decay `1e-4`，gradient clip `1.0`，EMA `0.9999`；
- validation：固定 held-out H50 manifest，每数据集 10k pairs；
- solver：16-step Euler；主实验 `num_samples=1`；training seed 0。

八个环境分别训练独立 generator。正式入口是
`exp/train/latent_subgoal/20260904_train_node4_ogbench8_latent_path_flow_main_s0.sh`。
