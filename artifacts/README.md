---
pretty_name: LeWorldModel++ Evaluation Artifacts
license: other
---

# LeWorldModel++ evaluation artifacts

This dataset contains the data and exact checkpoints selected for the public
LeWorldModel++ evaluation launchers. The matching source revision is
[`314cd60`](https://github.com/pickxiguapi/ogbench/commit/314cd606c3267ccf83f4d35e2d7e646b17098211).

## Layout

```text
lewm-control-suite/
├── data/
└── checkpoints/
    ├── lewm/{cube,pusht,reacher,tworoom}/
    ├── action-prior/{cube,pusht,reacher,tworoom}/
    └── latent-path-flow/{h25,longh}/{cube,pusht,reacher,tworoom}/

visual-ogbench/
├── data/
└── checkpoints/{lewm,action-prior,latent-path-flow}/<dataset-tag>/
```

Each checkpoint directory includes the final checkpoint consumed by the public
bash launchers and its available configuration metadata. `manifest.json`
records the byte size and SHA-256 digest of every file.

## Download

Download only the checkpoints:

```bash
uvx --from huggingface_hub hf download IffYuan/leworldmodel-pp-artifacts \
  --repo-type dataset --include "*/checkpoints/**" --local-dir artifacts
```

Omit `--include` to download the complete prepared-data bundle.

## Data provenance

The LeWM Control Suite data comes from the official
[LeWM collection](https://huggingface.co/collections/quentinll/lewm). Visual
OGBench data comes from the official
[OGBench release](https://github.com/seohongpark/ogbench). The checkpoint and
result metadata retain their original task, training, and protocol fields.
