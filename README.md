# LeWorldModel++

**LeWorldModel++ (LeWM++)** is a planning-interface framework for long-horizon image-goal control with frozen latent world models (e.g. LeWorldModel, DINO-WM). It addresses three coupled failures that arise when short model rollouts must serve distant goals: mismatched planning targets, uninformed action search, and restrictive terminal-state costs.

LeWM++ combines **LatentPathFlow** to generate reachable local targets, an **Action Chunk Prior** to initialize CEM from goal-directed offline behavior, and **Min-over-Horizon (MoH)** to score the closest predicted approach to the target. The controller executes one optimized action chunk and replans, preserving global intent through the final image goal while operating within a locally reliable prediction horizon.

## Installation

```bash
git clone https://github.com/pickxiguapi/leworldmodel_pp.git
cd leworldmodel_pp
uv sync --extra train --extra dev
```

## Training

Run all commands in this README from the repository root. Activate the
environment first:

```bash
source .venv/bin/activate
```

All launchers write checkpoints, latent caches, logs, and evaluation results
to `outputs/` by default. Change `EXPERIMENT_ROOT` at the top of a launcher if
you want a different location. Dataset and external-checkpoint paths remain
empty and must be filled in explicitly.

### LeWM Control Suite

1. Download the four official HDF5 datasets from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm):

- `quentinll/lewm-cube`: `cube_single_expert.tar.zst`;
- `quentinll/lewm-pusht`: `pusht_expert_train.h5.zst`;
- `quentinll/lewm-reacher`: `reacher.tar.zst`;
- `quentinll/lewm-tworooms`: `tworoom.tar.zst`.

2. Extract the archives and place the four HDF5 files in one directory using the
filenames below. Then convert each HDF5 file to its JPEG-backed Lance table:

```bash
LEWM_DATA_ROOT=/absolute/path/to/lewm-control-suite

for name in cube_single_expert pusht_expert_train reacher tworoom; do
  python scripts/convert_lewm_hdf5_to_lance.py \
    "$LEWM_DATA_ROOT/$name.h5" \
    "$LEWM_DATA_ROOT/$name.lance"
done
```

The prepared dataset directory should have this layout:

```text
lewm-control-suite/
├── cube_single_expert.h5
├── cube_single_expert.lance/
├── pusht_expert_train.h5
├── pusht_expert_train.lance/
├── reacher.h5
├── reacher.lance/
├── tworoom.h5
└── tworoom.lance/
```

After setting the paths at the top of each launcher, train all components in
dependency order:

```bash
# 3. Train the LeWM dynamics models.
bash experiments/train/train_lewm_4tasks.sh

# 4. Encode the offline datasets with the trained LeWM checkpoints.
bash experiments/precompute_lewm_latents_4tasks.sh

# 5. Train the final-goal-conditioned Action Chunk Priors.
bash experiments/train/train_action-prior-chunk_4tasks.sh

# 6. Train the H25 LatentPathFlow models.
bash experiments/train/train_subgoal_latent_path_flow_h25_4tasks.sh

# 7. Train the long-horizon LatentPathFlow models used for H50/H75/H100.
bash experiments/train/train_subgoal_latent_path_flow_longh_4tasks.sh
```

### Visual OGBench

Expected filenames under `OGBENCH_DATA_ROOT` are the eight environment names ending in `.npz`, plus matching `-val.npz` files, such as `visual-cube-single-play-v0.npz` and `visual-cube-single-play-v0-val.npz`.

```bash
# 1. Train one frozen LeWM model for each Visual OGBench dataset.
bash experiments/train/train_lewm_visual_ogbench8.sh

# 2. Precompute checkpoint-bound latent datasets.
bash experiments/precompute_visual_ogbench8_latents.sh

# 3. Train the final-goal-conditioned Action Chunk Priors.
bash experiments/train/train_action-prior-chunk_visual_ogbench8.sh

# 4. Train the general-uniform-future LatentPathFlow models.
bash experiments/train/train_latent_path_flow_visual_ogbench8.sh
```

Visual OGBench artifacts are written below `EXPERIMENT_ROOT` and consumed automatically by its evaluation launchers. Each launcher contains an editable `GPU_IDS` array and runs all eight datasets.

## Pretrained artifacts

The exact checkpoints selected for the release evaluation are stored in
[`IffYuan/LeWorldModelplusplus`](https://huggingface.co/IffYuan/LeWorldModelplusplus).
Download them with:

```bash
uvx --from huggingface_hub hf download IffYuan/LeWorldModelplusplus \
  --include "*/checkpoints/**" --local-dir artifacts
```

The checkpoint repository uses this layout:

```text
artifacts/
├── lewm-control-suite/
│   └── checkpoints/
│       ├── lewm/{cube,pusht,reacher,tworoom}/
│       ├── action-prior/{cube,pusht,reacher,tworoom}/
│       └── latent-path-flow/{h25,longh}/{cube,pusht,reacher,tworoom}/
└── visual-ogbench/
    └── checkpoints/{lewm,action-prior,latent-path-flow}/<dataset-tag>/
```

Point the corresponding checkpoint-root variables at these directories.


## Evaluation

### Planning effectiveness and long-horizon scaling (LeWM Control Suite)

Evaluate LeWM++ and the LeWM baseline on H25, H50, H75, and H100, then collect the results into one CSV file:

The LeWM baseline follows the official CEM300x30, H5/RH5 protocol with an
action block of 5. LeWM++ uses CEM300x5 and H2/RH1 with the same action block.

```bash
for horizon in 25 50 75 100; do
  bash "experiments/eval/eval_lewmpp_h${horizon}_4tasks.sh"
  bash "experiments/eval/eval_lewm_baseline_h${horizon}_4tasks.sh"
done

python impls/aggregate_lewmpp_results.py \
  --results-root outputs/eval \
  --output outputs/eval/lewm_control_suite_summary.csv
```

### More challenging tasks on Visual OGBench

Evaluate LeWM++ and the LeWM baseline on all eight datasets. Each launcher runs
the complete three-run evaluation and the LeWM++ launcher writes its aggregate
summary automatically:

```bash
bash experiments/eval/eval_lewmpp_visual_ogbench8.sh
bash experiments/eval/eval_lewm_baseline_visual_ogbench8.sh
```

To regenerate the LeWM++ summary from completed results without rerunning the
environments:

```bash
python impls/aggregate_visual_ogbench8_results.py \
  --results-root outputs/eval/visual_ogbench_lewmpp \
  --output outputs/eval/visual_ogbench_lewmpp/summary.json
```

## Acknowledgments

LeWM++ builds on [LeWorldModel](https://github.com/lucas-maes/le-wm) and [OGBench](https://github.com/seohongpark/ogbench). We thank their authors for releasing the latent world-model implementation, benchmark environments, datasets, and evaluation APIs that make this work possible. The DINO-WM transfer experiments described in the paper use the original [DINO-WM](https://github.com/gaoyuezhou/dino_wm) and are not included in this repository.

## License and citation

The repository retains the MIT license. A LeWM++ citation block will be added when the paper receives a public identifier.
