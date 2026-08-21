#!/usr/bin/env bash
set -euo pipefail

# Server 23：将四个标准 LeWM JPEG95 Lance 数据集直接上传到 Hugging Face dataset repo。
DATA_ROOT=/data/dzb/stablewm-data/datasets
HF_PYTHON=python3
HF_TOKEN_PATH=/tmp/iffyuan-hf-token

export HF_TOKEN_PATH
export PYTHONPATH=/home/dzb/hf-upload-python${PYTHONPATH:+:$PYTHONPATH}

"$HF_PYTHON" - <<'PY'
from pathlib import Path

from huggingface_hub import HfApi

repo_id = "IffYuan/LeWM-Datasets"
data_root = "/data/dzb/stablewm-data/datasets"
token = Path("/tmp/iffyuan-hf-token").read_text().strip()
api = HfApi(token=token)

readme = b"""---
pretty_name: LeWM Datasets
---

# LeWM Datasets

Canonical JPEG95 Lance datasets used by the LeWM-JAX experiments.

Source snapshot: Server 23, `/data/dzb/stablewm-data/datasets/`.

- `cube_single_expert.lance`
- `pusht_expert_train.lance`
- `reacher.lance`
- `tworoom.lance`

Use these four directories as the shared source of truth when synchronizing
training data to other experiment servers.
"""

api.upload_file(
    path_or_fileobj=readme,
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Document canonical Server 23 LeWM Lance datasets",
)
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=data_root,
    allow_patterns=[
        "cube_single_expert.lance/**",
        "pusht_expert_train.lance/**",
        "reacher.lance/**",
        "tworoom.lance/**",
    ],
    num_workers=8,
    print_report=True,
    print_report_every=60,
)
PY
