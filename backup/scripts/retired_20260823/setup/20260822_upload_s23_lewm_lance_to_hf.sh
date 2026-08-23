#!/usr/bin/env bash
set -euo pipefail

# Server 23：将四个标准 LeWM JPEG95 Lance 数据集直接上传到 Hugging Face dataset repo。
DATA_ROOT=/data/dzb/stablewm-data/datasets
HF_PYTHON=python3
HF_TOKEN_PATH=/tmp/iffyuan-hf-token
HF_ENDPOINT=https://hf-mirror.com

export HF_ENDPOINT HF_TOKEN_PATH
export PYTHONPATH=/home/dzb/hf-upload-python${PYTHONPATH:+:$PYTHONPATH}

"$HF_PYTHON" - <<'PY'
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import huggingface_hub._commit_api as commit_api
from huggingface_hub import HfApi


# hf-mirror's LFS batch response currently advertises hf-mirror.org, which does
# not resolve. The same signed upload endpoint is available on hf-mirror.com.
_original_lfs_upload = commit_api.lfs_upload


def _mirror_lfs_upload(operation, lfs_batch_action, token=None, headers=None, endpoint=None):
    patched_action = deepcopy(lfs_batch_action)
    for action in (patched_action.get("actions") or {}).values():
        parsed = urlsplit(action["href"])
        if parsed.hostname == "hf-mirror.org":
            action["href"] = urlunsplit(
                (parsed.scheme, "hf-mirror.com", parsed.path, parsed.query, parsed.fragment)
            )
    return _original_lfs_upload(
        operation=operation,
        lfs_batch_action=patched_action,
        token=token,
        headers=headers,
        endpoint=endpoint,
    )


commit_api.lfs_upload = _mirror_lfs_upload

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
    num_workers=4,
    print_report=True,
    print_report_every=60,
)
PY
