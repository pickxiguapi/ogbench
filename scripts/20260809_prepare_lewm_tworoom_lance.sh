#!/usr/bin/env bash
set -euo pipefail

cd /home/dzb/ogbench

PYTHONPATH=/home/dzb/stable-worldmodel \
  /home/dzb/ogbench/.venv/bin/python \
  scripts/convert_lewm_hdf5_to_lance.py \
  /home/dzb/stable-worldmodel \
  /data/dzb/stablewm-data/datasets/tworoom.h5 \
  /data/dzb/stablewm-data/datasets/tworoom.lance
