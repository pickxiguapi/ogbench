#!/usr/bin/env bash
set -euo pipefail

cd /root/data/yyf/ogbench

PYTHONPATH=/root/data/yyf/stable-worldmodel \
  /root/data/yyf/stable-worldmodel/.venv/bin/python \
  scripts/convert_lewm_hdf5_to_lance.py \
  /root/data/yyf/stable-worldmodel \
  /root/data/yyf/stable-worldmodel/datasets/tworoom.h5 \
  /root/data/yyf/stable-worldmodel/datasets/tworoom.lance
