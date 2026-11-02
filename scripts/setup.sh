#!/usr/bin/env bash
# Reproducible environment for the experiments (tested on NVIDIA B200 MIG 1g.45gb, driver 570, CUDA 12.8).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q -U pip uv
.venv/bin/uv pip install --python .venv/bin/python -r requirements.txt
# The default PyPI torch wheel is built for CUDA 12.6 and has no sm_100 kernels.
.venv/bin/uv pip install --python .venv/bin/python --reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch; x=torch.ones(8,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), x.sum().item())"
