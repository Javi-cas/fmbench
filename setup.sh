#!/bin/bash
# One-time setup: model weights + one Python environment per model. Safe to re-run (skips what is already there).
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p weights logs
command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
HF="uvx --from huggingface_hub hf"

echo "== weights: State SE-600M";  [ -n "$(ls weights/SE-600M/*.ckpt weights/SE-600M/**/*.ckpt 2>/dev/null)" ] || $HF download arcinstitute/SE-600M --local-dir weights/SE-600M
echo "== weights: Geneformer";     [ -d weights/Geneformer/geneformer ] || $HF download ctheodoris/Geneformer --local-dir weights/Geneformer \
      --include "geneformer/**" "*104M*" "setup.py" "pyproject.toml" "requirements.txt" "MANIFEST.in" "README.md"
echo "== weights: scGPT whole-human"; [ -f weights/scGPT_human/best_model.pt ] || uvx gdown --folder \
      "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O weights/scGPT_human

echo "== env: tools";      [ -d .venv_tools ] || { uv venv .venv_tools --python 3.11 && uv pip install -p .venv_tools anndata pandas scipy numpy; }
echo "== env: scGPT";      [ -d .venv_scgpt ] || { uv venv .venv_scgpt --python 3.10 && uv pip install -p .venv_scgpt scgpt "torch==2.3.1" "torchtext==0.18.0" "numpy<2" anndata scanpy; }
echo "== env: Geneformer"; [ -d .venv_gf ]    || { uv venv .venv_gf --python 3.11 && uv pip install -p .venv_gf ./weights/Geneformer anndata; }
echo "== env: State";      command -v state >/dev/null || uv tool install arc-state

echo "== input h5ad files"; [ -f h5ad/all_lines.h5ad ] || .venv_tools/bin/python fm/make_h5ad.py
echo "setup done"; ls -la weights/*
