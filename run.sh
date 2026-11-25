#!/bin/bash
# Runs the three embeddings one after another on the CPU. Each writes emb/emb_R<n>.npz; a failure in one does not stop the others.
cd "$(dirname "$0")"; mkdir -p logs; export PATH="$HOME/.local/bin:$PATH"; set -o pipefail
run() { echo "== $1 $(date)"; if "${@:2}" 2>&1 | tee "logs/$1.log"; then echo "== $1 OK $(date)"; else echo "== $1 FAILED $(date) (see logs/$1.log)"; fi; }
[ -f emb/emb_R1.npz ] || run R1_scgpt      .venv_scgpt/bin/python fm/embed_scgpt.py
[ -f emb/emb_R2.npz ] || run R2_geneformer .venv_gf/bin/python    fm/embed_geneformer.py
[ -f emb/emb_R3.npz ] || run R3_state      .venv_tools/bin/python fm/embed_state.py
ls -la emb/
