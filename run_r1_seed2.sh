#!/bin/bash
# Amendment 4: after run.sh finishes, re-embed the same R1 cells under seed 2 and measure the disagreement with emb_R1.npz.
cd "$(dirname "$0")"; set -o pipefail
while pgrep -f "bash ./run.sh" >/dev/null; do sleep 60; done
[ -f emb/emb_R1.npz ] || { echo "emb/emb_R1.npz missing: primary R1 did not finish, not running seed 2"; exit 1; }
echo "== R1_scgpt_seed2 $(date)"
[ -f emb/emb_R1_seed2.npz ] || .venv_scgpt/bin/python fm/embed_scgpt.py 2 2>&1 | tee logs/R1_scgpt_seed2.log || { echo "== seed 2 FAILED $(date)"; exit 1; }
echo "== disagreement $(date)"
.venv_tools/bin/python fm/r1_seed_disagreement.py 2>&1 | tee logs/R1_seed_disagreement.log
echo "== done $(date)"
