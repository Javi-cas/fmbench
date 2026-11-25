#!/bin/bash
# Amendment 6: re-embed the R3 cells under seed 2, measure the disagreement with emb_R3.npz, then re-score so the
# noise-floor row for R3 is added. The re-score must reproduce every existing number (checked at the end).
cd "$(dirname "$0")"; set -o pipefail
echo "== R3_state_seed2 $(date)"
[ -f emb/emb_R3_seed2.npz ] || .venv_tools/bin/python fm/embed_state.py 2 2>&1 | tee logs/R3_state_seed2.log || { echo "== seed 2 FAILED $(date)"; exit 1; }
echo "== disagreement $(date)"
.venv_tools/bin/python fm/r1_seed_disagreement.py R3 2>&1 | tee logs/R3_seed_disagreement.log
echo "== re-score $(date)"
cp results/score_fm.json results/score_fm_before_R3seed2.json
.venv_tools/bin/python fm/score.py > logs/score_fm_with_R3seed2.log 2>&1 || { echo "== re-score FAILED"; exit 1; }
.venv_tools/bin/python - <<'EOF'
import json
a, b = json.load(open("results/score_fm_before_R3seed2.json")), json.load(open("results/score_fm.json"))
bad = [(n, g, k) for n in a["folds"] for g in a["folds"][n]["gene_sets"] for k, v in a["folds"][n]["gene_sets"][g]["rho"].items()
       if abs(b["folds"][n]["gene_sets"][g]["rho"][k] - v) > 1e-12]
print("existing rho values reproduced exactly" if not bad else f"CHANGED: {bad}")
EOF
echo "== done $(date)"
