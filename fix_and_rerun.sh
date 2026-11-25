#!/bin/bash
# Run this ON THE MAC (not in the Claude VM): the venvs are macOS venvs, so wheels must be
# installed by macOS Python. Fixes the two failed arms, then re-runs only what is missing.
# Corrected 2026-10-25 (see AMENDMENTS.md, Amendment 3). Safe to re-run.
cd "$(dirname "$0")"; export PATH="$HOME/.local/bin:$PATH"; set -e

# R1 scGPT: datasets 2.14.4 calls pa.PyExtensionType, which pyarrow 25 no longer has. scgpt 0.2.4 pins
# datasets<3.0 and only imports it (scgpt.scbank), so pin pyarrow back rather than move datasets.
uv pip install -p .venv_scgpt "pyarrow==14.0.2"
# (macOS has no os.sched_getaffinity, used by scGPT for DataLoader workers: shimmed in fm/embed_scgpt.py.)

# R2 Geneformer: the setup.sh download was partial; weights/Geneformer/geneformer held only the four .pkl
# dictionaries, so the installed package was empty. Fetch the source at the same HF revision as the weights.
GF_REV=1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5
if [ ! -f weights/Geneformer/geneformer/emb_extractor.py ]; then
  uvx --from huggingface_hub hf download ctheodoris/Geneformer --revision $GF_REV --local-dir weights/Geneformer \
    geneformer/__init__.py geneformer/classifier.py geneformer/classifier_utils.py geneformer/collator_for_classification.py \
    geneformer/emb_extractor.py geneformer/evaluation_utils.py geneformer/in_silico_perturber.py \
    geneformer/in_silico_perturber_stats.py geneformer/mtl_classifier.py geneformer/perturber_utils.py \
    geneformer/pretrainer.py geneformer/tokenizer.py geneformer/mtl/__init__.py geneformer/mtl/collators.py \
    geneformer/mtl/data.py geneformer/mtl/eval_utils.py geneformer/mtl/model.py geneformer/mtl/train.py geneformer/mtl/utils.py
fi
# requirements.txt pins transformers==4.46 (setup.py does not); Geneformer imports internals removed in transformers 5.
# --reinstall/--refresh: uv would otherwise reuse its cached (empty) 0.1.0 wheel.
uv pip install -p .venv_gf -r weights/Geneformer/requirements.txt ./weights/Geneformer \
  --reinstall-package geneformer --refresh-package geneformer
# (Geneformer's torch.tensor(..., device="cuda") is redirected by the factory shim in fm/common.py.)

# Do not relaunch on an import error. Geneformer is imported from outside the repo so the source folder cannot mask it.
.venv_scgpt/bin/python -W ignore -c "from scgpt.tasks import embed_data; print('R1 import OK')"
(cd /tmp && "$OLDPWD/.venv_gf/bin/python" -W ignore -c "from geneformer import TranscriptomeTokenizer, EmbExtractor; print('R2 import OK')")

set +e
nohup caffeinate -i bash ./run.sh > logs/run_$(date +%m%d_%H%M).log 2>&1 &
echo "relaunched, pid $!"
