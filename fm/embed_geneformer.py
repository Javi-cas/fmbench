"""R2: Geneformer V2-104M, CLS cell embedding per line (CPU). Repo snapshot in weights/Geneformer (installed with pip)."""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES, cpu_only, save
cpu_only()
import numpy as np
from geneformer import TranscriptomeTokenizer, EmbExtractor
t0 = time.time(); W = f"{ROOT}/weights/Geneformer"
cands = [d for d in glob.glob(f"{W}/*V2*104M*") if "CLcancer" not in d and os.path.exists(f"{d}/config.json")]
assert cands, f"no Geneformer V2-104M model folder with config.json under {W}"
model_dir = sorted(cands)[0]; print("model", model_dir, flush=True)
tok_in, tok_out, emb_out = f"{ROOT}/work/gf_in", f"{ROOT}/work/gf_tok", f"{ROOT}/work/gf_emb"
for d in (tok_in, tok_out, emb_out): os.makedirs(d, exist_ok=True)
if not os.path.exists(f"{tok_in}/all_lines.h5ad"): os.symlink(f"{ROOT}/h5ad/all_lines.h5ad", f"{tok_in}/all_lines.h5ad")
nproc = max(1, min(8, os.cpu_count() // 2))
if not os.path.exists(f"{tok_out}/lines.dataset"):
    TranscriptomeTokenizer({"line": "line"}, nproc=nproc, model_version="V2").tokenize_data(tok_in, tok_out, "lines", file_format="h5ad")
ex = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cls", max_ncells=None, emb_layer=-1,
                  emb_label=["line"], forward_batch_size=8, nproc=nproc, model_version="V2")
df = ex.extract_embs(model_dir, f"{tok_out}/lines.dataset", emb_out, "lines")
lab = df["line"].values; E = df.drop(columns=[c for c in df.columns if c == "line"]).values
save("R2", "Geneformer V2-104M (cls, emb_layer -1)", E, lab, t0)
