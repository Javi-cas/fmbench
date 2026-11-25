"""R1: scGPT whole-human, mean cell embedding per line (CPU). Weights in weights/scGPT_human (best_model.pt, args.json, vocab.json)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES, cpu_only, save
cpu_only()
import anndata as ad, numpy as np
from scgpt.tasks import embed_data
# macOS has no os.sched_getaffinity; scGPT uses it only for DataLoader num_workers. Empty set -> num_workers=0 (main process).
if not hasattr(os, "sched_getaffinity"): os.sched_getaffinity = lambda pid: set()
# Optional seed (Amendment 4): seeds the binning tie-break (numpy) and the 1,199-gene subsample (torch); writes R1_seed<n>.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else None
if SEED is not None:
    import torch; np.random.seed(SEED); torch.manual_seed(SEED)
ARM = "R1" if SEED is None else f"R1_seed{SEED}"
t0 = time.time(); a = ad.read_h5ad(f"{ROOT}/h5ad/all_lines.h5ad")
a.var["gene_name"] = a.var_names.astype(str)
out = embed_data(a, f"{ROOT}/weights/scGPT_human", gene_col="gene_name", max_length=1200, batch_size=64,
                 device="cpu", use_fast_transformer=False, return_new_adata=True)
E = out.X if not hasattr(out.X, "toarray") else out.X.toarray()
save(ARM, "scGPT whole-human" + ("" if SEED is None else f" (seed {SEED})"), E, a.obs["line"].values, t0)
