"""Shared helpers: CPU threads, the cuda->cpu shim, and writing per-line mean embeddings."""
import os, time, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214", "Jurkat", "HepG2"]
os.makedirs(f"{ROOT}/emb", exist_ok=True)

def cpu_only():
    """Use every core, and send any hard-coded .to('cuda') / .cuda() call to the CPU instead."""
    import torch
    torch.set_num_threads(os.cpu_count())
    def fix(x):
        if isinstance(x, str) and x.startswith("cuda"): return "cpu"
        if isinstance(x, torch.device) and x.type == "cuda": return torch.device("cpu")
        return x
    t_to, m_to = torch.Tensor.to, torch.nn.Module.to
    def tensor_to(self, *a, **k):
        if "device" in k: k["device"] = fix(k["device"])
        return t_to(self, *[fix(x) for x in a], **k)
    def module_to(self, *a, **k):
        if "device" in k: k["device"] = fix(k["device"])
        return m_to(self, *[fix(x) for x in a], **k)
    torch.Tensor.to, torch.nn.Module.to = tensor_to, module_to
    torch.Tensor.cuda = lambda self, *a, **k: t_to(self, "cpu")
    torch.nn.Module.cuda = lambda self, *a, **k: m_to(self, "cpu")
    # tensor factories called with device="cuda" (e.g. Geneformer get_embs: torch.tensor(..., device="cuda"))
    def factory(f):
        def g(*a, **k):
            if "device" in k: k["device"] = fix(k["device"])
            return f(*a, **k)
        return g
    for n in ("tensor", "as_tensor", "zeros", "ones", "empty", "full", "arange", "rand", "randn", "randint", "eye",
              "linspace", "zeros_like", "ones_like", "empty_like", "full_like"):
        setattr(torch, n, factory(getattr(torch, n)))
    print(f"CPU only, {os.cpu_count()} threads", flush=True)

def save(arm, name, cell_embs, labels, t0):
    """cell_embs: (cells x dim) array; labels: line name per cell. Writes emb/emb_<arm>.npz (per-line means) + cells."""
    E = np.asarray(cell_embs, dtype=np.float32); labels = np.asarray(labels).astype(str)
    assert len(E) == len(labels) and np.isfinite(E).all(), "non-finite or misaligned embeddings"
    means = {l: E[labels == l].mean(0).astype(np.float64) for l in LINES}
    counts = {l: int((labels == l).sum()) for l in LINES}
    np.savez(f"{ROOT}/emb/emb_{arm}.npz", **means)
    np.savez_compressed(f"{ROOT}/emb/cells_{arm}.npz", E=E, line=labels)
    info = dict(arm=arm, model=name, dim=int(E.shape[1]), cells=counts, minutes=round((time.time() - t0) / 60, 1))
    json.dump(info, open(f"{ROOT}/emb/emb_{arm}.json", "w"), indent=1); print(json.dumps(info), flush=True)
