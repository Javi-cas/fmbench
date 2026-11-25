"""R3: State SE-600M cell embedding per line (CPU), via the `state emb transform` CLI, one line at a time.
Optional seed (Amendment 6): `embed_state.py 2` runs the same CLI with RNGs seeded and writes R3_seed2."""
import os, sys, glob, time, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES, save
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else None
STATE_PY = os.path.expanduser("~/.local/share/uv/tools/arc-state/bin/python")
t0 = time.time(); W = f"{ROOT}/weights/SE-600M"
assert os.path.exists(f"{W}/config.yaml"), f"no SE-600M model folder at {W}"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
Es, labs = [], []
for l in LINES:
    out = f"{ROOT}/work/state_{l}.npy" if SEED is None else f"{ROOT}/work/state_seed{SEED}_{l}.npy"
    if not os.path.exists(out):
        cli = ["state"] if SEED is None else [STATE_PY, f"{ROOT}/fm/state_seeded.py", str(SEED)]
        cmd = cli + ["emb", "transform", "--model-folder", W,           # model card usage: the tool picks the checkpoint
                     "--input", f"{ROOT}/h5ad/cells_{l}.h5ad", "--output", out]
        print(" ".join(cmd), flush=True); subprocess.run(cmd, check=True, env=env)
    E = np.load(out); Es.append(E); labs += [l] * len(E); print(l, E.shape, flush=True)
save("R3" if SEED is None else f"R3_seed{SEED}",
     "State SE-600M (default checkpoint chosen by `state emb transform --model-folder`)" + ("" if SEED is None else f" (seed {SEED})"),
     np.concatenate(Es), labs, t0)
