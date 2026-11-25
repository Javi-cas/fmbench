"""Tier 1 data extraction (runs on the device VM, read-only on VCC26). PLAN.md sha256 10ff753f...8ea.
Per source line: log2((target_mean_cpm + 1) / (global_control_mean_cpm + 1)) for fold knockdowns (Jurkat or HepG2,
>= 50 cells) on the union of the two folds' tested genes, targets with >= 20 cells in that source (Amendment 1).
Also the full-axis control mean CPM of every line (sources + held-out lines) for the raw-profile arm R0.
Large uncompressed npz members are memory-mapped so the 3 GB VM never loads a full matrix.
"""
import os, sys, zipfile, numpy as np
from numpy.lib import format as F
V = os.path.expanduser("~/mnt/VCC26/scratch/vcc"); A, E = f"{V}/atlas_data", f"{V}/ext"
OUT = os.path.expanduser("~/mnt/scratch/fmbench/data"); os.makedirs(OUT, exist_ok=True)
SRC = {"K562": f"{A}/K562_GWPS_CPM_full_statistics.npz", "HCT116": f"{E}/essential_src/HCT116_full_statistics.npz",
       "HEK293T": f"{E}/essential_src/HEK293T_allbatches_statistics.npz", "H1_2025": f"{A}/H1_2025_full_statistics.npz",
       "H1_GSE295214": f"{A}/H1_GSE295214_statistics.npz"}
MIN_SRC_CELLS = 20

def member(path, name):
    """np array for one npz member; memmap if stored uncompressed."""
    z = zipfile.ZipFile(path); info = z.getinfo(name + ".npy")
    if info.compress_type != 0:
        return np.load(path)[name]
    with z.open(info) as fh:
        v = F.read_magic(fh); shape, fo, dt = F._read_array_header(fh, v); hdr = fh.tell()
    with open(path, "rb") as fh:
        fh.seek(info.header_offset + 26); n, m = (int(x) for x in np.frombuffer(fh.read(4), "<u2"))
    off = info.header_offset + 30 + n + m + hdr
    return np.memmap(path, dtype=dt, mode="r", shape=shape, offset=off, order="F" if fo else "C")

folds = {}
for c in ("jurkat", "hepg2"):
    t = np.load(f"{E}/fold_{c}/truth.npz", allow_pickle=False)
    folds[c] = dict(perts=t["perts"].astype(str), genes=t["genes"].astype(str), cols=t["cols"])
G = folds["jurkat"]["genes"]; assert (folds["hepg2"]["genes"] == G).all()
U = np.union1d(folds["jurkat"]["cols"], folds["hepg2"]["cols"]); UG = G[U]
P = np.union1d(folds["jurkat"]["perts"], folds["hepg2"]["perts"])
print(f"gene union {len(U)}, knockdown union {len(P)}", flush=True)

ctrl = {}
for n, path in SRC.items():
    g = member(path, "genes").astype(str); gi = {x: i for i, x in enumerate(g)}
    t = member(path, "targets").astype(str); nc = np.asarray(member(path, "n_cells"))
    keep = np.flatnonzero(np.isin(t, P) & (nc >= MIN_SRC_CELLS))
    cm = np.asarray(member(path, "global_control_mean_cpm"), dtype=np.float64)
    meas = np.asarray(member(path, "measured_genes"))
    colsrc = np.array([gi.get(x, -1) for x in UG]); ok = colsrc >= 0
    TM = member(path, "target_mean_cpm")
    L = np.full((len(keep), len(U)), np.nan, np.float32)
    for a in range(0, len(keep), 256):
        rows = np.asarray(TM[keep[a:a + 256]], dtype=np.float64)[:, colsrc[ok]]
        L[a:a + 256][:, ok] = np.log2((rows + 1) / (cm[colsrc[ok]][None, :] + 1))
    L[:, ok & ~meas[np.where(ok, colsrc, 0)]] = np.nan
    np.savez_compressed(f"{OUT}/src_{n}.npz", targets=t[keep], n_cells=nc[keep], L=L, genes=UG)
    full = np.full(len(G), np.nan); gfull = np.array([gi.get(x, -1) for x in G]); okf = gfull >= 0
    full[okf] = cm[gfull[okf]]; ctrl[n] = full
    print(f"{n}: {len(keep)} fold knockdowns with >= {MIN_SRC_CELLS} cells, {ok.sum()}/{len(U)} genes", flush=True)
for c, nm in (("jurkat", "Jurkat"), ("hepg2", "HepG2")):
    ctrl[nm] = np.load(f"{E}/fold_{c}/ctrl_moments.npz")["mean"].astype(np.float64) * 1e6
np.savez_compressed(f"{OUT}/controls_cpm.npz", genes=G, **{k: v.astype(np.float32) for k, v in ctrl.items()})
print("done", flush=True)
