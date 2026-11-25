"""Tier 1: 2,000 random control cells per line, raw counts on the genes measured in all seven lines (Amendment 1).
Runs on the device VM, read-only on VCC26, one line per call:  python3 extract_controls.py genes | <line>
Lines: Jurkat HepG2 K562 HCT116 HEK293T H1_2025 H1_GSE295214.  Seed 20261024.
"""
import os, sys, re, glob, json, numpy as np, h5py
R = os.path.expanduser("~/mnt/VCC26/raw"); OUT = os.path.expanduser("~/mnt/scratch/fmbench/cells")
os.makedirs(OUT, exist_ok=True)
SEED, N = 20261024, 2000
H5AD = {"Jurkat": f"{R}/gse264667/GSE264667_jurkat_raw_singlecell_01.h5ad",
        "HepG2": f"{R}/gse264667/GSE264667_hepg2_raw_singlecell_01.h5ad",
        "K562": f"{R}/replogle_k562_gwps_raw/replogle_k562_gwps_raw/K562_gwps_raw_singlecell_01.h5ad"}
H1 = f"{R}/arc_h1_2025/adata_Training.h5ad"
XA = {"HCT116": f"{R}/xatlas_hct116", "HEK293T": f"{R}/xatlas_hek293t"}
GEMS = sorted(glob.glob(f"{R}/gse295214/h5/*_filtered_feature_bc_matrix.h5"))

def cat(h, key):
    g = h[key]
    if isinstance(g, h5py.Group):                                   # anndata >= 0.8 categorical
        c = g["categories"][:].astype(str); return c[g["codes"][:]]
    c = h[key.rsplit("/", 1)[0] + "/__categories/" + key.rsplit("/", 1)[1]][:].astype(str)  # legacy layout
    return c[g[:]]

def genes_of(line):
    if line in H5AD:
        with h5py.File(H5AD[line], "r") as h: return cat(h, "var/gene_name")
    if line == "H1_2025":
        with h5py.File(H1, "r") as h: return h["var/_index"][:].astype(str)
    if line in XA:
        import pyarrow.parquet as pq
        return pq.read_table(f"{R}/xatlas_meta/gene_metadata.parquet").column("gene_name").to_numpy(zero_copy_only=False).astype(str)
    with h5py.File(GEMS[0], "r") as h:
        ft = h["matrix/features/feature_type"][:].astype(str); return h["matrix/features/name"][:].astype(str)[ft == "Gene Expression"]

def sample(n_total, rng):
    return np.sort(rng.choice(n_total, size=min(N, n_total), replace=False))

def take(line, common):
    rng = np.random.default_rng(SEED); gi = {g: i for i, g in enumerate(common)}
    if line in H5AD:
        with h5py.File(H5AD[line], "r") as h:
            ctrl = np.flatnonzero(cat(h, "obs/gene") == "non-targeting"); rows = ctrl[sample(len(ctrl), rng)]
            vg = cat(h, "var/gene_name"); col = np.array([gi.get(g, -1) for g in vg]); ok = col >= 0
            X = np.zeros((len(rows), len(common)), np.float32)
            for a in range(0, len(rows), 100):
                X[a:a + 100][:, col[ok]] = h["X"][rows[a:a + 100]][:, ok]
            return X, dict(controls=int(len(ctrl)))
    if line == "H1_2025":
        with h5py.File(H1, "r") as h:
            tg = cat(h, "obs/target_gene"); ctrl = np.flatnonzero(tg == "non-targeting"); rows = ctrl[sample(len(ctrl), rng)]
            vg = h["var/_index"][:].astype(str); col = np.array([gi.get(g, -1) for g in vg])
            ip = h["X/indptr"]; X = np.zeros((len(rows), len(common)), np.float32)
            for i, r in enumerate(rows):
                s, e = int(ip[r]), int(ip[r + 1]); j = h["X/indices"][s:e]; v = h["X/data"][s:e]; c = col[j]; m = c >= 0
                X[i, c[m]] = v[m]
            return X, dict(controls=int(len(ctrl)))
    if line in XA:
        import pyarrow.parquet as pq
        meta = pq.read_table(f"{R}/xatlas_meta/gene_metadata.parquet").to_pandas()
        tok = np.full(int(meta.gene_token_id.max()) + 1, -1)
        for t, g in zip(meta.gene_token_id, meta.gene_name): tok[t] = gi.get(str(g), -1)
        files = sorted(glob.glob(f"{XA[line]}/*.parquet"), key=lambda p: int(re.search(r"Batch(\d+)", p).group(1)))
        idx = []
        for k, f in enumerate(files):
            t = pq.read_table(f, columns=["gene_target", "pass_guide_filter"])
            m = (t.column(0).to_numpy(zero_copy_only=False).astype(str) == "Non-Targeting") & t.column(1).to_numpy(zero_copy_only=False).astype(bool)
            idx += [(k, r) for r in np.flatnonzero(m)]
        pick = [idx[i] for i in sample(len(idx), rng)]
        part = f"{OUT}/parts_{line}"; os.makedirs(part, exist_ok=True); import time; t0 = time.time()
        for k in sorted({p[0] for p in pick}):                     # resumable: one part file per batch
            pf = f"{part}/b{k}.npy"
            if os.path.exists(pf): continue
            if time.time() - t0 > 140: sys.exit(f"partial: rerun to continue ({len(os.listdir(part))} batches done)")
            rows = set(r for kk, r in pick if kk == k); base = 0; Xk = []
            for b in pq.ParquetFile(files[k]).iter_batches(batch_size=512, columns=["gene_token_id", "gene_expression"]):
                want = [r - base for r in range(base, base + b.num_rows) if r in rows]; base += b.num_rows
                if not want: continue
                tb = b.take(want)
                for ids, vals in zip(tb.column(0).to_pylist(), tb.column(1).to_pylist()):
                    v = np.zeros(len(common), np.float32); c = tok[np.asarray(ids)]; m = c >= 0; v[c[m]] = np.asarray(vals)[m]; Xk.append(v)
            np.save(pf, np.array(Xk, np.float32))
        X = np.concatenate([np.load(f"{part}/b{k}.npy") for k in sorted({p[0] for p in pick})])
        return X, dict(controls=int(len(idx)), batches=len(files))
    # H1_GSE295214: GEX QC and guide calls as in atlasshift_local/gse295214.py; GEMs visited in seeded random order
    import scipy.sparse as sp
    rx = re.compile(r"^(.+?)_[+-]_\d+"); order = rng.permutation(len(GEMS)); cells = []; used = []
    for k in order:
        with h5py.File(GEMS[k], "r") as f:
            m = f["matrix"]; shape = tuple(int(x) for x in m["shape"][:])
            Xg = sp.csc_matrix((m["data"][:].astype(np.float32), m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
            ft = m["features/feature_type"][:].astype(str); fn = m["features/name"][:].astype(str)
        gex = np.flatnonzero(ft == "Gene Expression"); gd = np.flatnonzero(ft == "CRISPR Guide Capture")
        XG = Xg[:, gex]; lib = np.asarray(XG.sum(1)).ravel()
        mt = np.asarray(XG[:, np.char.startswith(fn[gex], "MT-")].sum(1)).ravel() / np.maximum(lib, 1)
        Rg = Xg[:, gd].toarray(); o = np.sort(Rg, 1); top, sec = o[:, -1], o[:, -2]
        lab = np.array(["non-targeting" if x.startswith("non-targeting") else "x" for x in fn[gd]])[Rg.argmax(1)]
        keep = np.flatnonzero((lib >= 5000) & (mt <= 0.20) & (top >= 5) & (top >= 3 * sec) & (lab == "non-targeting"))
        col = np.array([gi.get(g, -1) for g in fn[gex]]); ok = col >= 0
        sub = XG[keep][:, ok].toarray(); full = np.zeros((len(keep), len(common)), np.float32); full[:, col[ok]] = sub
        cells.append(full); used.append(os.path.basename(GEMS[k]))
        if sum(len(c) for c in cells) >= 4 * N: break
    Xall = np.concatenate(cells); X = Xall[sample(len(Xall), rng)]
    return X, dict(controls_in_visited_gems=int(len(Xall)), gems=used)

if __name__ == "__main__":
    a = sys.argv[1]
    if a == "genes":
        lines = list(H5AD) + ["H1_2025"] + list(XA) + ["H1_GSE295214"]
        sets = {l: set(genes_of(l)) for l in lines}
        common = sorted(set.intersection(*sets.values()))
        np.save(f"{OUT}/common_genes.npy", np.array(common))
        print({l: len(s) for l, s in sets.items()}, "common", len(common))
    else:
        common = np.load(f"{OUT}/common_genes.npy").astype(str)
        X, info = take(a, list(common))
        import scipy.sparse as sp
        S = sp.csr_matrix(X)
        np.savez_compressed(f"{OUT}/cells_{a}.npz", data=S.data, indices=S.indices, indptr=S.indptr, shape=S.shape, genes=common)
        info.update(line=a, cells=int(X.shape[0]), median_umi_common=float(np.median(X.sum(1))))
        json.dump(info, open(f"{OUT}/cells_{a}.json", "w"), indent=1)
        print(json.dumps(info)[:400])
