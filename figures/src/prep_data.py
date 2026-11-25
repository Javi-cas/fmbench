"""Real data behind the redesigned figures (no benchmark results used)."""
import numpy as np, scipy.sparse as sp, umap
FB = "/mnt/user-data/uploads/scratch/fmbench"
LINES = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214", "Jurkat", "HepG2"]
Xs, lab = [], []
rng = np.random.default_rng(0)
for l in LINES:
    z = np.load(f"{FB}/cells/cells_{l}.npz"); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    idx = rng.choice(X.shape[0], 700, replace=False); Xs.append(X[idx]); lab += [l] * 700
X = sp.vstack(Xs).astype(np.float64); lib = np.asarray(X.sum(1)).ravel()
L = np.log1p(np.asarray((sp.diags(1e4 / lib) @ X).todense()))
hv = np.argsort(L.var(0))[::-1][:2000]; Z = L[:, hv]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
U, S, Vt = np.linalg.svd(Z, full_matrices=False); P = U[:, :30] * S[:30]
E = umap.UMAP(n_neighbors=30, min_dist=0.35, random_state=7).fit_transform(P)
np.savez("data_umap.npz", E=E, line=np.array(lab))
# real knockdown responses: shared knockdowns x most variable genes, K562 and HCT116
a, b = np.load(f"{FB}/data/src_K562.npz"), np.load(f"{FB}/data/src_HCT116.npz")
ta, tb = a["targets"].astype(str), b["targets"].astype(str); common = [t for t in ta if t in set(tb)]
ia = {t: i for i, t in enumerate(ta)}; ib = {t: i for i, t in enumerate(tb)}
A = a["L"][[ia[t] for t in common]]; B = b["L"][[ib[t] for t in common]]
ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
strength = np.nanmean(np.abs(A[:, ok]), 1) + np.nanmean(np.abs(B[:, ok]), 1)
ks = np.argsort(strength)[::-1][:16]
G = np.flatnonzero(ok); gv = np.argsort((A[ks][:, G].var(0) + B[ks][:, G].var(0)))[::-1][:28]
np.savez("data_heat.npz", K562=A[ks][:, G[gv]], HCT116=B[ks][:, G[gv]], kd=np.array(common)[ks], genes=a["genes"].astype(str)[G[gv]])
print("umap", E.shape, "heat", A[ks][:, G[gv]].shape, list(np.array(common)[ks][:6]))
