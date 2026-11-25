"""Build the input h5ad files for the three foundation models from cells/cells_<line>.npz (raw counts, 6,360 common genes).
Writes h5ad/cells_<line>.h5ad and h5ad/all_lines.h5ad (obs: line, n_counts; var index: gene symbol; var: gene_name, ensembl_id)."""
import os, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214", "Jurkat", "HepG2"]
os.makedirs(f"{ROOT}/h5ad", exist_ok=True)
ens = pd.read_csv(f"{ROOT}/cells/common_genes_ensembl.tsv", sep="\t").set_index("gene_name")["ensembl_id"]
parts = []
for l in LINES:
    z = np.load(f"{ROOT}/cells/cells_{l}.npz")
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).astype(np.float32)
    g = z["genes"].astype(str)
    var = pd.DataFrame({"gene_name": g, "ensembl_id": ens.loc[g].values}, index=g)
    obs = pd.DataFrame({"line": l, "n_counts": np.asarray(X.sum(1)).ravel()}, index=[f"{l}_{i}" for i in range(X.shape[0])])
    a = ad.AnnData(X=X, obs=obs, var=var); a.write_h5ad(f"{ROOT}/h5ad/cells_{l}.h5ad"); parts.append(a)
    print(l, a.shape, "median counts", float(np.median(obs.n_counts)), flush=True)
ad.concat(parts, merge="same").write_h5ad(f"{ROOT}/h5ad/all_lines.h5ad")
print("done")
