"""Per-cell information content of each representation (2026-10-26, Tier 2 decision; Tier 1 is not modified).
Variance decomposition over 7 lines x 2,000 cells, on the primary run of each arm:
  total = mean ||x - grand mean||^2;  between = sum_l (n_l/N) ||m_l - grand mean||^2;  within = mean_l mean_i ||x_i - m_l||^2
  seed noise = mean over cells of 0.5 ||x1 - x2||^2 (per-replicate noise variance from two independent draws of the same
  cell, paired by position as in fm/r1_seed_disagreement.py); zero by construction for R0 and R2 (deterministic)
  reproducible within = within - seed noise; cross-check: mean (x1 - m1_l).(x2 - m2_l), which needs no equal-noise assumption
Shares are over total; signal-to-noise = reproducible within / seed noise. Per line: the line's within and noise over
the pooled total, and the line's own reproducible fraction. Participation ratio (sum lambda)^2 / sum lambda^2 of the
pooled within-line covariance.
R0 per cell: log1p of per-cell CPM, CPM on the 6,360-gene panel as in Tier 1. Rule reference: all 6,360 genes;
sensitivity: R0's 2,000 Tier 1 genes. Sensitivity outside the rule: R0's count-sampling noise by binomial thinning
(p = 0.5, seed 20261026); per-replicate noise at half depth = 0.5 ||va - vb||^2, scaled by 1/2 to full depth (Poisson
variance in CPM scales with 1/depth; approximate after log1p).
Writes results/percell_info.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES
from score import r0_vectors

def pr(C):
    lam = np.clip(np.linalg.eigvalsh(C), 0, None); return float(lam.sum() ** 2 / (lam ** 2).sum())

def decompose(X, lab, X2=None):
    g = X.mean(0); tot = float(((X - g) ** 2).sum(1).mean())
    M = {l: X[lab == l].mean(0) for l in LINES}; Xc = X - np.stack([M[l] for l in lab])
    between = float(sum((lab == l).mean() * ((M[l] - g) ** 2).sum() for l in LINES))
    wl = {l: float((Xc[lab == l] ** 2).sum(1).mean()) for l in LINES}; within = float(np.mean(list(wl.values())))
    o = dict(dim=int(X.shape[1]), total=tot, between_share=between / tot, within_share=within / tot,
             pr_within=pr(Xc.T @ Xc / len(X)), per_line={})
    if X2 is None:
        nl = {l: 0.0 for l in LINES}; cross = None
    else:
        nl = {l: float((0.5 * ((X[lab == l] - X2[lab == l]) ** 2).sum(1)).mean()) for l in LINES}
        M2 = {l: X2[lab == l].mean(0) for l in LINES}; X2c = X2 - np.stack([M2[l] for l in lab])
        cl = {l: float((Xc[lab == l] * X2c[lab == l]).sum(1).mean()) for l in LINES}
        cross = float(np.mean(list(cl.values())))
        C12 = (Xc.T @ X2c + X2c.T @ Xc) / (2 * len(X)); o["pr_reproducible_cross"] = pr(C12)
    noise = float(np.mean(list(nl.values()))); rep = within - noise
    o.update(seed_noise_share=noise / tot, reproducible_within_share=rep / tot,
             snr=(rep / noise if noise > 0 else None), reproducible_within_share_cross=(cross / tot if cross is not None else None))
    for l in LINES:
        r = wl[l] - nl[l]
        o["per_line"][l] = dict(within_share=wl[l] / tot, reproducible_within_share=r / tot, seed_noise_share=nl[l] / tot,
                                reproducible_fraction_of_within=r / wl[l], snr=(r / nl[l] if nl[l] > 0 else None))
    return o

def main():
    import anndata as ad
    out = {}
    for arm in ("R1", "R2", "R3"):
        c1 = np.load(f"{ROOT}/emb/cells_{arm}.npz"); X, lab = c1["E"].astype(np.float64), c1["line"].astype(str)
        X2 = None
        if os.path.exists(f"{ROOT}/emb/cells_{arm}_seed2.npz"):
            c2 = np.load(f"{ROOT}/emb/cells_{arm}_seed2.npz"); assert (c2["line"] == c1["line"]).all() and c2["E"].shape == X.shape
            X2 = c2["E"].astype(np.float64)
        out[arm] = decompose(X, lab, X2); print(arm, "done", flush=True)
    a = ad.read_h5ad(f"{ROOT}/h5ad/all_lines.h5ad"); lab = a.obs["line"].values.astype(str)
    C = a.X.toarray().astype(np.float64); depth = C.sum(1, keepdims=True)
    V = np.log1p(C / depth * 1e6)
    _, r0g = r0_vectors(); gi = np.flatnonzero(np.isin(a.var_names.values.astype(str), r0g)); assert len(gi) == 2000
    out["R0_6360"] = decompose(V, lab); out["R0_2000"] = decompose(V[:, gi], lab); print("R0 done", flush=True)
    rng = np.random.default_rng(20261026); A = rng.binomial(C.astype(np.int64), 0.5).astype(np.float64); B = C - A
    Va = np.log1p(A / A.sum(1, keepdims=True) * 1e6); Vb = np.log1p(B / B.sum(1, keepdims=True) * 1e6)
    for key, cols in (("R0_6360", slice(None)), ("R0_2000", gi)):
        half = 0.5 * ((Va[:, cols] - Vb[:, cols]) ** 2).sum(1); full = half / 2; o = out[key]
        within = o["within_share"] * o["total"]
        o["count_noise_sensitivity"] = dict(count_noise_share_est=float(full.mean() / o["total"]),
                                            within_minus_count_noise_share=float((within - full.mean()) / o["total"]),
                                            per_line_count_noise_fraction_of_within={l: float(full[lab == l].mean() / (o["per_line"][l]["within_share"] * o["total"])) for l in LINES})
    json.dump(out, open(f"{ROOT}/results/percell_info.json", "w"), indent=1)
    print(f"{'rep':8s} {'dim':>5s} {'between':>8s} {'within':>8s} {'noise':>8s} {'repro':>8s} {'repro(x)':>8s} {'SNR':>7s} {'PR':>7s}")
    for k, o in out.items():
        snr = f"{o['snr']:.2f}" if o["snr"] is not None else "inf"; rx = o["reproducible_within_share_cross"]
        print(f"{k:8s} {o['dim']:5d} {o['between_share']:8.4f} {o['within_share']:8.4f} {o['seed_noise_share']:8.4f} "
              f"{o['reproducible_within_share']:8.4f} {(f'{rx:.4f}' if rx is not None else 'n/a'):>8s} {snr:>7s} {o['pr_within']:7.1f}")
    for k in ("R0_6360", "R0_2000"):
        s = out[k]["count_noise_sensitivity"]; print(k, "count-noise share est", round(s["count_noise_share_est"], 4), "within minus count noise", round(s["within_minus_count_noise_share"], 4))
    print("per line: reproducible within share / reproducible fraction of within / SNR")
    for l in LINES:
        cells = []
        for k in out:
            p = out[k]["per_line"][l]; snr = f"{p['snr']:.1f}" if p["snr"] is not None else "inf"
            cells.append(f"{k}: {p['reproducible_within_share']:.4f}/{p['reproducible_fraction_of_within']:.3f}/{snr}")
        print(f"{l:14s} " + "  ".join(cells))

if __name__ == "__main__":
    main()
