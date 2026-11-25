"""EXPLORATORY (2026-10-26; see results/ensemble_probe/README.md). Scores JaviCell ensemble variants next to the Tier 1
arms with the gate metric (verbatim, via fm/score.py), the Tier 1 knockdowns, gene sets and bootstrap (2,000 resamples,
seed 20261014; the same resample indices as Tier 1). Each variant is scored once; nothing is fitted on Jurkat or HepG2.
Tier 1 arms are rebuilt with their stored T and must reproduce results/score_fm.json exactly. Writes results/ensemble_probe/."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import (FOLD, SOURCES, N_BOOT, SEED, ROOT, CVAE, LRJ_RESP, fold, source_mats, predict, softmax_w, centred_cos,
                   r0_vectors, emb_vectors, target_mask, rho_stats, rho_from)
OUT = f"{ROOT}/results/ensemble_probe"; EXT = "/Volumes/VCC26/scratch/vcc/ext"
CTX = {"Jurkat": "jurkat", "HepG2": "hepg2"}
W0, CD4_W, ENS_W, SPEC = {"K562": 2.0, "HCT116": 1.0, "HEK293T": 1.0, "H1": 2.0}, 0.5, 0.5, 1.5
V7W = {"K562": 1.0, "HCT116": 0.5}
SHRINK = (0.25, 0.5, 0.75, 1.0)

def ens_files(c):
    d, e = f"{EXT}/fold_{c}", f"{EXT}/fold_{c}_eqw"
    F = {"ens": f"{d}/pred_ens.npz", "ens_nocd4": f"{d}/pred_ens_nocd4.npz", "atlas": f"{d}/pred_atlas.npz",
         "atlas_nocd4": f"{d}/pred_atlas_nocd4.npz", "eqw_ens": f"{e}/pred_ens.npz", "eqw_ens_nocd4": f"{e}/pred_ens_nocd4.npz",
         "eqw_atlas_nocd4": f"{e}/pred_atlas_nocd4.npz"}
    for s in SHRINK: F[f"ens_shrink{int(s * 100):03d}"] = f"{OUT}/pred_{c}_ens_shrink{int(s * 100):03d}.npz"
    return F

def nominal_weights(c, W, cd4_w, ens_w, spec=1.0):
    """Per-knockdown nominal weights in log2fc space from the code: atlas side W*present/(sum W present + cd4_w*cd4),
    times (1 - ens_w'); v7 side ens_w' split by v7's own weights over its present sources. spec rescales the v7 side's
    specific-effect amplitude (1.5 in v7) before renormalising the two sides."""
    z = np.load(f"{OUT}/presence_{c}.npz"); names = list(z["names"]); pres, cd4 = z["pres"], z["cd4"]
    v7n = list(z["v7names"]); v7p = z["v7pres"]
    cols = ["K562", "HCT116", "HEK293T", "H1", "CD4"]; Wt = np.zeros((len(pres), len(cols)))
    den = sum(W[n] * pres[:, i] for i, n in enumerate(names)) + cd4_w * cd4
    a = np.zeros_like(Wt)
    for i, n in enumerate(names): a[:, cols.index(n)] = np.divide(W[n] * pres[:, i], den, out=np.zeros(len(den)), where=den > 0)
    a[:, 4] = np.divide(cd4_w * cd4, den, out=np.zeros(len(den)), where=den > 0)
    vden = sum(V7W[k] * v7p[:, j] for j, k in enumerate(v7n)); v = np.zeros_like(Wt)
    for j, k in enumerate(v7n): v[:, cols.index(k)] = np.divide(V7W[k] * v7p[:, j], vden, out=np.zeros(len(vden)), where=vden > 0)
    has_a, has_v = a.sum(1) > 0, v.sum(1) > 0
    wa = np.where(has_v, (1 - ens_w), 1.0) * has_a; wv = np.where(has_a, ens_w * spec, 1.0) * has_v
    tot = wa + wv; Wt = (a * wa[:, None] + v * wv[:, None]) / np.where(tot > 0, tot, 1)[:, None]
    return cols, z["perts"].astype(str), Wt

def neff(w): w = np.asarray(w, float); w = w / w.sum(); return float(1 / (w ** 2).sum())

def main():
    res = json.load(open(f"{ROOT}/results/score_fm.json")); V0, _ = r0_vectors()
    reps = {"R0": V0, "R1": emb_vectors("R1"), "R2": emb_vectors("R2"), "R3": emb_vectors("R3")}
    lrj_genes = set(np.load(LRJ_RESP)["genes"].astype(str)); out = {"weights": {}, "scores": {}}
    log = []; put = lambda s: (print(s, flush=True), log.append(s))
    for n in FOLD:
        c = CTX[n]; f = fold(n); M = source_mats(f); cov = f["cov"]; fr = res["folds"][n]; perts = f["perts_all"]
        prim = np.all([np.isfinite(M[s]).any(0) for s in SOURCES], 0)
        cv = np.load(CVAE[n])["pred"].astype(np.float64)
        sec = np.isin(f["genes"][f["cols"]], list(lrj_genes)) & np.isfinite(cv[cov]).any(0); assert sec.sum() == 451
        # ---- effective source weights, averaged over the covered knockdowns
        has = np.stack([np.isfinite(M[s]).any(1) for s in SOURCES], 1)[cov]
        W = {}
        for a in ["U"] + list(reps):
            if a == "U": g = np.ones(len(SOURCES))
            else:
                w = softmax_w(centred_cos(reps[a]), n, SOURCES, fr["T"][a]); g = np.array([w[s] for s in SOURCES])
            k = has * g[None]; k = k / k.sum(1, keepdims=True)
            W[a] = dict(sources=SOURCES, mean_weight=k.mean(0).tolist(), neff_of_mean=neff(k.mean(0)),
                        mean_per_kd_neff=float(np.mean(1 / (k ** 2).sum(1))))
        for name, Wd, cw, ew, sp in [("ens (nominal)", W0, CD4_W, ENS_W, 1.0), ("ens (v7 x1.5 amplitude)", W0, CD4_W, ENS_W, SPEC),
                                     ("ens_nocd4", W0, 0.0, ENS_W, SPEC), ("atlas (no v7 side)", W0, CD4_W, 0.0, SPEC),
                                     ("eqw_ens", {k: 1.0 for k in W0}, 1.0, ENS_W, SPEC)] + \
                [(f"ens_shrink{int(s * 100):03d}", {k: (1 - s) * W0[k] + s * 1.5 for k in W0}, CD4_W, ENS_W, SPEC) for s in SHRINK]:
            cols, pp, Wt = nominal_weights(c, Wd, cw, ew, sp); assert (pp == perts).all(); k = Wt[cov]
            W[name] = dict(sources=cols, mean_weight=k.mean(0).tolist(), neff_of_mean=neff(k.mean(0)),
                           mean_per_kd_neff=float(np.mean(1 / np.maximum((k ** 2).sum(1), 1e-12))))
        out["weights"][n] = W
        # ---- predictions
        P = {"U": predict(M, SOURCES, {s: 1.0 for s in SOURCES})}
        for a, V in reps.items(): P[a] = predict(M, SOURCES, softmax_w(centred_cos(V), n, SOURCES, fr["T"][a]))
        for k, p in ens_files(c).items():
            z = np.load(p); assert (z["perts"].astype(str) == perts).all(); P[k] = z["pred"].astype(np.float64)
        tgt = target_mask(list(perts[cov]), f["genes"], f["cols"])
        out["scores"][n] = {}
        for gs, mask in (("primary", prim), ("secondary_451", sec)):
            st = {k: rho_stats(v[cov][:, mask], f["L"][:, mask], tgt[:, mask]) for k, v in P.items()}
            rho = {k: float(rho_from(v)) for k, v in st.items()}
            ref = fr["gene_sets"]["primary" if gs == "primary" else "secondary"]["rho"]
            for k in ["U", "R0", "R1", "R2", "R3", "ensemble"]:
                kk = "ens" if k == "ensemble" else k
                assert abs(rho[kk] - ref[k]) < 1e-12, f"{n} {gs} {k} does not reproduce Tier 1"
            ens_v = [k for k in P if k.startswith(("ens", "atlas", "eqw"))]
            pairs = [(k, "ens") for k in ens_v if k != "ens"] + [(k, "U") for k in ens_v + ["R0", "R1", "R2", "R3"]]
            rng = np.random.default_rng(SEED); bt = {p: np.empty(N_BOOT) for p in pairs}
            for b in range(N_BOOT):
                ix = rng.integers(0, int(cov.sum()), int(cov.sum()))
                for a, b2 in pairs: bt[(a, b2)][b] = rho_from(st[a][ix]) - rho_from(st[b2][ix])
            out["scores"][n][gs] = {"rho": rho, "diff": {f"{a} - {b2}": dict(point=rho[a] - rho[b2], lo=float(np.percentile(bt[(a, b2)], 2.5)),
                                    hi=float(np.percentile(bt[(a, b2)], 97.5))) for a, b2 in pairs}}
            put(f"== {n} [{gs}: {int(mask.sum())} genes] " + "  ".join(f"{k} {v:.4f}" for k, v in rho.items()))
            for k, d in out["scores"][n][gs]["diff"].items(): put(f"   {k:28s} {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
        for a, w in W.items(): put(f"   weights {n} {a:26s} neff {w['neff_of_mean']:.2f} (per-kd {w['mean_per_kd_neff']:.2f}) " +
                                   " ".join(f"{s}={x:.3f}" for s, x in zip(w["sources"], w["mean_weight"])))
    json.dump(out, open(f"{OUT}/probe_scores.json", "w"), indent=1)
    open(f"{OUT}/probe_scores.txt", "w").write("\n".join(log) + "\n")

if __name__ == "__main__":
    main()
