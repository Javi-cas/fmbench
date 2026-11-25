"""EXPLORATORY (2026-10-26; see results/r1_loo/README.md). Exact additive per-knockdown decomposition of an arm's margin
over U. The gate's pooled Spearman is rho = sum_k a_k / D, with a_k = sum_g (rx - mean rx)(ry - mean ry) over knockdown
k's genes (global mean ranks, ranks fixed once on the full set) and D = sqrt(Sxx Syy); so d_k = a_k(arm)/D(arm) - a_k(U)/D(U)
sums exactly to the margin. Leave-one-out contributions (fm/r1_loo.py) sum to about zero by construction (jackknife),
so they cannot say which knockdowns carry the margin; this can. Symmetric-noise reference: the same trims applied to the
d_k distribution mirrored about its mean (2 mean - d_k pooled with d_k). Writes results/r1_loo/r1_additive.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import (FOLD, SOURCES, N_BOOT, SEED, ROOT, fold, source_mats, predict, softmax_w, centred_cos, emb_vectors,
                   target_mask, rho_stats, rho_from)
from r1_loo import gini, TRIMS
OUT = f"{ROOT}/results/r1_loo"

def additive(st):
    n, sx, sy, sxx, syy, sxy = st.T; N = n.sum(); mx, my = sx.sum() / N, sy.sum() / N
    a = sxy - my * sx - mx * sy + n * mx * my
    D = np.sqrt((sxx.sum() - N * mx * mx) * (syy.sum() - N * my * my)); return a / D

def trim_frac(d, q):
    m = int(round(q * len(d))); s = np.sort(d)[::-1]; return float((d.sum() - s[:m].sum()) / d.sum())

def main():
    res = json.load(open(f"{ROOT}/results/score_fm.json")); reps = {a: emb_vectors(a) for a in ("R1", "R2", "R3")}
    out = {}; log = []; put = lambda s: (print(s, flush=True), log.append(s))
    for n in FOLD:
        f = fold(n); M = source_mats(f); cov = f["cov"]; fr = res["folds"][n]; perts = f["perts_all"][cov]
        prim = np.all([np.isfinite(M[s]).any(0) for s in SOURCES], 0)
        P = {"U": predict(M, SOURCES, {s: 1.0 for s in SOURCES})}
        for a, V in reps.items(): P[a] = predict(M, SOURCES, softmax_w(centred_cos(V), n, SOURCES, fr["T"][a]))
        tgt = target_mask(list(perts), f["genes"], f["cols"])[:, prim]
        st = {k: rho_stats(v[cov][:, prim], f["L"][:, prim], tgt) for k, v in P.items()}
        N = len(perts); rng = np.random.default_rng(SEED); IX = [rng.integers(0, N, N) for _ in range(N_BOOT)]
        out[n] = {}
        for a in ("R1", "R2", "R3"):
            d = additive(st[a]) - additive(st["U"]); m0 = rho_from(st[a]) - rho_from(st["U"])
            assert abs(d.sum() - m0) < 1e-12, "additive decomposition does not sum to the margin"
            order = np.argsort(-d, kind="stable"); mir = np.concatenate([d, 2 * d.mean() - d])
            o = dict(margin=float(m0), median=float(np.median(d)), iqr=[float(np.percentile(d, 25)), float(np.percentile(d, 75))],
                     mean=float(d.mean()), positive_share=float((d > 0).mean()), gini_abs=gini(d), trims={})
            for q in TRIMS:
                m = int(round(q * N)); keep = np.ones(N, bool); keep[order[:m]] = False
                pt = rho_from(st[a][keep]) - rho_from(st["U"][keep]); bt = []
                for ix in IX:
                    ix = ix[keep[ix]]; bt.append(rho_from(st[a][ix]) - rho_from(st["U"][ix]))
                o["trims"][f"top{int(q * 100)}"] = dict(n_removed=m, margin=float(pt), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                                                        frac_recomputed=float(pt / m0), frac_additive=trim_frac(d, q), frac_symmetric_reference=trim_frac(mir, q))
            keep = np.ones(N, bool); half = None
            for m in range(1, N + 1):
                keep[order[m - 1]] = False
                if rho_from(st[a][keep]) - rho_from(st["U"][keep]) <= 0.5 * m0: half = m; break
            cs = np.cumsum(d[order]); o["n_to_half_by_trimming"] = half
            o["n_to_half_by_cumsum"] = int(np.searchsorted(cs, 0.5 * m0) + 1) if m0 > 0 else None
            o["top10"] = [(perts[i], float(d[i])) for i in order[:10]]
            out[n][a] = o
            put(f"== {n} {a} - U: margin {m0:+.5f} = sum of d_k; d_k median {o['median']:+.2e}, IQR [{o['iqr'][0]:+.2e}, {o['iqr'][1]:+.2e}], "
                f"mean {o['mean']:+.2e}, positive share {o['positive_share']:.3f}, Gini |d| {o['gini_abs']:.3f}")
            for k, v in o["trims"].items():
                put(f"   without {k:6s} ({v['n_removed']:3d} kd): margin {v['margin']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}] = {v['frac_recomputed']:.2f} of original "
                    f"(additive {v['frac_additive']:.2f}; symmetric-noise reference {v['frac_symmetric_reference']:.2f})")
            put(f"   knockdowns to halve the margin: {half} by trimming, {o['n_to_half_by_cumsum']} by cumulative d_k (largest first)")
            put("   top 10: " + ", ".join(f"{p} {v:+.1e}" for p, v in o["top10"]))
    json.dump(out, open(f"{OUT}/r1_additive.json", "w"), indent=1); open(f"{OUT}/r1_additive.txt", "w").write("\n".join(log) + "\n")

if __name__ == "__main__":
    main()
