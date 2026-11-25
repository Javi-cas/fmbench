"""EXPLORATORY (2026-10-26; see results/r1_loo/README.md). Leave-one-knockdown-out decomposition of each FM arm's margin
over U on the primary gene set, with the gate's metric (verbatim, via fm/score.py). Leave-one-out drops one knockdown's row
of the gate's per-knockdown rank statistics (ranks fixed once on the full set, exactly as the gate's bootstrap treats
resamples); a full recompute (re-centring and re-ranking without k) is checked on a sample. Trimmed-margin CIs use the
Tier 1 bootstrap resamples (seed 20261014), with removed knockdowns dropped from each resample. Writes results/r1_loo/."""
import os, sys, json, numpy as np
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import (FOLD, SOURCES, N_BOOT, SEED, ROOT, fold, source_mats, predict, softmax_w, centred_cos, r0_vectors,
                   emb_vectors, target_mask, rho_stats, rho_from)
OUT = f"{ROOT}/results/r1_loo"; os.makedirs(OUT, exist_ok=True)
TRIMS = (0.01, 0.05, 0.10)

def rho_sums(S):
    n, sx, sy, sxx, syy, sxy = S.T if S.ndim == 2 else S
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))

def gini(x):
    x = np.sort(np.abs(x)); n = len(x); return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))

def main():
    res = json.load(open(f"{ROOT}/results/score_fm.json")); V0, _ = r0_vectors()
    reps = {"R1": emb_vectors("R1"), "R2": emb_vectors("R2"), "R3": emb_vectors("R3")}
    out = {}; log = []; put = lambda s: (print(s, flush=True), log.append(s))
    for n in FOLD:
        f = fold(n); M = source_mats(f); cov = f["cov"]; fr = res["folds"][n]; perts = f["perts_all"][cov]
        prim = np.all([np.isfinite(M[s]).any(0) for s in SOURCES], 0)
        P = {"U": predict(M, SOURCES, {s: 1.0 for s in SOURCES})}
        for a, V in reps.items(): P[a] = predict(M, SOURCES, softmax_w(centred_cos(V), n, SOURCES, fr["T"][a]))
        tgt = target_mask(list(perts), f["genes"], f["cols"])[:, prim]; L = f["L"][:, prim]
        st = {k: rho_stats(v[cov][:, prim], L, tgt) for k, v in P.items()}
        for k in st: assert abs(rho_from(st[k]) - fr["gene_sets"]["primary"]["rho"][k]) < 1e-12, f"{n} {k} does not reproduce Tier 1"
        N = len(perts); rng = np.random.default_rng(SEED); IX = [rng.integers(0, N, N) for _ in range(N_BOOT)]
        # K562-to-truth distance per knockdown (second question): 1 - Spearman(K562 response, truth), target gene masked
        K = M["K562"][cov][:, prim]; dk = np.full(N, np.nan)
        for i in range(N):
            ok = np.isfinite(K[i]) & ~tgt[i]
            if ok.sum() > 50 and np.isfinite(K[i][ok]).all() and np.ptp(K[i][ok]) > 0: dk[i] = 1 - spearmanr(K[i][ok], L[i][ok])[0]
        out[n] = {}
        for a in ("R1", "R2", "R3"):
            SA, SU = st[a].sum(0), st["U"].sum(0); m0 = rho_sums(SA) - rho_sums(SU)
            loo = (rho_sums(SA[None] - st[a])) - (rho_sums(SU[None] - st["U"]))
            c = m0 - loo                                                     # contribution(k)
            order = np.argsort(-c, kind="stable")
            def margin_without(drop):
                keep = np.ones(N, bool); keep[drop] = False
                pt = rho_from(st[a][keep]) - rho_from(st["U"][keep]); bt = []
                for ix in IX:
                    ix = ix[keep[ix]]; bt.append(rho_from(st[a][ix]) - rho_from(st["U"][ix]))
                return float(pt), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
            trims = {}
            for q in TRIMS:
                m = int(round(q * N)); trims[f"top{int(q * 100)}"] = dict(n_removed=m, margin=margin_without(order[:m]))
            r5 = np.random.default_rng(20261026).choice(N, int(round(0.05 * N)), replace=False)
            trims["random5"] = dict(n_removed=len(r5), margin=margin_without(r5))
            keep = np.ones(N, bool); half = None                             # smallest m with trimmed margin <= half
            for m in range(1, N + 1):
                keep[order[m - 1]] = False
                if rho_from(st[a][keep]) - rho_from(st["U"][keep]) <= 0.5 * m0: half = m; break
            cs = np.cumsum(c[order]); half_cum = int(np.searchsorted(cs, 0.5 * c.sum()) + 1) if c.sum() > 0 else None
            okd = np.isfinite(dk)
            o = dict(margin=float(m0), n=N, contrib_median=float(np.median(c)), contrib_iqr=[float(np.percentile(c, 25)), float(np.percentile(c, 75))],
                     positive_share=float((c > 0).mean()), sum_contrib=float(c.sum()), gini_abs=gini(c), trims=trims,
                     n_to_half_by_trimming=half, n_to_half_by_cumsum=half_cum,
                     top10=[(perts[i], float(c[i])) for i in order[:10]],
                     corr_contrib_vs_k562_distance=float(spearmanr(c[okd], dk[okd])[0]), n_with_k562=int(okd.sum()),
                     contrib_by_k562_distance_tertile=[float(np.mean(c[okd][(dk[okd] >= lo) & (dk[okd] <= hi)]))
                                                       for lo, hi in zip(np.nanpercentile(dk, [0, 33.33, 66.67]), np.nanpercentile(dk, [33.33, 66.67, 100]))])
            if a == "R1":                                                   # check: fixed-rank LOO vs full recompute on a sample
                chk = list(order[:5]) + list(np.random.default_rng(1).choice(N, 5, replace=False)); diffs = []
                for i in chk:
                    keep = np.ones(N, bool); keep[i] = False; t2 = target_mask(list(perts[keep]), f["genes"], f["cols"])[:, prim]
                    full = rho_from(rho_stats(P[a][cov][keep][:, prim], L[keep], t2)) - rho_from(rho_stats(P["U"][cov][keep][:, prim], L[keep], t2))
                    diffs.append((perts[i], float(m0 - full), float(c[i])))
                o["full_recompute_check"] = diffs
            out[n][a] = o
            t = trims
            put(f"== {n} {a} - U: margin {m0:+.5f}; contribution median {o['contrib_median']:+.2e}, IQR [{o['contrib_iqr'][0]:+.2e}, {o['contrib_iqr'][1]:+.2e}], "
                f"positive share {o['positive_share']:.3f}, Gini |c| {o['gini_abs']:.3f}, sum of contributions {o['sum_contrib']:+.5f}")
            for k, v in t.items(): put(f"   without {k:8s} ({v['n_removed']:3d} kd): margin {v['margin'][0]:+.5f} [{v['margin'][1]:+.5f}, {v['margin'][2]:+.5f}]  = {v['margin'][0] / m0:.2f} of original")
            put(f"   knockdowns to remove half the margin: {half} by trimming, {half_cum} by cumulative contribution")
            put(f"   corr(contribution, 1 - rho(K562, truth)) = {o['corr_contrib_vs_k562_distance']:+.3f} over {o['n_with_k562']} kd; mean contribution by distance tertile (near, mid, far): "
                + ", ".join(f"{x:+.2e}" for x in o["contrib_by_k562_distance_tertile"]))
            if a == "R1": put("   full-recompute check (kd, full, fixed-rank): " + "; ".join(f"{p} {x:+.2e}/{y:+.2e}" for p, x, y in o["full_recompute_check"]))
            put("   top 10: " + ", ".join(f"{p} {v:+.1e}" for p, v in o["top10"]))
    json.dump(out, open(f"{OUT}/r1_loo.json", "w"), indent=1); open(f"{OUT}/r1_loo.txt", "w").write("\n".join(log) + "\n")

if __name__ == "__main__":
    main()
