"""Tier 1 scoring, as pre-registered: PLAN.md (sha256 1bed50d8...a27, Amendment 1) and AMENDMENTS.md (Amendment 5).
Prediction for held-out line h = softmax(centred cosine / T)-weighted average of the five sources' per-target responses;
T by inner leave-one-source-out on the sources only. Gate metric and derangement copied verbatim from head_to_head.py.
Writes results/score_fm.json and results/score_fm.txt. Usage: .venv_tools/bin/python fm/score.py
"""
import os, sys, json, time, numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES
SOURCES, TS = LINES[:5], [0.01, 0.03, 0.1, 0.3, 1]
EXT = "/Volumes/VCC26/scratch/vcc/ext"
FOLD = {"Jurkat": f"{EXT}/fold_jurkat", "HepG2": f"{EXT}/fold_hepg2"}
CVAE = {"Jurkat": "/Volumes/VCC26/cv_model/gate/pred_cvae_jurkat.npz", "HepG2": "/Volumes/VCC26/cv_model/gate/pred_cvae_hepg2.npz"}
LRJ_RESP = os.path.expanduser("~/scratch/linresp/data/responses.npz")
H2H_ZIP, H2H = os.path.expanduser("~/scratch/linresp/lrj-perturb.zip"), "lrj-perturb/results/h2h/head_to_head.json"
PARITY = {"Jurkat": 0.125, "HepG2": 0.122}
N_BOOT, SEED = 2000, 20261014
OUT = f"{ROOT}/results"

# ---- gate_oct14.py functions, verbatim (via head_to_head.py)
def target_mask(perts, genes, cols):
    gi = {g: j for j, g in enumerate(genes[cols])}
    M = np.zeros((len(perts), len(cols)), bool)
    for i, p in enumerate(perts):
        if p in gi: M[i, gi[p]] = True
    return M
def specific(X, tgt):
    X = np.where(tgt | ~np.isfinite(X), np.nan, X)
    return X - np.nanmean(X, axis=0, keepdims=True)
def rho_stats(P, R, tgt):
    Ps, Rs = specific(P, tgt), specific(R, tgt)
    ok = np.isfinite(Ps) & np.isfinite(Rs)
    rx = np.full(Ps.shape, np.nan); ry = np.full(Rs.shape, np.nan)
    rx[ok] = rankdata(Ps[ok]); ry[ok] = rankdata(Rs[ok])
    rx = np.where(ok, rx, 0.0); ry = np.where(ok, ry, 0.0)
    return np.stack([ok.sum(1), rx.sum(1), ry.sum(1), (rx * rx).sum(1), (ry * ry).sum(1), (rx * ry).sum(1)], 1)
def rho_from(st):
    n, sx, sy, sxx, syy, sxy = st.sum(0)
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
def shuffle_perm(n, seed=20261008):
    o = np.random.default_rng(seed).permutation(n); perm = np.empty(n, dtype=int); perm[o] = np.roll(o, -1); return perm

# ---- representations: one vector per line
def r0_vectors():
    """Amendment 1.2 + 5: log1p of mean per-cell CPM (on the 6,360-gene panel) of the same 2,000 cells; top-2,000 variance genes over the five sources."""
    import anndata as ad
    a = ad.read_h5ad(f"{ROOT}/h5ad/all_lines.h5ad"); X = a.X.tocsr().astype(np.float64)
    tot = np.asarray(X.sum(1)).ravel(); lab = a.obs["line"].values.astype(str)
    X = X.multiply(1e6 / tot[:, None]).tocsr()
    prof = {l: np.log1p(np.asarray(X[lab == l].mean(0)).ravel()) for l in LINES}
    var = np.var(np.stack([prof[s] for s in SOURCES]), axis=0); top = np.sort(np.argsort(-var, kind="stable")[:2000])
    return {l: prof[l][top] for l in LINES}, a.var_names.values[top].astype(str)

def emb_vectors(arm):
    z = np.load(f"{ROOT}/emb/emb_{arm}.npz"); return {l: z[l].astype(np.float64) for l in LINES}

def centred_cos(V):
    c = np.mean([V[s] for s in SOURCES], 0); C = {l: V[l] - c for l in LINES}
    return lambda a, b: float(C[a] @ C[b] / np.linalg.norm(C[a]) / np.linalg.norm(C[b]))

# ---- data
def fold(name):
    d = FOLD[name]; tz = np.load(f"{d}/truth.npz"); cov = np.load(f"{d}/covered.npy")
    return dict(perts_all=tz["perts"].astype(str), cov=cov, L=tz["L"][cov].astype(np.float64),
                cols=tz["cols"], genes=tz["genes"].astype(str))

def source_mats(f):
    """Per source: (fold knockdowns x fold genes) responses, NaN where the source lacks the target or the gene."""
    M = {}
    for s in SOURCES:
        z = np.load(f"{ROOT}/data/src_{s}.npz"); gi = {g: j for j, g in enumerate(z["genes"])}; ti = {t: i for i, t in enumerate(z["targets"])}
        col = np.array([gi[g] for g in f["genes"][f["cols"]]])
        A = np.full((len(f["perts_all"]), len(col)), np.nan)
        rows = np.array([ti.get(p, -1) for p in f["perts_all"]]); ok = rows >= 0
        A[ok] = z["L"][rows[ok]][:, col]; M[s] = A
    return M

def predict(M, srcs, w):
    """Weighted mean over sources with a finite value, weights renormalised per entry (Amendment 1.4)."""
    num = np.zeros_like(M[srcs[0]]); den = np.zeros_like(num)
    for s in srcs:
        fin = np.isfinite(M[s]); num += np.where(fin, M[s], 0.0) * w[s]; den += fin * w[s]
    with np.errstate(invalid="ignore", divide="ignore"): return num / den
def softmax_w(cos, h, srcs, T):
    v = np.array([cos(h, s) for s in srcs]) / T; e = np.exp(v - v.max()); return dict(zip(srcs, e / e.sum()))

def choose_T(cos, M, f, prim, log):
    """Inner leave-one-source-out on the fold's covered knockdowns and primary genes; Jurkat/HepG2 never enter."""
    perts = f["perts_all"]; cov = f["cov"]; score = {}
    cache = {}
    for s in SOURCES:
        others = [o for o in SOURCES if o != s]
        rows = cov & np.isfinite(M[s][:, prim]).all(1) & np.any([np.isfinite(M[o]).any(1) for o in others], 0)
        cache[s] = (rows, others, target_mask(list(perts[rows]), f["genes"], f["cols"][prim]))
    for T in TS:
        r = []
        for s in SOURCES:
            rows, others, tgt = cache[s]
            P = predict({o: M[o][rows][:, prim] for o in others}, others, softmax_w(cos, s, others, T))
            r.append(rho_from(rho_stats(P, M[s][rows][:, prim], tgt)))
        score[T] = dict(mean=float(np.mean(r)), per_source=dict(zip(SOURCES, map(float, r))),
                        knockdowns=dict((s, int(cache[s][0].sum())) for s in SOURCES))
    best = max(TS, key=lambda T: (round(score[T]["mean"], 12), -T))
    log(f"   T search: " + "  ".join(f"T={T}: {score[T]['mean']:+.4f}" for T in TS) + f"  -> T={best}")
    return best, score

def main():
    os.makedirs(OUT, exist_ok=True); lines_out = []
    log = lambda s: (print(s, flush=True), lines_out.append(s))
    t0 = time.time(); F = {n: fold(n) for n in FOLD}
    for n, f in F.items():                                                        # parity with the gate
        tgt = target_mask(list(f["perts_all"][f["cov"]]), f["genes"], f["cols"])
        r = rho_from(rho_stats(np.load(f"{FOLD[n]}/pred_ens.npz")["pred"][f["cov"]].astype(np.float64), f["L"], tgt))
        log(f"parity {n}: ensemble rho {r:.5f} (target {PARITY[n]}) {'OK' if round(r, 3) == PARITY[n] else 'FAIL'}")
        if round(r, 3) != PARITY[n]: sys.exit("parity failed")
    V0, r0_genes = r0_vectors()
    reps = {"R0": V0, "R1": emb_vectors("R1"), "R2": emb_vectors("R2"), "R3": emb_vectors("R3")}
    for extra in ("R1_seed2", "R3_seed2"):
        if os.path.exists(f"{ROOT}/emb/emb_{extra}.npz"): reps[extra] = emb_vectors(extra)
    log(f"arms: {list(reps)} + U; R0 on {len(r0_genes)} genes")
    lrj_genes = set(np.load(LRJ_RESP)["genes"].astype(str)); import zipfile; h2h = json.loads(zipfile.ZipFile(H2H_ZIP).read(H2H))
    res = {"arms": list(reps) + ["U"], "folds": {}}
    for n, f in F.items():
        perts_all, cov, cols, genes = f["perts_all"], f["cov"], f["cols"], f["genes"]
        M = source_mats(f)
        prim = np.all([np.isfinite(M[s]).any(0) for s in SOURCES], 0)
        cv = np.load(CVAE[n])["pred"].astype(np.float64)
        sec = np.isin(genes[cols], list(lrj_genes)) & np.isfinite(cv[cov]).any(0)
        log(f"== {n}: {cov.sum()} knockdowns, {len(cols)} fold genes, primary {prim.sum()}, secondary {sec.sum()}")
        ens = np.load(f"{FOLD[n]}/pred_ens.npz")["pred"].astype(np.float64)
        tgt_all = target_mask(list(perts_all[cov]), genes, cols)
        r_ens_sec = rho_from(rho_stats(ens[cov][:, sec], f["L"][:, sec], tgt_all[:, sec]))
        ref = h2h[n]["common"]["rho"]["ensemble"]; sec_ok = sec.sum() == 451 and abs(r_ens_sec - ref) < 1e-6
        log(f"   secondary parity: {sec.sum()} genes, ensemble rho {r_ens_sec:.4f} (target 451, {ref:.6f}) {'OK' if sec_ok else 'FAIL: secondary set dropped'}")
        out = {"knockdowns": int(cov.sum()), "genes_primary": int(prim.sum()), "genes_secondary": int(sec.sum()),
               "secondary_ok": bool(sec_ok), "T": {}, "T_search": {}, "weights": {}, "gene_sets": {}}
        P = {"U": predict(M, SOURCES, {s: 1.0 for s in SOURCES})}
        for arm, V in reps.items():
            cos = centred_cos(V); log(f" {arm}")
            T, sc = choose_T(cos, M, f, prim, log); out["T"][arm] = T; out["T_search"][arm] = sc
            w = softmax_w(cos, n, SOURCES, T); out["weights"][arm] = {s: float(v) for s, v in w.items()}
            out.setdefault("cos", {})[arm] = {s: cos(n, s) for s in SOURCES}
            P[arm] = predict(M, SOURCES, w)
        perm = shuffle_perm(len(perts_all))
        for k in list(P): P[k + "_der"] = P[k][perm]
        P["ensemble"] = ens
        fm = [a for a in ("R1", "R2", "R3")]
        pairs = [(a, a + "_der") for a in ["R0", "U", "R1", "R2", "R3"]] + [(a, "R0") for a in fm] + [(a, "U") for a in fm] + [("R0", "U")]
        pairs += [(a.split("_")[0], a) for a in reps if a.endswith("seed2")] + [(a, a + "_der") for a in reps if a.endswith("seed2")]
        gsets = [("primary", prim)] + ([("secondary", sec)] if sec_ok else [])
        for gs, mask in gsets:
            pp = pairs + ([(a, "ensemble") for a in fm] if gs == "secondary" else [])
            st = {k: rho_stats(v[cov][:, mask], f["L"][:, mask], tgt_all[:, mask]) for k, v in P.items()}
            rng = np.random.default_rng(SEED); bt = {p: np.empty(N_BOOT) for p in pp}
            for b in range(N_BOOT):
                ix = rng.integers(0, int(cov.sum()), int(cov.sum()))
                for a, c in pp: bt[(a, c)][b] = rho_from(st[a][ix]) - rho_from(st[c][ix])
            g = {"rho": {k: float(rho_from(v)) for k, v in st.items()}, "diff": {}}
            for a, c in pp:
                d = bt[(a, c)]
                g["diff"][f"{a} - {c}"] = dict(point=float(rho_from(st[a]) - rho_from(st[c])),
                                               lo95=float(np.percentile(d, 2.5)), hi95=float(np.percentile(d, 97.5)),
                                               lo983=float(np.percentile(d, 100 / 120)), hi983=float(np.percentile(d, 100 - 100 / 120)))
            out["gene_sets"][gs] = g
            log(f"   [{gs}: {int(mask.sum())} genes] " + "  ".join(f"{k} {v:+.4f}" for k, v in g["rho"].items()))
            for k, v in g["diff"].items(): log(f"     {k:22s} {v['point']:+.4f}  95% [{v['lo95']:+.4f}, {v['hi95']:+.4f}]  98.3% [{v['lo983']:+.4f}, {v['hi983']:+.4f}]")
        if sec_ok: out["lrj_quoted"] = {"rho_secondary": h2h[n]["common"]["rho"]["lrj"]}
        res["folds"][n] = out
    # decisions (Amendment 5)
    D = {}
    for gs in ("primary",):
        valid = {a: {n: res["folds"][n]["gene_sets"][gs]["diff"][f"{a} - {a}_der"]["lo95"] > 0 for n in FOLD} for a in ["R0", "U", "R1", "R2", "R3"]}
        D["valid"] = valid
        for H, ref in (("H1", "R0"), ("H2", "U")):
            for lvl in ("95", "983"):
                for a in ("R1", "R2", "R3"):
                    up = [res["folds"][n]["gene_sets"][gs]["diff"][f"{a} - {ref}"][f"lo{lvl}"] > 0 and valid[a][n] for n in FOLD]
                    D[f"{H}_{lvl}_{a}"] = "helps" if all(up) else ("inconclusive" if any(up) else "does not help")
    res["decisions"] = D
    log("decisions: " + json.dumps(D))
    log(f"done in {time.time() - t0:.0f} s")
    json.dump(res, open(f"{OUT}/score_fm.json", "w"), indent=1)
    open(f"{OUT}/score_fm.txt", "w").write("\n".join(lines_out) + "\n")

if __name__ == "__main__":
    main()
