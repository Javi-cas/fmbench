"""R1 seed 2 under the H1/H2 rules (report only; Tier 1 results are not modified). Rebuilds U, R0, R1 and R1 seed 2
predictions with fm/score.py's own functions and the T values stored in results/score_fm.json, checks every rho against
that file, and draws the same 2,000 bootstrap resamples (seed 20261014). Writes results/r1_seed2_verdicts.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import (FOLD, SOURCES, N_BOOT, SEED, ROOT, fold, source_mats, predict, softmax_w, centred_cos,
                   r0_vectors, emb_vectors, target_mask, rho_stats, rho_from)
res = json.load(open(f"{ROOT}/results/score_fm.json")); V0, _ = r0_vectors()
reps = {"R0": V0, "R1": emb_vectors("R1"), "R1_seed2": emb_vectors("R1_seed2")}
out = {}
for n in FOLD:
    f = fold(n); M = source_mats(f); cov = f["cov"]; fr = res["folds"][n]
    prim = np.all([np.isfinite(M[s]).any(0) for s in SOURCES], 0); assert prim.sum() == fr["genes_primary"]
    P = {"U": predict(M, SOURCES, {s: 1.0 for s in SOURCES})}
    for a, V in reps.items(): P[a] = predict(M, SOURCES, softmax_w(centred_cos(V), n, SOURCES, fr["T"][a]))
    tgt = target_mask(list(f["perts_all"][cov]), f["genes"], f["cols"])[:, prim]
    st = {k: rho_stats(v[cov][:, prim], f["L"][:, prim], tgt) for k, v in P.items()}
    rho = {k: float(rho_from(v)) for k, v in st.items()}
    for k, v in rho.items(): assert abs(v - fr["gene_sets"]["primary"]["rho"][k]) < 1e-12, f"{n} {k} does not reproduce"
    pairs = [("R1", "U"), ("R1_seed2", "U"), ("R1", "R0"), ("R1_seed2", "R0"), ("R1", "R1_seed2")]
    rng = np.random.default_rng(SEED); bt = {p: np.empty(N_BOOT) for p in pairs}
    for b in range(N_BOOT):
        ix = rng.integers(0, int(cov.sum()), int(cov.sum()))
        for a, c in pairs: bt[(a, c)][b] = rho_from(st[a][ix]) - rho_from(st[c][ix])
    out[n] = {"rho": rho, "diff": {f"{a} - {c}": dict(point=rho[a] - rho[c], lo95=float(np.percentile(bt[(a, c)], 2.5)),
              hi95=float(np.percentile(bt[(a, c)], 97.5)), lo983=float(np.percentile(bt[(a, c)], 100 / 120)),
              hi983=float(np.percentile(bt[(a, c)], 100 - 100 / 120))) for a, c in pairs}}
    for k, d in out[n]["diff"].items():
        assert k not in fr["gene_sets"]["primary"]["diff"] or abs(d["lo95"] - fr["gene_sets"]["primary"]["diff"][k]["lo95"]) < 1e-12, f"{k} CI differs"
json.dump(out, open(f"{ROOT}/results/r1_seed2_verdicts.json", "w"), indent=1)
for n, o in out.items():
    print(n, {k: round(v, 6) for k, v in o["rho"].items()})
    for k, d in o["diff"].items(): print(f"  {k:16s} {d['point']:+.6f}  95% [{d['lo95']:+.6f}, {d['hi95']:+.6f}]  98.3% [{d['lo983']:+.6f}, {d['hi983']:+.6f}]")
