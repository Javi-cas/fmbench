"""EXPLORATORY (2026-10-26; see results/ensemble_probe/README.md). Rebuild the JaviCell fold ensemble in-process, read-only.
Mirrors javicell/scripts/bench/effect_fold.py predict() line for line, with the atlas-side source weights as a parameter,
and writes ONLY to fmbench/results/ensemble_probe/. Nothing in ~/scratch/vcc is written (no predict() call, no context
file, run with `python -B` so no bytecode is written; JAVIBENCH_CTX points at the fold's existing jb/ directory).
Gate: the unmodified rebuild must equal the producer's saved pred_{ens,ens_nocd4,atlas,atlas_nocd4}.npz bit for bit, and
the equal-weight rebuild must equal fold_<ctx>_eqw/pred_ens.npz; otherwise no variant is written.
Usage: <javicell .venv python> -B fm/ensemble_probe_build.py jurkat|hepg2"""
import os, sys, json, numpy as np
from pathlib import Path
c = sys.argv[1]
VCC = Path("/Volumes/VCC26/scratch/vcc"); J = VCC / "javicell"
OUT = Path(__file__).resolve().parents[1] / "results" / "ensemble_probe"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(J / "scripts" / "bench"))
import effect_fold as ef                                     # constants and read-only helpers; nothing at import writes
d = ef.fold_dir(c); assert d.resolve() == (VCC / "ext" / f"fold_{c}").resolve()
os.environ["JAVIBENCH_CTX"] = str(d / "jb"); os.environ.setdefault("JAVIBENCH_SRC", str(ef.S / "cache"))
assert (d / "jb" / "context.json").exists()
for p in (ef.ROOT / "atlasshift_local" / "upstream", ef.ROOT / "atlasshift_local", ef.ROOT / "scripts"): sys.path.insert(0, str(p))
from model import load_xatlas, fuse_source_centered, aligned_cd4, add_cd4_family, desired_mean
from extras import ensemble, javicell_moments
import javibench as jb
from vcp.localeval import Reference

tz = np.load(d / "truth.npz", allow_pickle=False)
perts, genes, cols = tz["perts"].astype(str), tz["genes"].astype(str), tz["cols"]
ref = Reference.load(d / "ref")
mean, bulk = ef.control_moments(c, genes)                   # reads the existing ctrl_moments.npz
assert (d / "ctrl_moments.npz").exists()
names = [n for n in ef.W if ef.SRC_FILES[n].exists()]
srcs = [load_xatlas(ef.SRC_FILES[n], prior_counts=100000, matched=True) for n in names]
cd4, avail = aligned_cd4(ef.CD4_FILE, list(perts), genes, common_subtract=1, center_scope="source")
E, _ = jb.build_effects("v7", list(perts), genes, ef.ref_profile(ref, c, genes))
jv = javicell_moments(E, mean, bulk, ef.AMP_CPM, ef.AMP_BULK)

def build(Wd, cd4_w):
    base = {sp_: fuse_source_centered(srcs, [Wd[n] for n in names], list(perts), genes, space=sp_, common_subtract=1)
            for sp_ in ("log2fc", "bulk_delta")}
    def atlas(use_cd4):
        out = []
        for space, control, amp in (("log2fc", mean, ef.AMP_CPM), ("bulk_delta", bulk, ef.AMP_BULK)):
            eff, cov = base[space]
            if use_cd4:
                eff, _ = add_cd4_family(eff, cov, cd4, avail, cd4_w, space=space, control_probability=control)
            out.append(desired_mean(control, eff, space=space, amplitude=amp, clip=3))
        return out
    to_l2 = lambda des: np.log2((1e6 * des[0][:, cols] + 1) / (ref.ctrl_mean[None, :].astype(np.float64) + 1)).astype(np.float32)
    a1, a0 = atlas(True), atlas(False)
    return {"atlas": to_l2(a1), "atlas_nocd4": to_l2(a0), "ens": to_l2(ensemble(a1, jv, ef.ENS_W)),
            "ens_nocd4": to_l2(ensemble(a0, jv, ef.ENS_W))}

# gate 1: unmodified recipe reproduces the producer's files exactly
V0 = build(dict(ef.W), ef.CD4_W); ok = {}
for k, v in V0.items():
    z = np.load(d / f"pred_{k}.npz"); ok[k] = bool((z["perts"].astype(str) == perts).all() and np.array_equal(z["pred"], v))
# gate 2: the parameterisation reproduces the producer's equal-weight build (W = 1, CD4 weight 1)
Ve = build({n: 1.0 for n in ef.W}, 1.0); ze = np.load(VCC / "ext" / f"fold_{c}_eqw" / "pred_ens.npz")
ok["eqw_ens"] = bool(np.array_equal(ze["pred"], Ve["ens"]))
print(c, "bit-identity:", ok, flush=True)
json.dump(dict(fold=c, bit_identical=ok, sources=names, W=ef.W, CD4_W=ef.CD4_W, ENS_W=ef.ENS_W, AMP_CPM=ef.AMP_CPM,
               AMP_BULK=ef.AMP_BULK), open(OUT / f"build_{c}.json", "w"), indent=1)
if not all(ok.values()):
    sys.exit(f"{c}: rebuild is not bit-identical; no shrinkage variant written")

# shrinkage of the atlas-side source weights toward uniform at fixed total (sum 6, uniform 1.5 each);
# CD4 weight, v7 blend and every other setting unchanged
tot = sum(ef.W[n] for n in names); u = tot / len(names)
for s in (0.25, 0.5, 0.75, 1.0):
    Ws = {n: (1 - s) * ef.W[n] + s * u for n in names}
    v = build(Ws, ef.CD4_W)["ens"]
    np.savez(OUT / f"pred_{c}_ens_shrink{int(s * 100):03d}.npz", perts=perts, pred=v, W=json.dumps(Ws))
    print(c, "shrink", s, {n: round(w, 3) for n, w in Ws.items()}, flush=True)

# nominal per-knockdown weights from the code (log2fc space, before amplitude, clip and exp): atlas side
# W_n * present / (sum W present + CD4_W * cd4 available), times (1 - ENS_W); v7 side ENS_W split by its own source weights
lookup = {n: set(s.targets[np.asarray(s.n_cells) >= 20]) if hasattr(s, "n_cells") else set(s.targets) for n, s in zip(names, srcs)}
pres = np.array([[p in lookup[n] for n in names] for p in perts], float)
cd4a = np.asarray(avail).any(1).astype(float)
v7src = {k: jb.source(k) for k in jb.VARIANTS["v7"]["sources"]}
v7idx = {k: (v7src[k].index() if v7src[k] is not None else {}) for k in v7src}
v7pres = np.array([[p in v7idx[k] for k in v7src] for p in perts], float)
np.savez(OUT / f"presence_{c}.npz", perts=perts, names=np.array(names), pres=pres, cd4=cd4a,
         v7names=np.array(list(v7src)), v7pres=v7pres)
print(c, "presence saved; v7 sources present per knockdown:", v7pres.sum(0).tolist(), flush=True)
