"""Amendments 4 and 6: disagreement between two runs of one arm on the same cells (emb/cells_<arm>.npz, unseeded, vs
cells_<arm>_seed2.npz). Embedding-level only. Usage: r1_seed_disagreement.py [R1|R3] (default R1); writes emb/<arm>_seed_disagreement.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, LINES
SOURCES, HELD = LINES[:5], LINES[5:]
ARM = sys.argv[1] if len(sys.argv) > 1 else "R1"
c1, c2 = np.load(f"{ROOT}/emb/cells_{ARM}.npz"), np.load(f"{ROOT}/emb/cells_{ARM}_seed2.npz")
assert (c1["line"] == c2["line"]).all() and c1["E"].shape == c2["E"].shape, "runs do not cover the same cells in the same order"
E1, E2, lab = c1["E"].astype(np.float64), c2["E"].astype(np.float64), c1["line"]
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
rng = np.random.default_rng(20261024)
out = {"per_cell": {}, "per_line": {}, "similarity": {}}

# Per cell: same cell across seeds, against the spread between different cells of the same line (run 1).
pc = (unit(E1) * unit(E2)).sum(1)
for l in LINES:
    m = np.where(lab == l)[0]; i, j = rng.choice(m, 4000), rng.choice(m, 4000); k = i != j
    within = (unit(E1[i[k]]) * unit(E1[j[k]])).sum(1)
    out["per_cell"][l] = dict(same_cell_cos_median=float(np.median(pc[m])), same_cell_cos_p05=float(np.percentile(pc[m], 5)),
                              same_cell_cos_min=float(pc[m].min()), other_cell_same_line_cos_median=float(np.median(within)))

# Per line: mean vectors, raw and after the plan's centring (mean of the five source vectors; Amendment 1.3).
M1 = {l: E1[lab == l].mean(0) for l in LINES}; M2 = {l: E2[lab == l].mean(0) for l in LINES}
C1 = {l: M1[l] - np.mean([M1[s] for s in SOURCES], 0) for l in LINES}
C2 = {l: M2[l] - np.mean([M2[s] for s in SOURCES], 0) for l in LINES}
for l in LINES:
    out["per_line"][l] = dict(raw_mean_cos=cos(M1[l], M2[l]), centred_cos=cos(C1[l], C2[l]),
                              centred_rel_change=float(np.linalg.norm(C1[l] - C2[l]) / np.linalg.norm(C1[l])))

# The quantity the predictor uses: centred cosine of each line to each source. Seed change vs the spread across sources.
for h in LINES:
    srcs = [s for s in SOURCES if s != h]
    s1 = np.array([cos(C1[h], C1[s]) for s in srcs]); s2 = np.array([cos(C2[h], C2[s]) for s in srcs])
    out["similarity"][h] = dict(sources=srcs, run1=s1.round(4).tolist(), run2=s2.round(4).tolist(),
                                max_abs_change=float(np.abs(s1 - s2).max()), spread_across_sources_run1=float(s1.max() - s1.min()),
                                same_nearest_source=bool(np.argmax(s1) == np.argmax(s2)), same_rank_order=bool((np.argsort(s1) == np.argsort(s2)).all()))
json.dump(out, open(f"{ROOT}/emb/{ARM}_seed_disagreement.json", "w"), indent=1)
print(f"{'line':14s} cell cos med/p05 | other-cell cos | line cos raw/centred | centred rel change | sim max change / spread | same order")
for l in LINES:
    p, q, s = out["per_cell"][l], out["per_line"][l], out["similarity"][l]
    print(f"{l:14s} {p['same_cell_cos_median']:.4f}/{p['same_cell_cos_p05']:.4f} | {p['other_cell_same_line_cos_median']:.4f} | "
          f"{q['raw_mean_cos']:.6f}/{q['centred_cos']:.4f} | {q['centred_rel_change']:.3f} | "
          f"{s['max_abs_change']:.4f} / {s['spread_across_sources_run1']:.4f} | {s['same_rank_order']}")
