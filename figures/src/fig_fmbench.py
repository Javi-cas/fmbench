"""Figure: foundation-model benchmark (Tier 1) schematic."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/figs")
from figlib import *

W, H = 1950, 1060
s = SVG(W, H)
s.add('<g transform="translate(-40,0)">')
Y0, HC = 30, 1000
C = [(70, 430), (540, 460), (1040, 440), (1520, 440)]
for L, (x, w), (t, sub) in zip("abcd", C, [("Data", "CRISPRi Perturb-seq screens"), ("Representation", "One vector per cell line"),
                                 ("Prediction", "Borrow responses from similar lines"), ("Evaluation", "The challenge gate's own metric")]):
    column(s, x, Y0, w, HC, t, sub, letter=L)
for k in range(3):
    xa = C[k][0] + C[k][1]; line(s, xa + 6, 530, C[k + 1][0] - 6, 530, color=FAINT, sw=2.2)

# ---------------- column 1: data
x0 = 92
text(s, x0, 140, "Source lines: responses known", size=19, color=TEAL, weight=700)
text(s, 420, 172, "knockdowns × genes", size=14, color=MUTED, anchor="middle")
src = [("K562", "Replogle et al. 2022"), ("HCT116", "X-Atlas/Orion"), ("HEK293T", "X-Atlas/Orion"),
       ("H1", "Arc VCC 2025"), ("H1", "GSE295214")]
for i, (n, d) in enumerate(src):
    y = 222 + i * 72
    cells(s, 122, y, TEAL, TEAL_L, r=8.5, seed=i)
    text(s, 160, y - 11, n, size=19, weight=700); text(s, 160, y + 13, d, size=15, color=MUTED)
    heat(s, 380, y - 26, 5, 8, 10.4, seed=10 + i)
line(s, x0, 588, 478, 588, color=FRAME, sw=1.2, arrow=False, dash="5 5")
text(s, x0, 624, "Held-out line: test only", size=19, color=CORAL, weight=700)
y = 690
cells(s, 122, y, CORAL, CORAL_L, r=8.5, seed=9)
text(s, 160, y - 11, "Jurkat or HepG2", size=19, weight=700); text(s, 160, y + 13, "Nadig et al. 2025", size=15, color=MUTED)
heat(s, 380, y - 26, 5, 8, 10.4, hatch=True)
text(s, 421.6, y, "?", size=28, color=CORAL, weight=700, anchor="middle")
text(s, x0, 752, "The model sees its control cells only;", size=15, color=MUTED)
text(s, x0, 774, "measured responses are used for scoring.", size=15, color=MUTED)
for k, t in enumerate(["6,360 genes measured in all 7 lines", "2,000 control cells per line",
                       "Jurkat 1,811 · HepG2 1,059 knockdowns"]):
    chip(s, x0, 830 + k * 44, t, INK, PANEL, size=15)

# ---------------- column 2: representation
x0 = 562
for i in range(10):
    for j in range(10):
        v = (np.sin(i * 1.7 + j * 0.9) + 1) / 2
        s.add(f'<rect x="{574 + j*9}" y="{152 + i*9}" width="9" height="9" fill="{mix("#FFFFFF", MUTED, 0.15 + 0.6*v)}"/>')
rect(s, 574, 152, 90, 90, stroke=INK, sw=1.0, rx=0)
text(s, 619, 268, "2,000 cells", size=15, color=MUTED, anchor="middle"); text(s, 619, 288, "× 6,360 genes", size=15, color=MUTED, anchor="middle")
line(s, 674, 197, 704, 197)
transformer(s, 712, 172, 118, 58, VIOLET, VIOLET_L, "encoder")
text(s, 778, 268, "R0 to R3", size=15, color=MUTED, anchor="middle")
line(s, 852, 197, 878, 197)
for k in range(5): vec(s, 886 + k * 14, 160, 8, 9.6, VIOLET, seed=30 + k, vertical=True)
text(s, 924, 268, "cell vectors", size=15, color=MUTED, anchor="middle")
text(s, x0 + 10, 336, "average over cells", size=16, color=MUTED)
line(s, 712, 336, 790, 336)
vec(s, 798, 327, 12, 14, VIOLET, seed=41)
text(s, x0 + 10, 404, "Four representations, one predictor", size=18, weight=700)
arms = [("R0", "Raw control profile", "log1p CPM, 2,000 most variable genes", FAINT, PANEL),
        ("R1", "scGPT", "whole-human checkpoint", VIOLET, VIOLET_L),
        ("R2", "Geneformer V2-104M", "CLS-token cell embedding", VIOLET, VIOLET_L),
        ("R3", "State SE-600M", "Arc Institute embedding model", VIOLET, VIOLET_L)]
for i, (b, n, d, col, lt) in enumerate(arms):
    y = 456 + i * 84
    s.add(f'<circle cx="{x0 + 24}" cy="{y}" r="21" fill="{lt}" stroke="{col}" stroke-width="1.4"/>')
    text(s, x0 + 24, y, b, size=16, color=INK if b == "R0" else VIOLET, weight=700, anchor="middle")
    text(s, x0 + 60, y - 11, n, size=19, weight=700); text(s, x0 + 60, y + 13, d, size=15, color=MUTED)
    if i < 3: line(s, x0 + 4, y + 42, 978, y + 42, color="#E3E6EA", sw=1.0, arrow=False)
rect(s, x0, 812, 416, 120, fill=VIOLET_L, stroke=VIOLET, sw=1.0, rx=10)
text(s, x0 + 18, 842, "Contamination check", size=17, color=VIOLET, weight=700)
for k, t in enumerate(["No model's documented pretraining contains", "knockdown responses of Jurkat or HepG2."]):
    text(s, x0 + 18, 872 + k * 24, t, size=15, color=INK)
text(s, x0 + 10, 972, "Vectors centred on the five sources", size=15, color=MUTED)

# ---------------- column 3: prediction
x0 = 1062
text(s, x0, 140, "Similarity to each source", size=19, weight=700)
vec(s, 1080, 176, 8, 11, CORAL, seed=50, vertical=True)
text(s, 1086, 284, "held-out", size=14, color=CORAL, anchor="middle")
text(s, 1128, 220, "vs", size=16, color=MUTED, anchor="middle")
xs = [1166, 1229, 1292, 1355, 1418]; names = ["K562", "HCT116", "HEK293T", "H1", "H1"]
wts = [0.95, 0.42, 0.38, 0.12, 0.2]
for k, (xx, nm) in enumerate(zip(xs, names)):
    vec(s, xx, 176, 8, 11, TEAL, seed=60 + k, vertical=True)
    text(s, xx + 5.5, 284, nm, size=14, color=MUTED, anchor="middle")
bars(s, 1160, 372, wts, 22, 41, 56, [AMBER] * 5)
text(s, 1150, 340, "w", size=17, color=AMBER, weight=700, anchor="end", italic=True)
text(s, 1260, 398, "w = softmax(centred cosine / T)", size=16, color=INK, anchor="middle")
path(s, "M1094.7 431 A10 10 0 1 1 1094.7 421", color=MUTED, sw=1.6, arrow=True)
text(s, 1104, 426, "T chosen by leave-one-source-out", size=15, color=MUTED)
text(s, 1104, 448, "on the sources only", size=15, color=MUTED)
text(s, x0, 512, "Weighted average of responses", size=19, weight=700)
hx = [1076, 1162, 1248]; lab = ["w · K562", "w · HCT116", "w · HEK293T"]
for k, xx in enumerate(hx):
    heat(s, xx, 546, 5, 6, 10, seed=10 + k); text(s, xx + 30, 618, lab[k], size=14, color=MUTED, anchor="middle")
    text(s, xx + 73, 571, "+" if k < 2 else "", size=22, color=MUTED, anchor="middle")
text(s, 1338, 548, "+ …", size=15, color=MUTED, anchor="middle")
line(s, 1320, 571, 1356, 571)
heat(s, 1364, 546, 5, 6, 10, seed=77, outline=CORAL)
rect(s, 1364, 546, 60, 50, stroke=CORAL, sw=2.0, rx=0)
text(s, 1394, 618, "predicted", size=14, color=CORAL, anchor="middle")
rect(s, x0, 662, 396, 104, fill=AMBER_L, stroke=AMBER, sw=1.0, rx=10)
rich(s, x0 + 198, 698, [("ŷ", INK, 700), ("(k, g) = Σ", INK, 400), ("s", INK, 400, "sub"), (" w", AMBER, 700), ("s", AMBER, 400, "sub"), (" · LFC", INK, 400), ("s", INK, 400, "sub"), ("(k, g)", INK, 400)], size=22, anchor="middle")
text(s, x0 + 198, 738, "over sources that measured knockdown k", size=15, color=MUTED, anchor="middle")
text(s, x0, 812, "Baseline U: equal weights (plain pooling)", size=16, weight=700)
for k, t in enumerate(["LFC = log2 fold change vs that line's controls", "Sources: targets with ≥ 20 cells", "Identical for every arm"]):
    text(s, x0, 852 + k * 26, t, size=15, color=MUTED)

# ---------------- column 4: evaluation
x0 = 1542
ax, ay, aw, ah = 1572, 146, 190, 160
line(s, ax, ay + ah, ax + aw, ay + ah, color=INK, sw=1.2, arrow=False); line(s, ax, ay + ah, ax, ay, color=INK, sw=1.2, arrow=False)
rng = np.random.default_rng(3); px = rng.normal(0, 1, 70); py = 0.55 * px + rng.normal(0, 0.8, 70)
for a, b in zip(np.clip(px, -2.2, 2.2), np.clip(py, -2.4, 2.4)):
    s.add(f'<circle cx="{ax + aw/2 + a*38:.1f}" cy="{ay + ah/2 - b*30:.1f}" r="3.2" fill="{NAVY}" opacity="0.55"/>')
text(s, ax + aw / 2, ay + ah + 20, "predicted", size=14, color=MUTED, anchor="middle")
text(s, ax, ay - 16, "measured", size=14, color=MUTED, anchor="start")
text(s, 1790, 186, "Pooled", size=17, weight=700); text(s, 1790, 210, "Spearman ρ", size=17, weight=700)
for k, t in enumerate(["gene-centred,", "knockdown-specific", "effects; target", "gene masked"]):
    text(s, 1790, 240 + k * 22, t, size=15, color=MUTED)
text(s, x0, 384, "Pre-registered tests", size=19, weight=700)
tests = [("H1", "Foundation model vs raw profile", "CI excludes 0 on both lines"),
         ("H2", "Any weighting vs plain pooling", "same both-lines rule"),
         ("✓", "Beats its own shuffled labels", "knockdown rows deranged"),
         ("ref", "JaviCell ensemble, LRJ", "for context, no claim")]
for i, (b, t1, t2) in enumerate(tests):
    y = 414 + i * 82
    rect(s, x0, y, 396, 68, fill=NAVY_L if i < 2 else PANEL, stroke=NAVY if i < 2 else FRAME, sw=1.0, rx=10)
    text(s, x0 + 32, y + 34, b, size=17 if b != "ref" else 15, color=NAVY, weight=700, anchor="middle")
    text(s, x0 + 64, y + 22, t1, size=17, weight=700); text(s, x0 + 64, y + 46, t2, size=15, color=MUTED)
text(s, x0, 768, "Paired bootstrap over knockdowns,", size=15, color=MUTED)
text(s, x0, 790, "2,000 resamples; H1 Bonferroni over 3 models", size=15, color=MUTED)
rect(s, x0, 830, 396, 112, fill="#FFFFFF", stroke=NAVY, sw=1.2, rx=10, dash="6 4")
text(s, x0 + 18, 862, "Plan hashed before any result", size=17, color=NAVY, weight=700)
for k, t in enumerate(["Amendments dated and hashed;", "every arm reported either way."]):
    text(s, x0 + 18, 892 + k * 24, t, size=15, color=INK)

s.add("</g>")
s.out("/home/claude/figs/fig_fmbench.svg", "Foundation-model benchmark: do single-cell foundation models help predict knockdown responses in an unseen cell line?",
      "Four-column schematic: data (five source cell lines with known knockdown responses; Jurkat or HepG2 held out), representation of each line's control cells (raw profile or scGPT, Geneformer, State), prediction as a similarity-weighted average of source responses, and pre-registered evaluation with the challenge gate's metric.")
print("ok")
