"""Figure: Boltz-2 metabolite-regulator benchmark schematic."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/figs")
from figlib import *

W, H = 1950, 1060
s = SVG(W, H); s.add('<g transform="translate(-40,0)">')
Y0, HC = 30, 1000
C = [(70, 430), (540, 460), (1040, 440), (1520, 440)]
for L, (x, w), (t, sub) in zip("abcd", C, [("Benchmark", "12 known regulator pairs, 10 enzymes"), ("Co-folding", "Boltz-2, one complex per ligand"),
                                           ("Ranking", "Regulator against its six decoys"), ("Readouts", "What each outcome would mean")]):
    column(s, x, Y0, w, HC, t, sub, letter=L)
for k in range(3):
    xa = C[k][0] + C[k][1]; line(s, xa + 6, 530, C[k + 1][0] - 6, 530, color=FAINT, sw=2.2)

def ligand(s, cx, cy, r=11, color=CORAL, light=CORAL_L, phos=True):
    pts = " ".join(f"{cx + r*np.cos(a):.1f},{cy + r*np.sin(a):.1f}" for a in np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6)
    s.add(f'<polygon points="{pts}" fill="{light}" stroke="{color}" stroke-width="1.5"/>')
    if phos:
        s.add(f'<line x1="{cx + r}" y1="{cy}" x2="{cx + r + 8}" y2="{cy}" stroke="{color}" stroke-width="1.5"/>')
        s.add(f'<circle cx="{cx + r + 13}" cy="{cy}" r="5" fill="{color}"/>')

def site_icon(s, cx, cy, interface):
    if interface:
        s.add(f'<circle cx="{cx-6}" cy="{cy}" r="8" fill="{TEAL_L}" stroke="{TEAL}" stroke-width="1.3"/>')
        s.add(f'<circle cx="{cx+6}" cy="{cy}" r="8" fill="{TEAL_L}" stroke="{TEAL}" stroke-width="1.3"/>')
        s.add(f'<circle cx="{cx}" cy="{cy}" r="3.6" fill="{CORAL}"/>')
    else:
        s.add(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{TEAL_L}" stroke="{TEAL}" stroke-width="1.3"/>')
        s.add(f'<circle cx="{cx+3}" cy="{cy-2}" r="3.6" fill="{CORAL}"/>')

# ---------------- a: benchmark table
x0 = 92
text(s, x0, 136, "Enzyme", size=15, color=MUTED, weight=700); text(s, 268, 136, "Known regulator", size=15, color=MUTED, weight=700)
text(s, 470, 136, "Site", size=15, color=MUTED, weight=700, anchor="end")
pairs = [("PRKAG1", "AMPK γ1", "AMP", 0), ("FBP1", "", "AMP", 0), ("PKM", "", "fructose-1,6-bisP", 0),
         ("GLUD1", "", "GTP", 1), ("GLUD1", "", "ADP", 1), ("PRPS1", "", "ADP", 1), ("CTPS1", "", "CTP", 0),
         ("IMPDH2", "", "ATP", 0), ("PPAT", "", "AMP", 0), ("G6PD", "", "NADP⁺ (structural)", 0),
         ("PFKL", "", "ATP", 0), ("PFKL", "", "citrate", 0)]
for i, (e, alias, r, itf) in enumerate(pairs):
    y = 170 + i * 40
    if i % 2 == 0: rect(s, x0 - 8, y - 18, 394, 36, fill=PANEL, stroke="none", rx=6)
    text(s, x0, y, e, size=17, weight=700)
    if alias: text(s, x0 + 84, y, alias, size=14, color=MUTED)
    chip(s, 268, y - 13, r, CORAL, CORAL_L, size=14, pad=9, h=26)
    site_icon(s, 458, y, itf)
yb = 170 + 12 * 40 + 6
s.add(f'<circle cx="{x0+8}" cy="{yb+8}" r="0"/>')
site_icon(s, x0 + 8, yb + 6, 0); text(s, x0 + 24, yb + 6, "within one chain", size=14, color=MUTED)
site_icon(s, x0 + 186, yb + 6, 1); text(s, x0 + 204, yb + 6, "between subunits", size=14, color=MUTED)
line(s, x0, 694, 478, 694, color=FRAME, sw=1.2, arrow=False, dash="5 5")
text(s, x0, 728, "Six decoys per enzyme", size=19, weight=700)
dec = [("phosphorylated ×2", "3PG, G3P, PEP, R5P or G6P"), ("neutral sugars ×2", "sucrose, trehalose"), ("small metabolites ×2", "succinate, taurine")]
for k, (a, b) in enumerate(dec):
    y = 770 + k * 56
    ligand(s, x0 + 14, y, r=10, color=FAINT, light=PANEL, phos=(k == 0))
    text(s, x0 + 48, y - 10, a, size=16, weight=700); text(s, x0 + 48, y + 12, b, size=15, color=MUTED)
text(s, x0, 952, "Never the enzyme's own substrate,", size=14, color=MUTED, italic=True)
text(s, x0, 972, "product or effector.", size=14, color=MUTED, italic=True)

# ---------------- b: co-folding
x0 = 562
inputs = [("Sequence", "UniProt"), ("Alignment", "ColabFold MSA"), ("Ligand", "CCD / SMILES")]
for k, (a, b) in enumerate(inputs):
    cx = 612 + k * 150
    if k == 0:
        rect(s, cx - 40, 132, 80, 34, fill=TEAL_L, stroke=TEAL, sw=1.2, rx=6)
        text(s, cx, 149, "MKLV…", size=15, color=TEAL, weight=700, anchor="middle")
    elif k == 1:
        for j, L in enumerate([76, 62, 70, 50, 66, 40]):
            s.add(f'<line x1="{cx-38}" y1="{130 + j*7}" x2="{cx-38+L}" y2="{130 + j*7}" stroke="{TEAL}" stroke-width="3" opacity="{1 - j*0.12:.2f}"/>')
    else:
        ligand(s, cx - 6, 149, r=13)
    text(s, cx, 188, a, size=16, weight=700, anchor="middle"); text(s, cx, 208, b, size=14, color=MUTED, anchor="middle")
    line(s, cx, 222, 762 + (k - 1) * 40, 262)
transformer(s, 668, 274, 196, 62, VIOLET, VIOLET_L, "Boltz-2")
text(s, 780, 360, "structure + affinity heads", size=15, color=MUTED, anchor="middle")
line(s, 780, 376, 780, 410)
cx, cy = 780, 500
s.add(f'<path d="M{cx-92},{cy+10} C{cx-104},{cy-60} {cx-40},{cy-92} {cx+10},{cy-80} C{cx+40},{cy-74} {cx+44},{cy-40} {cx+22},{cy-28} '
      f'C{cx+6},{cy-20} {cx+12},{cy+6} {cx+40},{cy+4} C{cx+92},{cy} {cx+100},{cy+64} {cx+40},{cy+82} C{cx-20},{cy+100} {cx-82},{cy+70} {cx-92},{cy+10} Z" '
      f'fill="{TEAL_L}" stroke="{TEAL}" stroke-width="1.6"/>')
for k in range(3):
    yy = cy - 40 + k * 34; xx = cx - 66 + (k % 2) * 10
    d = f"M{xx},{yy} " + " ".join(f"q 8 {'-' if j % 2 == 0 else ''}12 16 0" for j in range(5))
    s.add(f'<path d="{d}" fill="none" stroke="{TEAL}" stroke-width="2.2" opacity="0.6"/>')
ligand(s, cx + 40, cy - 22, r=12)
text(s, cx + 104, cy - 52, "pocket", size=14, color=MUTED)
line(s, cx + 100, cy - 48, cx + 60, cy - 30, color=FAINT, sw=1.0, arrow=False)
outs = [("Structure confidence", "pLDDT, interface pTM"), ("Binder probability", "primary score"), ("Predicted affinity", "log10 IC50, secondary score")]
for k, (a, b) in enumerate(outs):
    y = 640 + k * 70
    rect(s, x0, y, 416, 58, fill=VIOLET_L if k else PANEL, stroke=VIOLET if k else FRAME, sw=1.0, rx=10)
    text(s, x0 + 18, y + 19, a, size=17, weight=700); text(s, x0 + 18, y + 41, b, size=15, color=MUTED)
for k, t in enumerate(["72 single-chain predictions", "1 diffusion sample each, run on CPU", "MSA computed once per enzyme, then reused"]):
    text(s, x0, 878 + k * 26, t, size=15, color=MUTED)

# ---------------- c: ranking
x0 = 1062
text(s, x0, 136, "One enzyme, seven ligands", size=19, weight=700)
text(s, x0, 160, "schematic, not a result", size=14, color=MUTED, italic=True)
vals = [0.86, 0.71, 0.55, 0.42, 0.30, 0.22, 0.12]
cols = [MUTED, CORAL, FAINT, MUTED, FAINT, FAINT, FAINT]
labs = ["P", "R", "S", "P", "S", "m", "m"]
bx, by0, bw, gap = 1100, 372, 34, 18
for k, (v, c) in enumerate(zip(vals, cols)):
    h = v * 150
    fill = CORAL if c == CORAL else (mix("#FFFFFF", c, 0.55))
    s.add(f'<rect x="{bx + k*(bw+gap)}" y="{by0 - h}" width="{bw}" height="{h}" fill="{fill}" rx="2"/>')
    text(s, bx + k * (bw + gap) + bw / 2, by0 + 18, labs[k], size=15, color=CORAL if c == CORAL else MUTED, weight=700, anchor="middle")
line(s, bx - 8, by0, bx + 7 * (bw + gap) - gap + 8, by0, color=INK, sw=1.0, arrow=False)
line(s, bx - 8, by0, bx - 8, by0 - 160, color=INK, sw=1.0, arrow=False)
text(s, bx - 8, by0 - 176, "binder probability", size=14, color=MUTED)
leg = [("R", "regulator", CORAL), ("P", "phosphorylated decoy", MUTED), ("S", "sugar", MUTED), ("m", "small metabolite", MUTED)]
for k, (a, b, c) in enumerate(leg):
    xx = x0 + (k % 2) * 200; yy = 420 + (k // 2) * 24
    text(s, xx, yy, a, size=14, color=c, weight=700); text(s, xx + 18, yy, b, size=14, color=MUTED)
rect(s, x0, 476, 396, 70, fill=AMBER_L, stroke=AMBER, sw=1.0, rx=10)
text(s, x0 + 198, 500, "score = share of decoys the regulator beats", size=16, anchor="middle")
text(s, x0 + 198, 526, "here 5 of 6 = 0.83; chance = 0.5", size=15, color=MUTED, anchor="middle")
text(s, x0, 590, "Across the 12 pairs", size=19, weight=700)
mx, my, mw, mh = 1086, 760, 340, 120
xs = np.linspace(0.15, 0.9, 120); dens = np.exp(-0.5 * ((xs - 0.5) / 0.096) ** 2)
X = lambda v: mx + (v - 0.15) / 0.75 * mw
pts = " ".join(f"{X(a):.1f},{my - d*mh:.1f}" for a, d in zip(xs, dens))
s.add(f'<polygon points="{X(0.15):.1f},{my} {pts} {X(0.9):.1f},{my}" fill="{PANEL}" stroke="{FAINT}" stroke-width="1.2"/>')
line(s, mx, my, mx + mw, my, color=INK, sw=1.0, arrow=False)
for v in (0.25, 0.5, 0.75):
    line(s, X(v), my, X(v), my + 6, color=INK, sw=1.0, arrow=False); text(s, X(v), my + 20, f"{v:.2f}", size=13, color=MUTED, anchor="middle")
line(s, X(0.658), my, X(0.658), my - mh - 6, color=NAVY, sw=1.6, arrow=False, dash="5 4")
text(s, X(0.658) + 8, my - mh + 4, "p < 0.05 needs", size=14, color=NAVY); text(s, X(0.658) + 8, my - mh + 22, "mean ≥ 0.66", size=14, color=NAVY, weight=700)
text(s, X(0.5), my - 34, "null", size=14, color=MUTED, anchor="middle")
text(s, mx + mw / 2, my + 44, "mean score (uniform-rank null, 100,000 draws)", size=14, color=MUTED, anchor="middle")
for k, t in enumerate(["Primary: binder probability", "Secondary: predicted affinity"]):
    text(s, x0, 880 + k * 26, t, size=15, color=MUTED)

# ---------------- d: readouts
x0 = 1542
cards = [("All six decoys", "Recognises real regulators?", "Yes → worth screening new", "metabolite-enzyme pairs", NAVY, NAVY_L),
         ("Phosphorylated decoys only", "Or just rewards phosphate charge?", "If only sugars are beaten, it", "cannot separate true regulators", NAVY, NAVY_L),
         ("Within chain vs interface", "GLUD1 and PRPS1 are regulated", "between subunits; single chains", "should miss those sites", NAVY, NAVY_L)]
for k, (t, a, b, c, col, lt) in enumerate(cards):
    y = 120 + k * 172
    rect(s, x0, y, 396, 150, fill=lt, stroke=col, sw=1.0, rx=12)
    text(s, x0 + 20, y + 30, t, size=18, color=col, weight=700)
    for j, tt in enumerate([a, b, c]):
        text(s, x0 + 20, y + 64 + j * 26, tt, size=15, color=INK if j == 0 else MUTED)
rect(s, x0, 650, 396, 150, fill="#FFFFFF", stroke=CORAL, sw=1.2, rx=12, dash="6 4")
text(s, x0 + 20, 680, "Caveat stated in advance", size=17, color=CORAL, weight=700)
for j, tt in enumerate(["Most of these complexes have pre-2023", "PDB structures, so Boltz-2 has probably", "seen them: this tests recall of known", "biology, not discovery."]):
    text(s, x0 + 20, 712 + j * 23, tt, size=15, color=INK)
rect(s, x0, 830, 396, 112, fill="#FFFFFF", stroke=NAVY, sw=1.2, rx=10, dash="6 4")
text(s, x0 + 18, 862, "Plan hashed before any prediction", size=17, color=NAVY, weight=700)
for k, t in enumerate(["Runner and device changes dated;", "same analysis whatever the result."]):
    text(s, x0 + 18, 892 + k * 24, t, size=15, color=INK)

s.add("</g>")
s.out("/home/claude/figs/fig_boltz.svg", "Boltz-2 regulator benchmark: does a co-folding model rank an enzyme's known allosteric regulator above decoy metabolites?",
      "Four-column schematic: 12 known enzyme-regulator pairs with six decoys each, co-folding with Boltz-2 to obtain structure confidence, binder probability and affinity, ranking the regulator among seven ligands against a uniform-rank null, and three pre-specified readouts plus the training-data caveat.")
print("ok")
