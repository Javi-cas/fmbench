"""Combined figure: v1 four-stage workflow + v2 real-data panels. Boltz-2 regulator benchmark."""
import sys; sys.path.insert(0, "/home/claude/figs")
from v2_common import *
import struct_draw as SD

FW, FH = 180, 100
fig = plt.figure(figsize=(FW * MM, FH * MM))
REG = "#C8553D"; DEC = "#8C8C8C"
S = [(1, 53), (58, 47), (109, 40), (153, 26)]
for (x, w), L, t in zip(S, "abcd", ["Benchmark", "Co-folding", "Ranking", "Readouts"]):
    stage(fig, x, 1, w, 98, L, t)
for k in range(3): gutter_arrow(fig, S[k][0] + S[k][1] + 0.3, S[k + 1][0] - 0.3, 50)
def ftxt(x, y, s, **kw): fig.text(x / FW, y / FH, s, **kw)

# ---------- a: benchmark matrix (real design)
enz = [("PRKAG1", ["AMP"], 0), ("FBP1", ["AMP"], 0), ("PKM", ["FBP"], 0), ("GLUD1", ["GTP", "ADP"], 1), ("PRPS1", ["ADP"], 1),
       ("CTPS1", ["CTP"], 0), ("IMPDH2", ["ATP"], 0), ("PPAT", ["AMP"], 0), ("G6PD", ["NADP$^+$"], 0), ("PFKL", ["ATP", "citrate"], 0)]
regs = ["AMP", "ADP", "ATP", "GTP", "CTP", "FBP", "NADP$^+$", "citrate"]
decs = ["3PG", "G3P", "sucrose", "trehalose", "succinate", "taurine"]
cols = regs + [None] + decs
ax = axes_mm(fig, 14, 12, 38, 62); ax.patch.set_alpha(0)
for i, (e, rr, itf) in enumerate(enz):
    for j, c in enumerate(cols):
        if c is None: continue
        if c in rr: ax.plot(j, i, "o", ms=4.4, mfc=REG, mec=REG, mew=0.5)
        elif c in decs: ax.plot(j, i, "o", ms=3.0, mfc=mpl.colors.to_rgba(DEC, 0.55), mec="none")
        else: ax.plot(j, i, "o", ms=1.0, mfc=RULE, mec="none")
ax.set_xlim(-0.7, len(cols) - 0.3); ax.set_ylim(len(enz) - 0.4, -0.6)
ax.set_yticks(range(len(enz))); ax.set_yticklabels([e + ("*" if itf else "") for e, _, itf in enz], fontsize=5.6)
ax.set_xticks([j for j, c in enumerate(cols) if c]); ax.set_xticklabels([c for c in cols if c], rotation=90, fontsize=5.2)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.2)
for sp_ in ax.spines.values(): sp_.set_visible(False)
for i in range(len(enz)): ax.axhline(i, color=RULE, lw=0.3, zorder=0)
ftxt(14.2, 88, "regulators", fontsize=5.8, color=REG, fontweight="bold")
ftxt(37.6, 88, "decoys", fontsize=5.8, color=DEC, fontweight="bold")
ftxt(3.2, 7.4, "* site between subunits", fontsize=5.2, color=MUTED)
ftxt(3.2, 3.8, "12 regulator + 10 × 6 decoy = 72 complexes", fontsize=5.2, color=MUTED)

# ---------- b: co-folding (inputs -> Boltz-2 -> real structure)
cv = canvas(fig, 58, 1, 47, 98)
ins = [("sequence", 8), ("MSA", 23.5), ("ligand", 39)]
cv.add_patch(FancyBboxPatch((3.2, 85.5), 9.6, 3.6, boxstyle="round,pad=0,rounding_size=0.8", fc=TEAL_L if False else "#DCEFEC", ec=SRC, lw=0.5))
cv.text(8, 87.3, "MKLV…", fontsize=5, color=SRC, fontweight="bold", ha="center", va="center")
for j, Lw in enumerate([9, 7.5, 8.4, 6, 7.8, 5]):
    cv.plot([19, 19 + Lw], [89.4 - j * 0.85, 89.4 - j * 0.85], color=SRC, lw=1.0, alpha=1 - j * 0.12, solid_capstyle="butt")
pts = np.array([[39 + 1.9 * np.cos(a), 87.3 + 1.9 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6])
cv.add_patch(Polygon(pts, closed=True, fc="#FBE4DA", ec=REG, lw=0.6)); cv.plot([40.9, 42.2], [87.3, 87.3], color=REG, lw=0.6)
cv.add_patch(Circle((42.9, 87.3), 0.7, fc=REG, ec="none"))
for name, x in ins:
    cv.text(x, 83, name, fontsize=5.2, color=MUTED, ha="center", va="center")
    arrow(cv, x, 81.4, 23.5 + (x - 23.5) * 0.35, 77.4)
cv.add_patch(Polygon([(12.5, 77), (34.5, 77), (31, 71), (16, 71)], closed=True, fc=MODEL, ec=MODEL, lw=0.5))
cv.text(23.5, 74, "Boltz-2", color="white", fontsize=6, fontweight="bold", ha="center", va="center")
arrow(cv, 23.5, 70.6, 23.5, 67.8)
axs = axes_mm(fig, 59.5, 37, 44, 31); axs.set_aspect("equal"); axs.set_axis_off()
SD.draw_struct(axs, scale=0.8); SD.draw_lig(axs, r=0.95)
P = SD.P; pad = 2; axs.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad); axs.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
lc2 = SD.L[:, :2].mean(0); R_ = 10
axs.add_patch(Rectangle(lc2 - R_, 2 * R_, 2 * R_, fill=False, ec=INK, lw=0.45, ls=(0, (2, 1.5)), zorder=SD.ZTOP + 1))
axz = axes_mm(fig, 76, 10, 27, 22); axz.set_aspect("equal"); axz.set_xticks([]); axz.set_yticks([])
for sp_ in axz.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.5); sp_.set_linestyle((0, (2, 1.5)))
SD.draw_struct(axz, scale=1.5); SD.draw_lig(axz, r=0.75)
axz.set_xlim(lc2[0] - R_, lc2[0] + R_); axz.set_ylim(lc2[1] - R_ * 0.81, lc2[1] + R_ * 0.81)
axz.text(lc2[0] - R_ + 0.8, lc2[1] + R_ * 0.81 - 1, "AMP", fontsize=5.6, fontweight="bold", va="top", zorder=SD.ZTOP + 2,
         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
ftxt(81.5, 35.4, "AMPK γ1 + AMP (test run)", fontsize=5.4, color=MUTED, va="center", ha="center")
for k, (lab_, c_) in enumerate([("> 90", "#0053D6"), ("70–90", "#65CBF3"), ("50–70", "#FFDB13"), ("< 50", "#FF7D45")]):
    y = 27 - k * 4.6
    fig.patches.append(Rectangle((60.6 / FW, (y - 1) / FH), 2.2 / FW, 2.0 / FH, transform=fig.transFigure, fc=c_, ec="none"))
    ftxt(63.6, y, lab_, fontsize=5, va="center")
ftxt(60.6, 31.4, "pLDDT", fontsize=5.2, color=MUTED, va="center")
ftxt(60.6, 4.6, "binder probability + affinity", fontsize=5.2, color=MUTED, va="center")

# ---------- c: ranking (schematic score + real null)
cv = canvas(fig, 109, 1, 40, 98)
cv.text(2.2, 89, "score per pair", fontsize=5.8, color=MUTED, va="center")
cv.text(38, 89, "schematic", fontsize=5, color=FAINT, style="italic", ha="right", va="center")
yl = 72
cv.plot([3, 37], [yl, yl], color=INK, lw=0.5)
for xx in (3, 37): cv.plot([xx, xx], [yl - 0.7, yl + 0.7], color=INK, lw=0.5)
cv.text(3, yl - 1.6, "0", fontsize=5, ha="center", va="top"); cv.text(37, yl - 1.6, "1", fontsize=5, ha="center", va="top")
cv.text(20, yl - 4.6, "binder probability", fontsize=5.2, color=MUTED, ha="center", va="top")
for x in [6.5, 10, 14, 18.5, 23, 32]: cv.add_patch(Circle((x, yl), 0.9, fc=mpl.colors.to_rgba(DEC, 0.6), ec="none", zorder=3))
cv.add_patch(Circle((27.5, yl), 1.25, fc=REG, ec=REG, zorder=4))
cv.text(27.5, yl + 2.2, "regulator", fontsize=5.2, color=REG, ha="center", va="bottom", fontweight="bold")
cv.plot([6.5, 6.5, 23, 23], [yl + 3.4, yl + 4.4, yl + 4.4, yl + 3.4], color=MUTED, lw=0.5)
cv.text(14.75, yl + 5.2, "beats 5 of 6 → 0.83", fontsize=5.2, ha="center", va="bottom")
cv.text(2.2, 55, "mean over 12 pairs vs null", fontsize=5.8, color=MUTED, va="center")
axn = axes_mm(fig, 112, 14, 34, 34); axn.patch.set_alpha(0)
rng = np.random.default_rng(20261023); null = rng.integers(0, 7, size=(100000, 12)).mean(1) / 6
crit = float(np.quantile(null, 0.95)); vals, counts = np.unique(np.round(null, 4), return_counts=True)
axn.bar(vals, counts / counts.sum(), width=1 / 72 * 0.8, color=mpl.colors.to_rgba(DEC, 0.6), lw=0)
top = (counts / counts.sum()).max(); axn.set_ylim(0, top * 1.5)
axn.axvline(crit, color=REG, lw=0.7, ls=(0, (2.5, 1.5)))
axn.text(crit + 0.015, top * 1.45, f"p < 0.05\nif ≥ {crit:.2f}", fontsize=5.2, color=REG, va="top")
axn.text(0.5, top * 1.04, "chance", fontsize=5, color=MUTED, ha="center", va="bottom")
axn.set_xlim(0.15, 0.95); axn.set_xticks([0.25, 0.5, 0.75]); axn.set_yticks([]); axn.spines["left"].set_visible(False)
axn.set_xlabel("mean score (100,000 null draws)", fontsize=5.2, labelpad=1.2)

# ---------- d: readouts (terse, with icons)
cv = canvas(fig, 153, 1, 26, 98)
def dots(y, n_dec, phos=False, x0=3):
    cv.add_patch(Circle((x0, y), 1.0, fc=REG, ec="none"))
    for k in range(n_dec):
        cv.add_patch(Circle((x0 + 3.0 + k * 2.4, y), 0.75, fc=mpl.colors.to_rgba(DEC, 0.6), ec="none"))
        if phos: cv.text(x0 + 3.0 + k * 2.4, y, "P", fontsize=3.6, color="white", ha="center", va="center", fontweight="bold")
items = [(86, "all six decoys", ["recognises real", "regulators?"]),
         (66, "phosphorylated only", ["or just rewards", "phosphate charge?"]),
         (46, "interface sites", ["GLUD1, PRPS1: missed", "by single chains?"])]
for k, (y, t, sub) in enumerate(items):
    if k == 0: dots(y, 6)
    elif k == 1: dots(y, 2, phos=True)
    else:
        for dx in (-1.3, 1.3): cv.add_patch(Circle((4 + dx, y), 1.8, fc="#DCEFEC", ec=SRC, lw=0.5))
        cv.add_patch(Circle((4, y), 0.7, fc=REG, ec="none"))
    cv.text(2.2, y - 3.6, t, fontsize=5.8, fontweight="bold", va="center")
    for j, s_ in enumerate(sub): cv.text(2.2, y - 7.2 - j * 3.2, s_, fontsize=5.2, color=MUTED, va="center")
cv.add_patch(FancyBboxPatch((1.6, 6), 22.8, 20, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=REG, lw=0.5, ls=(0, (2, 1.5))))
for j, s_ in enumerate(["caveat", "most complexes are", "pre-2023 PDB entries:", "tests recall, not", "discovery"]):
    cv.text(3, 22.6 - j * 3.5, s_, fontsize=5.6 if j == 0 else 5.1, fontweight="bold" if j == 0 else "normal", color=REG if j == 0 else INK, va="center")

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"v3_boltz.{ext}", dpi=600 if ext == "png" else None)
print("ok", round(crit, 3))
