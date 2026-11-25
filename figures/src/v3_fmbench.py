"""Combined figure: v1 four-stage workflow + v2 real-data panels. Foundation-model benchmark."""
import sys; sys.path.insert(0, "/home/claude/figs")
from v2_common import *
from scipy.cluster.hierarchy import linkage, leaves_list

FW, FH = 180, 104
fig = plt.figure(figsize=(FW * MM, FH * MM))
LINES = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214", "Jurkat", "HepG2"]
S = [(1, 45), (49, 40), (93, 45), (142, 37)]
for (x, w), L, t in zip(S, "abcd", ["Data", "Representation", "Prediction", "Evaluation"]):
    stage(fig, x, 1, w, 102, L, t)
for k in range(3): gutter_arrow(fig, S[k][0] + S[k][1] + 0.3, S[k + 1][0] - 0.3, 52)
def ftxt(x, y, s, **kw): fig.text(x / FW, y / FH, s, **kw)

# ---------- a: data (real)
u = np.load("data_umap.npz"); E, lab = u["E"], u["line"].astype(str)
ax = axes_mm(fig, 3.5, 58, 40, 34)
for l in LINES:
    m = lab == l; ax.scatter(E[m, 0], E[m, 1], s=0.45, lw=0, color=LINE_COL[l], alpha=0.75, rasterized=True)
for l in LINES:
    c = np.median(E[lab == l], 0); left = c[0] < np.percentile(E[:, 0], 12)
    below = l == "H1_GSE295214"
    ax.text(c[0] - (1.5 if left else 0), c[1] + (-1.8 if below else 1.5), LINE_LAB[l], color=LINE_COL[l], fontsize=5.4, fontweight="bold", ha="left" if left else "center", va="top" if below else "bottom")
ax.set_xticks([]); ax.set_yticks([]); ax.spines[["left", "bottom"]].set_visible(False); ax.patch.set_alpha(0)
x0, y0 = E[:, 0].min(), E[:, 1].min()
for d_ in ((3.5, 0), (0, 3.5)):
    ax.annotate("", (x0 + d_[0], y0 + d_[1]), (x0, y0), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=0.5, color=INK, mutation_scale=5))
ax.text(x0, y0 - 0.6, "UMAP", fontsize=5, va="top", color=MUTED)
ftxt(3.5, 94, "Control cells, 7 lines", fontsize=6, color=MUTED, va="center")
h = dict(np.load("data_heat.npz")); kd = h["kd"].astype(str)
both = np.hstack([h["K562"], h["HCT116"]]); ro = leaves_list(linkage(both, "average", metric="correlation"))
co = leaves_list(linkage(np.vstack([h["K562"], h["HCT116"]]).T, "average", metric="correlation"))
h["K562"], h["HCT116"] = h["K562"][ro][:, co], h["HCT116"][ro][:, co]; kd = kd[ro]
norm = mpl.colors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5); cmap = plt.get_cmap("RdBu_r")
ftxt(3.5, 53, "Knockdown responses", fontsize=6, color=MUTED, va="center")
for k, (name, x) in enumerate([("K562", 12), ("HCT116", 24)]):
    axh = axes_mm(fig, x, 12, 11, 34)
    axh.imshow(np.clip(h[name], -1.5, 1.5), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks(range(len(kd)) if k == 0 else [])
    if k == 0: axh.set_yticklabels(kd, fontsize=4.2, style="italic")
    axh.tick_params(length=0, pad=0.8)
    for sp_ in axh.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.4)
    axh.set_title(name, fontsize=5.8, color=LINE_COL[name], fontweight="bold", pad=1.5)
mpl.rcParams["hatch.color"] = mpl.colors.to_rgba(HELD, 0.45); mpl.rcParams["hatch.linewidth"] = 0.4
cv = canvas(fig, 36, 12, 7.5, 34)
cv.add_patch(Rectangle((0.3, 0), 7, 34, facecolor="white", edgecolor=HELD, lw=0.6, hatch="////"))
cv.text(3.8, 17, "?", color=HELD, fontsize=9, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
ftxt(39.8, 46.8, "held out", fontsize=5.2, color=HELD, fontweight="bold", ha="center", va="bottom")
ftxt(23.5, 10.4, "16 shared knockdowns × 28 genes", fontsize=5, color=MUTED, ha="center", va="top")
cb = axes_mm(fig, 24, 4, 11, 1.2)
cbar = mpl.colorbar.ColorbarBase(cb, cmap=cmap, norm=norm, orientation="horizontal", ticks=[-1.5, 0, 1.5])
cbar.outline.set_linewidth(0.4); cb.tick_params(labelsize=4.6, length=1.2, pad=0.8)
ftxt(23, 4.6, "log$_2$ FC", fontsize=5, ha="right", va="center")

# ---------- b: representation
cv = canvas(fig, 49, 1, 40, 102)
rng = np.random.default_rng(4)
for k in range(13):
    a_, r_ = rng.uniform(0, 2 * np.pi), rng.uniform(0, 3.2)
    cv.add_patch(Circle((6 + r_ * np.cos(a_), 82 + r_ * np.sin(a_)), 1.0, fc=mpl.colors.to_rgba(SRC, 0.25), ec=SRC, lw=0.35))
arrow(cv, 10.6, 82, 13.4, 82)
cv.add_patch(Polygon([(14, 76), (14, 88), (23, 85), (23, 79)], closed=True, fc=MODEL, ec=MODEL, lw=0.5))
cv.text(18.5, 82, "encoder", color="white", fontsize=5, ha="center", va="center", fontweight="bold")
arrow(cv, 23.6, 82, 26.4, 82)
for k in range(4): vec(cv, 27.2 + k * 1.9, 76.5, 6, 1.8, MODEL, seed=30 + k)
arrow(cv, 35.4, 82, 37.6, 82)
cv.text(6, 76.3, "2,000 cells", fontsize=5, color=MUTED, ha="center", va="top")
cv.text(31, 74.3, "per cell", fontsize=5, color=MUTED, ha="center", va="top")
vec(cv, 14, 64, 10, 2.0, MODEL, seed=41, vertical=False)
cv.text(24, 61.8, "mean: one vector per line", fontsize=5, color=MUTED, ha="center", va="top")
cv.plot([38.4, 38.4, 24, 24], [82, 70, 70, 67], color=MUTED, lw=0.6)
arrow(cv, 24, 67.5, 24, 66.4)
arms = [("R0", "raw profile", "2,000 variable genes", FAINT), ("R1", "scGPT", "whole-human", MODEL),
        ("R2", "Geneformer", "V2-104M", MODEL), ("R3", "State", "SE-600M", MODEL)]
for i, (b_, t1, t2, c_) in enumerate(arms):
    y = 50 - i * 9.5
    cv.add_patch(FancyBboxPatch((2.2, y - 2), 6.2, 4, boxstyle="round,pad=0,rounding_size=0.9", fc=mpl.colors.to_rgba(c_, 0.15), ec=c_, lw=0.5))
    cv.text(5.3, y, b_, fontsize=5.6, fontweight="bold", color=INK if b_ == "R0" else MODEL, ha="center", va="center")
    cv.text(10.5, y + 1.3, t1, fontsize=6, fontweight="bold", ha="left", va="center")
    cv.text(10.5, y - 1.8, t2, fontsize=5.2, color=MUTED, ha="left", va="center")
cv.text(2.2, 56.5, "Four representations", fontsize=6, color=MUTED, ha="left", va="center")
cv.text(2.2, 9.5, "same cells and genes", fontsize=5.2, color=MUTED); cv.text(2.2, 5.8, "for every arm (6,360 genes)", fontsize=5.2, color=MUTED)

# ---------- c: prediction
cv = canvas(fig, 93, 1, 45, 102)
cv.text(2.2, 94, "Similarity to each source", fontsize=6, color=MUTED, ha="left", va="center")
vec(cv, 4, 72, 8, 1.9, HELD, seed=11); cv.text(4.95, 70.4, "held out", fontsize=4.8, color=HELD, ha="center", va="top")
cv.text(9.6, 79.6, "vs", fontsize=5.4, color=MUTED, ha="center", va="center")
srcs = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214"]; w = [0.95, 0.42, 0.37, 0.10, 0.18]
for k, l in enumerate(srcs):
    x = 14 + k * 5.6
    vec(cv, x, 72, 8, 1.9, LINE_COL[l], seed=20 + k)
    cv.add_patch(Rectangle((x, 56), 1.9, 10 * w[k], fc=LINE_COL[l], ec="none"))
cv.plot([13, 14 + 4 * 5.6 + 2.9], [56, 56], color=INK, lw=0.5)
cv.text(12.2, 61, "w", fontsize=6.5, style="italic", ha="right", va="center")
cv.text(26.5, 53.5, "softmax(cosine / T)", fontsize=5.2, color=MUTED, ha="center", va="top")
cv.text(26.5, 49.6, "T: leave-one-source-out", fontsize=5.2, color=MUTED, ha="center", va="top")
cv.text(2.2, 41, "Weighted average of responses", fontsize=6, color=MUTED, ha="left", va="center")
tiles = [("K562", h["K562"][:8, :8]), ("HCT116", h["HCT116"][:8, :8])]
for k, (name, T_) in enumerate(tiles):
    x = 5 + k * 14
    cv.text(x - 0.4, 27, "w·", fontsize=5.6, style="italic", ha="right", va="center")
    cv.imshow(np.clip(T_, -1.5, 1.5), cmap=cmap, norm=norm, extent=(x, x + 8, 23, 31), interpolation="nearest", zorder=2)
    cv.add_patch(Rectangle((x, 23), 8, 8, fill=False, ec=LINE_COL[name], lw=0.7, zorder=3))
    if k == 0: cv.text(x + 10.1, 27, "+", fontsize=7, color=MUTED, ha="center", va="center")
cv.text(30.4, 27, "+ …", fontsize=6.5, color=MUTED, ha="center", va="center")
arrow(cv, 33.4, 27, 35.4, 27)
Pm = 0.62 * h["K562"][:8, :8] + 0.38 * h["HCT116"][:8, :8]
cv.imshow(np.clip(Pm, -1.5, 1.5), cmap=cmap, norm=norm, extent=(35.8, 43.8, 23, 31), interpolation="nearest", zorder=2)
cv.add_patch(Rectangle((35.8, 23), 8, 8, fill=False, ec=HELD, lw=1.0, zorder=3))
cv.text(39.8, 21.5, "prediction", fontsize=4.8, color=HELD, ha="center", va="top")
cv.text(22.5, 12, r"$\hat{y}_{k,g}=\sum_s\, w_s\, \mathrm{LFC}_{s,k,g}$", fontsize=7, ha="center", va="center")
cv.text(22.5, 5.6, "baseline U: equal weights", fontsize=5.2, color=MUTED, ha="center", va="center")

# ---------- d: evaluation
ftxt(144.2, 95, "Gain over R0, 98.3% CI", fontsize=6, color=MUTED, va="center")
ftxt(144.2, 91.4, "results pending", fontsize=5.2, color=FAINT, style="italic", va="center")
for j, (fold, y) in enumerate([("Jurkat", 62), ("HepG2", 33)]):
    axe = axes_mm(fig, 155, y, 21, 22)
    axe.axvspan(0, 0.05, color=mpl.colors.to_rgba(MODEL, 0.07), lw=0); axe.axvline(0, color=INK, lw=0.5)
    for i in range(3):
        axe.plot([-0.02, 0.03], [i, i], color=FAINT, lw=0.9, ls=(0, (2, 1.5)))
        axe.plot([0.005], [i], "o", ms=2.0, mfc="white", mec=FAINT, mew=0.6)
    axe.set_ylim(-0.7, 2.7); axe.set_xlim(-0.04, 0.05); axe.invert_yaxis(); axe.patch.set_alpha(0)
    axe.set_yticks(range(3)); axe.set_yticklabels(["R1", "R2", "R3"])
    axe.set_xticks([-0.02, 0, 0.02, 0.04]); axe.set_xticklabels(["−.02", "0", ".02", ".04"] if j == 1 else [])
    axe.spines["left"].set_visible(False); axe.tick_params(axis="y", length=0)
    axe.set_title(fold, fontsize=6, color=LINE_COL[fold], fontweight="bold", pad=1.5)
ftxt(165.5, 28, "Δρ (gene-centred Spearman)", fontsize=5.2, ha="center", va="center")
tests = [("H1", "beats R0 on both lines"), ("H2", "beats plain pooling U"), ("✓", "beats shuffled labels")]
for i, (b_, t_) in enumerate(tests):
    y = 20 - i * 6
    fig.patches.append(FancyBboxPatch((144.2 / FW, (y - 2) / FH), 5.6 / FW, 4 / FH, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fc=NAVY_L if False else "#E7ECF2", ec="#2F4A63", lw=0.5))
    ftxt(147, y, b_, fontsize=5.4, fontweight="bold", color="#2F4A63", ha="center", va="center", fontfamily="DejaVu Sans" if b_ == "✓" else None)
    ftxt(151.4, y, t_, fontsize=5.6, va="center")
ftxt(144.2, 4.4, "plan hashed before any result", fontsize=5.2, color=MUTED, style="italic", va="center")

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"v3_fmbench.{ext}", dpi=600 if ext == "png" else None)
print("ok")
