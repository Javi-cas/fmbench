import sys; sys.path.insert(0, "/home/claude/figs")
from v2_common import *

fig = plt.figure(figsize=(180 * MM, 112 * MM))
LINES = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214", "Jurkat", "HepG2"]

# ---------- a: real control cells
u = np.load("data_umap.npz"); E, lab = u["E"], u["line"].astype(str)
ax = axes_mm(fig, 5, 62, 46, 42)
for l in LINES:
    m = lab == l
    ax.scatter(E[m, 0], E[m, 1], s=0.6, lw=0, color=LINE_COL[l], alpha=0.7, rasterized=True)
for l in LINES:
    c = np.median(E[lab == l], 0)
    ax.text(c[0], c[1] + 1.6, LINE_LAB[l], color=LINE_COL[l], fontsize=5.8, fontweight="bold", ha="center", va="bottom")
ax.set_xticks([]); ax.set_yticks([]); ax.spines[["left", "bottom"]].set_visible(False)
x0, y0 = E[:, 0].min(), E[:, 1].min()
ax.annotate("", (x0 + 3.5, y0), (x0, y0), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=0.5, color=INK, mutation_scale=5))
ax.annotate("", (x0, y0 + 3.5), (x0, y0), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=0.5, color=INK, mutation_scale=5))
ax.text(x0, y0 - 0.6, "UMAP", fontsize=5.5, va="top", color=MUTED)
ax.set_title("Control cells of seven lines (700 per line shown)", fontsize=6.5, loc="left", pad=3)
letter(fig, 0, 110, "a")

# ---------- b: real knockdown responses in two source lines; held-out block
h = dict(np.load("data_heat.npz")); kd = h["kd"].astype(str)
from scipy.cluster.hierarchy import linkage, leaves_list
both = np.hstack([h["K562"], h["HCT116"]]); ro = leaves_list(linkage(both, "average", metric="correlation"))
co = leaves_list(linkage(np.vstack([h["K562"], h["HCT116"]]).T, "average", metric="correlation"))
h["K562"], h["HCT116"] = h["K562"][ro][:, co], h["HCT116"][ro][:, co]; kd = kd[ro]
norm = mpl.colors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5); cmap = plt.get_cmap("RdBu_r")
for k, (name, x) in enumerate([("K562", 74), ("HCT116", 101)]):
    axh = axes_mm(fig, x, 63, 25, 40)
    axh.imshow(np.clip(h[name], -1.5, 1.5), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks(range(len(kd)) if k == 0 else [])
    if k == 0: axh.set_yticklabels(kd, fontsize=4.8, style="italic")
    axh.tick_params(length=0, pad=1)
    for sp_ in axh.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.4)
    axh.set_title(name, fontsize=6.5, color=LINE_COL[name], fontweight="bold", pad=2)
    axh.set_xlabel("28 most variable genes", fontsize=5.5, color=MUTED, labelpad=1.5)
cv = canvas(fig, 128, 63, 14, 40)
cv.add_patch(Rectangle((0.5, 0), 12, 40, facecolor="white", edgecolor=HELD, lw=0.6, hatch="////"))
mpl.rcParams["hatch.color"] = mpl.colors.to_rgba(HELD, 0.45); mpl.rcParams["hatch.linewidth"] = 0.4
cv.text(6.5, 20, "?", color=HELD, fontsize=11, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
fig.text(135 / 180, 104.6 / 112, "Jurkat / HepG2", fontsize=6.5, color=HELD, fontweight="bold", ha="center", va="bottom")
fig.text(135 / 180, 61.2 / 112, "held out", fontsize=5.5, color=MUTED, ha="center", va="top")
cb = axes_mm(fig, 101, 55.5, 25, 1.4)
cbar = mpl.colorbar.ColorbarBase(cb, cmap=cmap, norm=norm, orientation="horizontal", ticks=[-1.5, 0, 1.5])
cbar.outline.set_linewidth(0.4); cb.tick_params(labelsize=5, length=1.5, pad=1)
fig.text(100 / 180, 56.2 / 112, "log$_2$ FC", fontsize=5.5, ha="right", va="center")
letter(fig, 55, 110, "b")
fig.text(74 / 180, 107.6 / 112, "Knockdown responses (strongest shared knockdowns)", fontsize=6.5, ha="left", va="bottom")

# ---------- c: representation
cv = canvas(fig, 147, 54, 33, 52)
rng = np.random.default_rng(4)
for k in range(14):
    a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 3.4)
    cv.add_patch(Circle((5 + r * np.cos(a), 39 + r * np.sin(a)), 1.05, fc=mpl.colors.to_rgba(SRC, 0.25), ec=SRC, lw=0.35))
arrow(cv, 9.6, 39, 12.4, 39)
cv.add_patch(Polygon([(13, 33), (13, 45), (21, 42), (21, 36)], closed=True, fc=MODEL, ec=MODEL, lw=0.5))
cv.text(17, 39, "encoder", color="white", fontsize=5.2, ha="center", va="center", fontweight="bold")
arrow(cv, 21.6, 39, 24.4, 39)
vec(cv, 25, 33.5, 6, 1.9, MODEL, seed=3)
cv.text(5, 33.6, "control cells", fontsize=5.3, color=MUTED, ha="center", va="top")
cv.text(26, 31.8, "line vector", fontsize=5.3, color=MUTED, ha="center", va="top")
cv.text(0, 48.5, "Four representations", fontsize=6.5, ha="left", va="bottom")
arms = [("R0", "raw profile, 2,000 genes", FAINT), ("R1", "scGPT, whole-human", MODEL), ("R2", "Geneformer V2-104M", MODEL), ("R3", "State SE-600M", MODEL)]
for i, (b_, t_, c_) in enumerate(arms):
    y = 23 - i * 5.6
    cv.add_patch(FancyBboxPatch((0, y - 1.7), 5.6, 3.4, boxstyle="round,pad=0,rounding_size=0.8", fc=mpl.colors.to_rgba(c_, 0.15), ec=c_, lw=0.5))
    cv.text(2.8, y, b_, fontsize=5.6, fontweight="bold", color=INK if b_ == "R0" else MODEL, ha="center", va="center")
    cv.text(7.2, y, t_, fontsize=5.8, ha="left", va="center")
cv.text(0, 1.6, "mean over 2,000 cells; 6,360 genes", fontsize=5.3, color=MUTED)
letter(fig, 144, 110, "c")

# ---------- d: prediction
cv = canvas(fig, 4, 4, 116, 46)
letter(fig, 0, 52, "d")
cv.text(1, 44, "Similarity-weighted borrowing of responses", fontsize=6.5, ha="left", va="bottom")
vec(cv, 3, 20, 8, 2.1, HELD, seed=11)
cv.text(4.05, 18.6, "held-out", fontsize=5.3, color=HELD, ha="center", va="top")
cv.text(9.5, 28.4, "vs", fontsize=5.8, color=MUTED, ha="center", va="center")
srcs = ["K562", "HCT116", "HEK293T", "H1_2025", "H1_GSE295214"]; w = [0.95, 0.42, 0.37, 0.10, 0.18]
for k, l in enumerate(srcs):
    x = 14 + k * 5.4
    vec(cv, x, 20, 8, 2.1, LINE_COL[l], seed=20 + k)
    cv.add_patch(Rectangle((x, 6), 2.1, 9 * w[k], fc=LINE_COL[l], ec="none"))
cv.plot([12.8, 13 + 4 * 5.4 + 3.3], [6, 6], color=INK, lw=0.5)
cv.text(12.2, 10.5, "w", fontsize=6.5, style="italic", ha="right", va="center")
cv.text(25.6, 3.8, "softmax(cosine / T)", fontsize=5.5, color=MUTED, ha="center", va="top")
arrow(cv, 41, 28.4, 45, 28.4)
tiles = [("K562", h["K562"][:9, :11]), ("HCT116", h["HCT116"][:9, :11])]
for k, (name, T_) in enumerate(tiles):
    x = 50.5 + k * 21
    cv.text(x - 0.6, 28.4, "w·", fontsize=6, style="italic", ha="right", va="center")
    cv.imshow(np.clip(T_, -1.5, 1.5), cmap=cmap, norm=norm, extent=(x, x + 13, 22, 35), interpolation="nearest", zorder=2)
    cv.add_patch(Rectangle((x, 22), 13, 13, fill=False, ec=LINE_COL[name], lw=0.7, zorder=3))
    cv.text(x + 13 + 3.2, 28.4, "+", fontsize=8, color=MUTED, ha="center", va="center")
cv.text(93.2, 28.4, "…", fontsize=8, color=MUTED, ha="center", va="center")
arrow(cv, 95.6, 28.4, 99.4, 28.4)
P = 0.62 * h["K562"][:9, :11] + 0.38 * h["HCT116"][:9, :11]
cv.imshow(np.clip(P, -1.5, 1.5), cmap=cmap, norm=norm, extent=(100.5, 113.5, 22, 35), interpolation="nearest", zorder=2)
cv.add_patch(Rectangle((100.5, 22), 13, 13, fill=False, ec=HELD, lw=1.1, zorder=3))
cv.text(107, 20.4, "prediction", fontsize=5.5, color=HELD, ha="center", va="top")
cv.text(76, 13, r"$\hat{y}_{k,g}=\sum_s\, w_s\, \mathrm{LFC}_{s,k,g}$", fontsize=7.5, ha="center", va="center")
cv.text(76, 6.5, "T set by leave-one-source-out on the sources only", fontsize=5.5, color=MUTED, ha="center", va="center")
cv.text(76, 2.6, "Baseline U: equal weights", fontsize=5.5, color=MUTED, ha="center", va="center")

# ---------- e: pre-registered readout shell
letter(fig, 124, 52, "e")
fig.text(127 / 180, 50.2 / 112, "Pre-registered readout", fontsize=6.5, ha="left", va="top")
for j, (fold, x) in enumerate([("Jurkat", 136), ("HepG2", 160)]):
    axe = axes_mm(fig, x, 13, 19, 30)
    axe.axvspan(0, 0.05, color=mpl.colors.to_rgba(MODEL, 0.07), lw=0)
    axe.axvline(0, color=INK, lw=0.5)
    for i in range(3):
        axe.plot([-0.02, 0.03], [i, i], color=FAINT, lw=1.0, ls=(0, (2, 1.5)))
        axe.plot([0.005], [i], "o", ms=2.2, mfc="white", mec=FAINT, mew=0.6)
    axe.set_ylim(-0.7, 2.7); axe.set_xlim(-0.04, 0.05); axe.invert_yaxis()
    axe.set_yticks(range(3)); axe.set_yticklabels(["R1 scGPT", "R2 Geneformer", "R3 State"] if j == 0 else [])
    axe.set_xticks([-0.02, 0, 0.02, 0.04]); axe.set_xticklabels(["−.02", "0", ".02", ".04"])
    axe.set_title(fold, fontsize=6.5, color=LINE_COL[fold], fontweight="bold", pad=2)
    axe.spines["left"].set_visible(False); axe.tick_params(axis="y", length=0)
fig.text(157.5 / 180, 6.8 / 112, "Δρ vs raw profile (R0), 98.3% CI", fontsize=6, ha="center", va="center")
fig.text(157.5 / 180, 2.6 / 112, "helps only if the interval clears 0 on both lines", fontsize=5.5, color=MUTED, ha="center", va="center")
fig.text(178.5 / 180, 50.2 / 112, "results pending", fontsize=5.5, color=FAINT, style="italic", ha="right", va="top")

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"v2_fmbench.{ext}", dpi=600 if ext == "png" else None)
print("ok")
