import sys; sys.path.insert(0, "/home/claude/figs")
from v2_common import *
from matplotlib.collections import LineCollection

fig = plt.figure(figsize=(180 * MM, 92 * MM))
REG = "#C8553D"; DEC = "#8C8C8C"

# ---------- a: benchmark design matrix
enz = [("PRKAG1", ["AMP"], 0), ("FBP1", ["AMP"], 0), ("PKM", ["FBP"], 0), ("GLUD1", ["GTP", "ADP"], 1), ("PRPS1", ["ADP"], 1),
       ("CTPS1", ["CTP"], 0), ("IMPDH2", ["ATP"], 0), ("PPAT", ["AMP"], 0), ("G6PD", ["NADP$^+$"], 0), ("PFKL", ["ATP", "citrate"], 0)]
regs = ["AMP", "ADP", "ATP", "GTP", "CTP", "FBP", "NADP$^+$", "citrate"]
decs = ["3PG", "G3P", "sucrose", "trehalose", "succinate", "taurine"]
cols = regs + [None] + decs
ax = axes_mm(fig, 20, 10, 50, 58)
for i, (e, rr, itf) in enumerate(enz):
    for j, c in enumerate(cols):
        if c is None: continue
        if c in rr: ax.plot(j, i, "o", ms=5.2, mfc=REG, mec=REG, mew=0.5)
        elif c in decs: ax.plot(j, i, "o", ms=3.6, mfc=mpl.colors.to_rgba(DEC, 0.55), mec="none")
        else: ax.plot(j, i, "o", ms=1.2, mfc=RULE, mec="none")
ax.set_xlim(-0.7, len(cols) - 0.3); ax.set_ylim(len(enz) - 0.4, -0.6)
ax.set_yticks(range(len(enz))); ax.set_yticklabels([e + ("*" if itf else "") for e, _, itf in enz], fontsize=6)
ax.set_xticks([j for j, c in enumerate(cols) if c]); ax.set_xticklabels([c for c in cols if c], rotation=90, fontsize=5.8)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
for sp_ in ax.spines.values(): sp_.set_visible(False)
for i in range(len(enz)): ax.axhline(i, color=RULE, lw=0.3, zorder=0)
W, H = 180, 92
fig.text(20 / W, 85.2 / H, "known regulators", fontsize=6, color=REG, fontweight="bold", ha="left")
fig.text(51.5 / W, 85.2 / H, "decoys", fontsize=6, color=DEC, fontweight="bold", ha="left")
fig.text(2 / W, 5.4 / H, "* regulatory site at a subunit interface", fontsize=5.5, color=MUTED, ha="left")
fig.text(2 / W, 2.2 / H, "12 regulator pairs × 6 decoys = 72 complexes", fontsize=5.5, color=MUTED, ha="left")
letter(fig, 0, 91, "a")

# ---------- b: real Boltz-2 structure, AMPK gamma-1 + AMP (CPU test run)
st = np.load("data_struct.npz"); ca, pl, het, el = st["ca"], st["plddt"][:len(st["ca"])] * 100, st["het"], st["el"].astype(str)
c0 = ca.mean(0); lc = het.mean(0); z = lc - c0; z /= np.linalg.norm(z)
X = ca - c0; U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
xax = Vt[0] - (Vt[0] @ z) * z; xax /= np.linalg.norm(xax); yax = np.cross(z, xax)
R = np.stack([xax, yax, z]); P = X @ R.T; L = (het - c0) @ R.T
n = len(P); d = lambda i, j: np.linalg.norm(ca[i] - ca[j])
ss = np.array(["L"] * n)
hel = np.zeros(n, bool)
for i in range(n - 4):
    if 4.6 < d(i, i + 3) < 5.8 and 5.4 < d(i, i + 4) < 6.9: hel[i:i + 5] = True
strd = np.zeros(n, bool)
for i in range(1, n - 2):
    if not hel[i] and d(i - 1, i + 1) > 6.3 and d(i, i + 2) > 6.3: strd[i - 1:i + 2] = True
def runs(mask, minlen):
    out, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]: j += 1
            if j - i + 1 >= minlen: out.append((i, j))
            i = j + 1
        else: i += 1
    return out
HR = runs(hel, 6); Sx = [r for r in runs(strd & ~hel, 4)]
for i, j in HR: ss[i:j + 1] = "H"
for i, j in Sx: ss[i:j + 1] = "E"
# smooth path: helices follow their axis (running mean of 4 CA), other residues follow CA
Pa = P.copy()
for i, j in HR:
    for k in range(i, j + 1):
        lo, hi = max(i, k - 2), min(j, k + 2); Pa[k] = P[lo:hi + 1].mean(0)
from scipy.interpolate import CubicSpline
t = np.arange(n); tt = np.linspace(0, n - 1, n * 8); Q = CubicSpline(t, Pa)(tt)
ri = np.round(tt).astype(int).clip(0, n - 1); plq = np.interp(tt, t, pl)
def afcol(v): return "#0053D6" if v > 90 else "#65CBF3" if v > 70 else "#FFDB13" if v > 50 else "#FF7D45"
# chunks of one secondary-structure type (loops cut every 10 residues) drawn as continuous strokes, back to front
chunks, k0 = [], 0
for k in range(1, len(Q) + 1):
    if k == len(Q) or ss[ri[k]] != ss[ri[k0]] or (ss[ri[k0]] == "L" and ri[k] - ri[k0] >= 10):
        chunks.append((k0, min(k + 1, len(Q)))); k0 = k
def draw_struct(axs, scale=1.0):
    dz = np.array([Q[a_:b_, 2].mean() for a_, b_ in chunks]); dzn = (dz - dz.min()) / (np.ptp(dz) + 1e-9)
    WID = {"H": 4.6, "E": 3.2, "L": 1.0}
    for ci in np.argsort(dz):
        a_, b_ = chunks[ci]; kd_ = ss[ri[a_]]; seg = Q[a_:b_]
        base = np.array(mpl.colors.to_rgb(afcol(plq[a_:b_].mean()))); shade = 0.7 + 0.3 * dzn[ci]
        fill = 1 - (1 - base) * shade; edge = base * 0.5
        if kd_ != "L":
            axs.plot(seg[:, 0], seg[:, 1], color=edge, lw=(WID[kd_] + 1.0) * scale, solid_capstyle="round", solid_joinstyle="round", zorder=2 + ci * 2)
        axs.plot(seg[:, 0], seg[:, 1], color=fill, lw=WID[kd_] * scale, solid_capstyle="round", solid_joinstyle="round", zorder=3 + ci * 2)
ZTOP = 10 + 2 * len(chunks)

print("helices", len(HR), "strands", len(Sx))
ecol = {"C": "#909090", "O": "#E0473C", "N": "#3B5BD9", "P": "#F28C28"}
def draw_lig(axs, r=0.95):
    for k in np.argsort(L[:, 2]):
        axs.add_patch(Circle(L[k, :2], r, fc=ecol.get(el[k], "#909090"), ec="#333333", lw=0.25, zorder=ZTOP))
axs = axes_mm(fig, 82, 40, 54, 40); axs.set_aspect("equal"); axs.set_axis_off()
draw_struct(axs); draw_lig(axs)
pad = 3; axs.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad); axs.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
lc2 = L[:, :2].mean(0); R_ = 11
axs.add_patch(Rectangle(lc2 - R_, 2 * R_, 2 * R_, fill=False, ec=INK, lw=0.5, ls=(0, (2, 1.5)), zorder=ZTOP + 1))
axz = axes_mm(fig, 94, 13, 30, 25); axz.set_aspect("equal"); axz.set_xticks([]); axz.set_yticks([])
for sp_ in axz.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.5); sp_.set_linestyle((0, (2, 1.5)))
draw_struct(axz, scale=1.6); draw_lig(axz, r=0.75); axz.set_xlim(lc2[0] - R_, lc2[0] + R_); axz.set_ylim(lc2[1] - R_ * 0.83, lc2[1] + R_ * 0.83)
axz.text(lc2[0] - R_ + 1, lc2[1] + R_ * 0.83 - 1.2, "AMP pocket", fontsize=5.8, fontweight="bold", va="top", zorder=ZTOP + 2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
fig.text(86 / W, 85.2 / H, "Boltz-2 prediction: AMPK γ1 (PRKAG1) + AMP", fontsize=6.5, ha="left")
for k, (lab_, c_) in enumerate([("> 90", "#0053D6"), ("70–90", "#65CBF3"), ("50–70", "#FFDB13"), ("< 50", "#FF7D45")]):
    x = 95 + k * 10
    fig.patches.append(Rectangle((x / W, 6.2 / H), 2.6 / W, 2.2 / H, transform=fig.transFigure, fc=c_, ec="none"))
    fig.text((x + 3.4) / W, 7.3 / H, lab_, fontsize=5.5, va="center")
fig.text(85 / W, 7.3 / H, "pLDDT", fontsize=5.5, va="center", color=MUTED)
fig.text(85 / W, 2.6 / H, "single chain, one diffusion sample, CPU", fontsize=5.5, color=MUTED)
letter(fig, 80, 91, "b")

# ---------- c: per-pair score (schematic)
cv = canvas(fig, 140, 50, 39, 34)
letter(fig, 136, 91, "c")
cv.text(0, 32.5, "Score per pair", fontsize=6.5, ha="left", va="center")
cv.text(39, 32.5, "schematic", fontsize=5.5, color=FAINT, style="italic", ha="right", va="center")
cv.plot([2, 37], [18, 18], color=INK, lw=0.5)
for xx in (2, 37): cv.plot([xx, xx], [17.3, 18.7], color=INK, lw=0.5)
cv.text(2, 15.8, "0", fontsize=5.5, ha="center", va="top"); cv.text(37, 15.8, "1", fontsize=5.5, ha="center", va="top")
cv.text(19.5, 12.6, "binder probability", fontsize=5.5, color=MUTED, ha="center", va="top")
dx = [5.5, 9, 13, 17.5, 22, 31]
for x in dx: cv.add_patch(Circle((x, 18), 0.95, fc=mpl.colors.to_rgba(DEC, 0.6), ec="none", zorder=3))
cv.add_patch(Circle((27, 18), 1.3, fc=REG, ec=REG, zorder=4))
cv.text(27, 21.2, "regulator", fontsize=5.5, color=REG, ha="center", va="bottom", fontweight="bold")
cv.plot([5.5, 5.5, 22, 22], [22.5, 23.5, 23.5, 22.5], color=MUTED, lw=0.5)
cv.text(13.75, 24.4, "5 of 6 decoys beaten = 0.83", fontsize=5.5, color=INK, ha="center", va="bottom")
cv.text(19.5, 6.5, "chance = 0.5 (uniform rank among 7)", fontsize=5.5, color=MUTED, ha="center", va="center")

# ---------- d: real null for the mean over 12 pairs
letter(fig, 136, 46, "d")
axn = axes_mm(fig, 142, 16, 36, 22)
rng = np.random.default_rng(20261023); null = rng.integers(0, 7, size=(100000, 12)).mean(1) / 6
crit = float(np.quantile(null, 0.95))
vals, counts = np.unique(np.round(null, 4), return_counts=True)
axn.bar(vals, counts / counts.sum(), width=1 / 72 * 0.8, color=mpl.colors.to_rgba(DEC, 0.6), lw=0)
axn.axvline(crit, color=REG, lw=0.7, ls=(0, (2.5, 1.5)))
top = (counts / counts.sum()).max(); axn.set_ylim(0, top * 1.55)
axn.text(crit + 0.012, top * 1.5, f"p < 0.05 if\nmean ≥ {crit:.2f}", fontsize=5.5, color=REG, va="top", ha="left")
axn.set_xlim(0.15, 0.95); axn.set_xticks([0.25, 0.5, 0.75]); axn.set_yticks([])
axn.spines["left"].set_visible(False)
axn.set_xlabel("mean score over 12 pairs, null", fontsize=5.8, labelpad=1.5)
fig.text(140 / W, 42.6 / H, "Aggregate test", fontsize=6.5, ha="left", va="center")
fig.text(140 / W, 5.4 / H, "Also reported: phosphorylated decoys only;", fontsize=5.5, color=MUTED)
fig.text(140 / W, 2.2 / H, "within-chain vs interface sites", fontsize=5.5, color=MUTED)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"v2_boltz.{ext}", dpi=600 if ext == "png" else None)
print("ok", round(crit, 3), "mean pLDDT", round(float(pl.mean()), 1))
