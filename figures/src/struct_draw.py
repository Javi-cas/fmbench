"""Boltz-2 predicted structure (AMPK gamma-1 + AMP): projection, secondary structure from CA geometry, cartoon drawing."""
import numpy as np, matplotlib as mpl
from matplotlib.patches import Circle
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
def draw_struct(axs, scale=1.0, lw0=1.0):
    dz = np.array([Q[a_:b_, 2].mean() for a_, b_ in chunks]); dzn = (dz - dz.min()) / (np.ptp(dz) + 1e-9)
    WID = {"H": 4.6 * lw0, "E": 3.2 * lw0, "L": 1.0 * lw0}
    for ci in np.argsort(dz):
        a_, b_ = chunks[ci]; kd_ = ss[ri[a_]]; seg = Q[a_:b_]
        base = np.array(mpl.colors.to_rgb(afcol(plq[a_:b_].mean()))); shade = 0.7 + 0.3 * dzn[ci]
        fill = 1 - (1 - base) * shade; edge = base * 0.5
        if kd_ != "L":
            axs.plot(seg[:, 0], seg[:, 1], color=edge, lw=(WID[kd_] + 1.0) * scale, solid_capstyle="round", solid_joinstyle="round", zorder=2 + ci * 2)
        axs.plot(seg[:, 0], seg[:, 1], color=fill, lw=WID[kd_] * scale, solid_capstyle="round", solid_joinstyle="round", zorder=3 + ci * 2)
ZTOP = 10 + 2 * len(chunks)
ecol = {"C": "#909090", "O": "#E0473C", "N": "#3B5BD9", "P": "#F28C28"}
def draw_lig(axs, r=0.95):
    for k in np.argsort(L[:, 2]):
        axs.add_patch(Circle(L[k, :2], r, fc=ecol.get(el[k], "#909090"), ec="#333333", lw=0.25, zorder=ZTOP))
