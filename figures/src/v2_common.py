"""Journal-style matplotlib setup: 180 mm width, Arial-metric type, thin lines."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Circle
import numpy as np

MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 6.5, "axes.linewidth": 0.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "legend.fontsize": 6, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600, "lines.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.default": "regular"})

INK, MUTED, FAINT, RULE = "#222222", "#6B6B6B", "#A6A6A6", "#D9D9D9"
LINE_COL = {"K562": "#1F6F6B", "HCT116": "#3E9C93", "HEK293T": "#6FB8AE", "H1_2025": "#5876A8", "H1_GSE295214": "#8FA7CF",
            "Jurkat": "#C8553D", "HepG2": "#E39A3B"}
LINE_LAB = {"K562": "K562", "HCT116": "HCT116", "HEK293T": "HEK293T", "H1_2025": "H1 (VCC 2025)", "H1_GSE295214": "H1 (GSE295214)",
            "Jurkat": "Jurkat", "HepG2": "HepG2"}
HELD = "#C8553D"; SRC = "#1F6F6B"; MODEL = "#4D4D9F"; WARM = "#C98A1B"

def canvas(fig, x, y, w, h):
    """an axes in figure millimetres (origin bottom-left), with mm data coordinates."""
    W, H = fig.get_size_inches() / MM
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_axis_off()
    return ax

def axes_mm(fig, x, y, w, h):
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H])

def letter(fig, x, y, L):
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, y / H, L, fontsize=8, fontweight="bold", va="top", ha="left")

def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=0.6):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=6))

def vec(ax, x, y, n, cell, color, seed=0, vertical=True):
    rng = np.random.default_rng(seed); v = rng.uniform(0.15, 1, n)
    for k in range(n):
        xx, yy = (x, y + k * cell) if vertical else (x + k * cell, y)
        ax.add_patch(Rectangle((xx, yy), cell, cell, facecolor=mpl.colors.to_rgba(color, v[k]), edgecolor="white", lw=0.25))
    w, h = (cell, n * cell) if vertical else (n * cell, cell)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=0.4))

def stage(fig, x, y, w, h, L, title, color=INK):
    """thin rounded stage frame with panel letter and a one-line stage title (v1 workflow, v2 styling)."""
    cv = canvas(fig, x, y, w, h); cv.set_zorder(-5)
    cv.add_patch(FancyBboxPatch((0.25, 0.25), w - 0.5, h - 0.5, boxstyle="round,pad=0,rounding_size=2.2", fc="#FBFBFC", ec=RULE, lw=0.6))
    cv.text(2.2, h - 3.2, L, fontsize=8, fontweight="bold", va="center", ha="left")
    cv.text(6.2, h - 3.2, title, fontsize=7, fontweight="bold", va="center", ha="left", color=color)
    return cv

def gutter_arrow(fig, x1, x2, y):
    cv = canvas(fig, x1, y - 2, x2 - x1, 4)
    cv.annotate("", (x2 - x1 - 0.2, 2), (0.2, 2), arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3", color=FAINT, lw=0.9, mutation_scale=7))
