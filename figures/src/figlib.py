"""Small SVG drawing kit for publication-style schematic figures (Nature-like: Arial, flat colour, thin strokes)."""
import numpy as np, html

FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
INK, MUTED, FAINT, FRAME, PANEL = "#1F2933", "#5B6670", "#8A949E", "#C3C9D0", "#F6F7F9"
TEAL, TEAL_L = "#2A8C82", "#DCEFEC"          # source data / known
CORAL, CORAL_L = "#D9603B", "#FBE4DA"        # held-out / the thing being tested
VIOLET, VIOLET_L = "#5B5BB7", "#E4E3F6"      # models / representations
AMBER, AMBER_L = "#C98A1B", "#FBEFD6"        # prediction / combination
NAVY, NAVY_L = "#2F4A63", "#E1E8EF"          # evaluation
BLUE_D, RED_D = "#3B6FB6", "#C8414E"         # diverging heatmap ends

class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.e = w, h, []
    def add(self, s): self.e.append(s); return self
    def out(self, path, title, desc):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" font-family="{FONT}"><title>{html.escape(title)}</title><desc>{html.escape(desc)}</desc>'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M1 1L9 5L1 9z" fill="{MUTED}"/></marker>'
                '<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="7" height="7" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="7" stroke="{CORAL}" stroke-width="1.6" opacity="0.45"/></pattern></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>')
        open(path, "w").write(head + "".join(self.e) + "</svg>")

def esc(s): return html.escape(str(s))

def text(s, x, y, t, size=19, color=INK, weight=400, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ''
    s.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
          f'dominant-baseline="central"{st}>{esc(t)}</text>')

def rich(s, x, y, parts, size=19, anchor="start"):
    """parts: list of (text, color, weight[, 'sub']); subscripts drop by dy and the next part returns."""
    out, down = [], 0.0
    for p in parts:
        t, c, w = p[:3]; sub = len(p) > 3 and p[3] == "sub"
        d = (size * 0.3 if sub else 0.0) - down; down = size * 0.3 if sub else 0.0
        fs = f' font-size="{size*0.68:.1f}"' if sub else ''
        dy = f' dy="{d:.1f}"' if d else ''
        out.append(f'<tspan fill="{c}" font-weight="{w}"{fs}{dy}>{esc(t)}</tspan>')
    s.add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" dominant-baseline="central">{"".join(out)}</text>')

def rect(s, x, y, w, h, fill="none", stroke=FRAME, sw=1.2, rx=6, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

def line(s, x1, y1, x2, y2, color=MUTED, sw=1.6, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#ah)"' if arrow else ''
    s.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

def path(s, d, color=MUTED, sw=1.6, arrow=True, dash=None, fill="none"):
    dd = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#ah)"' if arrow else ''
    s.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{dd}{m}/>')

def column(s, x, y, w, h, title, sub=None, letter=None):
    rect(s, x, y, w, h, fill="#FFFFFF", stroke=FRAME, sw=1.4, rx=18)
    off = 0
    if letter: text(s, x + 22, y + 34, letter, size=28, weight=700); off = 30
    text(s, x + 22 + off, y + 34, title, size=26, weight=700)
    if sub: text(s, x + 22, y + 64, sub, size=17, color=MUTED)

def panel_letter(s, x, y, L): text(s, x, y, L, size=34, weight=700)

def cells(s, cx, cy, color, light, n=7, r=11, seed=0):
    """a small cluster of cells (circles with nucleus dots)."""
    rng = np.random.default_rng(seed)
    pts = [(0, 0)] + [(1.75 * r * np.cos(a), 1.75 * r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n - 1, endpoint=False) + rng.uniform(-0.25, 0.25, n - 1)]
    for dx, dy in pts:
        s.add(f'<circle cx="{cx+dx:.1f}" cy="{cy+dy:.1f}" r="{r}" fill="{light}" stroke="{color}" stroke-width="1.3"/>')
        s.add(f'<circle cx="{cx+dx+2:.1f}" cy="{cy+dy-1:.1f}" r="{r*0.32:.1f}" fill="{color}" opacity="0.75"/>')

def heat(s, x, y, nr, nc, cell, seed=0, hatch=False, outline=INK, scale=1.0, mask=None):
    """illustrative heatmap (rows = knockdowns, cols = genes); diverging blue-white-red."""
    rng = np.random.default_rng(seed)
    if hatch:
        rect(s, x, y, nc * cell, nr * cell, fill="url(#hatch)", stroke=CORAL, sw=1.2, rx=2)
        return
    V = np.tanh(rng.normal(0, 0.9, (nr, nc)) * scale)
    for i in range(nr):
        for j in range(nc):
            v = V[i, j]; c = mix("#FFFFFF", RED_D if v > 0 else BLUE_D, min(1, abs(v)))
            s.add(f'<rect x="{x + j*cell}" y="{y + i*cell}" width="{cell}" height="{cell}" fill="{c}"/>')
    rect(s, x, y, nc * cell, nr * cell, stroke=outline, sw=1.0, rx=0)

def vec(s, x, y, n, cell, color, seed=0, vertical=False):
    rng = np.random.default_rng(seed); v = rng.uniform(0.15, 1, n)
    for k in range(n):
        c = mix("#FFFFFF", color, v[k])
        xx, yy = (x, y + k * cell) if vertical else (x + k * cell, y)
        s.add(f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" fill="{c}" stroke="#FFFFFF" stroke-width="0.8"/>')
    w, h = (cell, n * cell) if vertical else (n * cell, cell)
    rect(s, x, y, w, h, stroke=color, sw=1.0, rx=1)

def transformer(s, x, y, w, h, color, light, label=None):
    for k in (2, 1, 0):
        rect(s, x + 7 * k, y - 7 * k, w, h, fill=light if k else color, stroke=color, sw=1.2, rx=7)
    if label: text(s, x + w / 2, y + h / 2, label, size=17, color="#FFFFFF", weight=700, anchor="middle")

def chip(s, x, y, t, color, light, size=16, pad=10, h=28, weight=400):
    w = textw(t, size) + 2 * pad
    rect(s, x, y, w, h, fill=light, stroke=color, sw=1.0, rx=h / 2)
    text(s, x + w / 2, y + h / 2, t, size=size, color=color, weight=weight, anchor="middle")
    return w

def textw(t, size): return 0.53 * size * len(t)

def mix(a, b, f):
    a = [int(a[i:i+2], 16) for i in (1, 3, 5)]; b = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))

def bars(s, x, y0, values, bw, gap, hmax, colors, base=True):
    for k, (v, c) in enumerate(zip(values, colors)):
        h = v * hmax; s.add(f'<rect x="{x + k*(bw+gap)}" y="{y0 - h}" width="{bw}" height="{h}" fill="{c}" rx="1.5"/>')
    if base: line(s, x - 6, y0, x + len(values) * (bw + gap), y0, color=INK, sw=1.0, arrow=False)
