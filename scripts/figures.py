#!/usr/bin/env python3
"""figures — regenerate the diagrams under public/img/ for six notes.

    python3 scripts/figures.py

Source of truth for the six generated SVGs; edit here and re-run rather than
hand-patching the files, which carry hundreds of computed curve points.

Palette is lifted from src/styles/global.css day-mode tokens, so the plates sit
flush on the page in DAY and read as developed prints in NIGHT.

Type is sized for the *rendered* width. The prose column is 512px and these are
authored 900 wide, so everything on the canvas lands at ~0.57x — a 16px label
reads as 9px, which is about the size of the site's own mono micro-labels.
"""
import math
import os
import random

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "img")

# AZUR day-mode tokens. PAPER is not `--paper`: it sits two-thirds of the way
# from `--bg` to `--paper`, so a plate reads as a sheet laid on the page rather
# than as another band of it.
PAPER = "#eaeff4"
INK = "#101c26"
SOFT = "#475666"
FAINT = "#788999"
LINE = "rgba(16,28,38,.14)"
LINE2 = "rgba(16,28,38,.26)"
SEAL = "#1f6b9e"

MONO = "ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace"
SERIF = "Spectral,'Iowan Old Style',Georgia,'Times New Roman',serif"

STYLE = f"""
  text{{fill:{INK};}}
  .m{{font-family:{MONO};letter-spacing:.1em;text-transform:uppercase;}}
  .mn{{font-family:{MONO};letter-spacing:.01em;}}
  .s{{font-family:{SERIF};font-style:italic;}}
  .sr{{font-family:{SERIF};}}
  .lbl{{font-size:16px;fill:{FAINT};}}
  .ttl{{font-size:16px;fill:{SEAL};}}
  .ax{{stroke:{LINE2};stroke-width:1.4;fill:none;}}
  .hair{{stroke:{LINE};stroke-width:1.4;fill:none;}}
"""


def svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img">\n'
        f'<style>{STYLE}</style>\n'
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n{body}\n</svg>\n'
    )


def write(slug, name, w, h, body):
    d = os.path.join(OUT, slug)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(svg(w, h, body))
    print(f"{p}  {os.path.getsize(p)}b")


def pts(seq):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in seq)


# ─────────────────────────── 1 · square loss ────────────────────────────────
def square_loss():
    W, H = 900, 500
    x0, x1, ybase, ytop = 96, 852, 414, 84
    m = -0.3039

    def yl(x):
        return 360 + m * (x - x0)

    b = []
    b.append(f'<path class="ax" d="M{x0} {ytop} V{ybase} H{x1}"/>')
    b.append(f'<text class="s" font-size="22" x="{x1 + 10}" y="{ybase + 7}">x</text>')
    b.append(f'<text class="s" font-size="22" x="{x0 - 26}" y="{ytop + 4}">y</text>')

    random.seed(7)
    for gx in (150, 300, 340, 480, 530, 680, 700, 820):
        gy = yl(gx) + random.uniform(-30, 30)
        b.append(f'<circle cx="{gx}" cy="{gy:.1f}" r="3.4" fill="{FAINT}" opacity=".55"/>')

    b.append(
        f'<line x1="{x0}" y1="{yl(x0):.1f}" x2="{x1}" y2="{yl(x1):.1f}" '
        f'stroke="{INK}" stroke-width="1.8"/>'
    )
    b.append(
        f'<text class="sr" font-size="25" x="{x0 + 16}" y="{yl(x0 + 16) - 16:.1f}">'
        f'θ<tspan font-size="16" dy="-9">T</tspan><tspan class="s" dy="9">x</tspan></text>'
    )

    A, sd = 54, 24
    resid = {230: 34, 420: -23, 610: 19, 780: -31}
    for cx in (230, 420, 610, 780):
        cy = yl(cx)
        curve = [
            (cx + A * math.exp(-(t * t) / (2 * sd * sd)), cy + t)
            for t in [i * 1.44 - 72 for i in range(101)]
        ]
        b.append(
            f'<polyline points="{pts(curve)}" fill="{SEAL}" fill-opacity=".07" '
            f'stroke="{SEAL}" stroke-opacity=".5" stroke-width="1.3"/>'
        )
        b.append(
            f'<line x1="{cx}" y1="{cy - 72:.1f}" x2="{cx}" y2="{cy + 72:.1f}" '
            f'stroke="{LINE2}" stroke-width="1.2" stroke-dasharray="2 3"/>'
        )
        b.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="2.6" fill="{INK}"/>')
        oy = cy + resid[cx]
        b.append(
            f'<line x1="{cx}" y1="{cy:.1f}" x2="{cx}" y2="{oy:.1f}" '
            f'stroke="{SEAL}" stroke-width="1.8"/>'
        )
        b.append(f'<circle cx="{cx}" cy="{oy:.1f}" r="5" fill="{SEAL}"/>')

    cx, cy = 610, yl(610)
    b.append(
        f'<text class="sr" font-size="22" x="{cx + 11}" y="{cy + 20:.1f}">'
        f'<tspan class="s">ε</tspan><tspan font-size="15" dy="-9">(i)</tspan></text>'
    )

    # ±σ on the last bell, so the spread has a name
    cx, cy = 780, yl(780)
    for s in (-sd, sd):
        b.append(
            f'<line x1="{cx - 6}" y1="{cy + s:.1f}" x2="{cx + 6}" y2="{cy + s:.1f}" '
            f'stroke="{SOFT}" stroke-width="1.3"/>'
        )
    b.append(
        f'<line x1="{cx - 26}" y1="{cy - sd:.1f}" x2="{cx - 26}" y2="{cy + sd:.1f}" '
        f'stroke="{SOFT}" stroke-width="1.3"/>'
    )
    b.append(
        f'<text class="sr" font-size="20" text-anchor="end" x="{cx - 34}" '
        f'y="{cy + 7:.1f}" fill="{SOFT}">2σ</text>'
    )

    b.append(
        f'<text class="sr" font-size="25" x="{x0 - 16}" y="46">'
        f'<tspan class="s">y</tspan><tspan font-size="16" dy="-9">(i)</tspan>'
        f'<tspan dy="9"> | </tspan><tspan class="s">x</tspan>'
        f'<tspan font-size="16" dy="-9">(i)</tspan><tspan dy="9">'
        f' ∼ 𝒩( θ</tspan><tspan font-size="16" dy="-9">T</tspan>'
        f'<tspan class="s" dy="9">x</tspan><tspan font-size="16" dy="-9">(i)</tspan>'
        f'<tspan dy="9">, σ</tspan><tspan font-size="16" dy="-9">2</tspan>'
        f'<tspan dy="9"> )</tspan></text>'
    )
    b.append(f'<line class="hair" x1="{x0 - 16}" y1="{H - 58}" x2="{x1}" y2="{H - 58}"/>')
    b.append(
        f'<text class="m lbl" x="{x0 - 16}" y="{H - 26}">'
        f'maximise ∏ p( y | x ; θ )</text>'
    )
    b.append(
        f'<text class="m" font-size="19" x="{x0 + 320}" y="{H - 24}" fill="{SEAL}">⇔</text>'
    )
    b.append(
        f'<text class="m lbl" x="{x0 + 372}" y="{H - 26}">'
        f'minimise ½ ∑ ( y − θ'
        f'<tspan font-size="12" dy="-6">T</tspan><tspan dy="6">x )</tspan>'
        f'<tspan font-size="12" dy="-6">2</tspan></text>'
    )
    write("square-loss", "gaussian-noise.svg", W, H, "\n".join(b))


# ────────────────────────── 2 · quadratic forms ─────────────────────────────
def semidefinite():
    W, H = 900, 400
    b = []
    # `A ≻ 0` / `A ⪰ 0` are spelled out: both glyphs are missing from most system
    # faces and fall back to `>` / `≥`, which say something else entirely.
    panels = [
        (58, "positive definite", "λ₁, λ₂ &gt; 0", "bowl"),
        (338, "positive semidefinite", "λ₁ &gt; 0,  λ₂ = 0", "trough"),
        (618, "indefinite", "λ₁ &gt; 0 &gt; λ₂", "saddle"),
    ]
    S = 224
    top = 92
    rot = -28

    for px, title, eig, kind in panels:
        cx, cy = px + S / 2, top + S / 2
        b.append(f'<text class="m ttl" x="{px}" y="{top - 38}">{title}</text>')
        b.append(f'<text class="m lbl" x="{px}" y="{top - 16}">{eig}</text>')
        b.append(
            f'<rect x="{px}" y="{top}" width="{S}" height="{S}" fill="none" '
            f'stroke="{LINE}" stroke-width="1.4"/>'
        )
        b.append(
            f'<defs><clipPath id="c{px}"><rect x="{px}" y="{top}" '
            f'width="{S}" height="{S}"/></clipPath></defs>'
        )
        b.append(f'<g clip-path="url(#c{px})">')
        b.append(
            f'<line class="hair" x1="{px}" y1="{cy}" x2="{px + S}" y2="{cy}"/>'
            f'<line class="hair" x1="{cx}" y1="{top}" x2="{cx}" y2="{top + S}"/>'
        )

        if kind == "bowl":
            for i, f in enumerate((1.0, 0.72, 0.46, 0.24)):
                b.append(
                    f'<ellipse cx="{cx}" cy="{cy}" rx="{96 * f:.1f}" ry="{58 * f:.1f}" '
                    f'transform="rotate({rot} {cx} {cy})" fill="{SEAL}" '
                    f'fill-opacity="{0.04 if i == 0 else 0}" stroke="{SOFT}" '
                    f'stroke-width="{1.7 if i == 0 else 1.3}" stroke-opacity=".75"/>'
                )
            b.append(f'<circle cx="{cx}" cy="{cy}" r="3.4" fill="{SEAL}"/>')

        elif kind == "trough":
            th = math.radians(rot)
            ux, uy = math.cos(th), math.sin(th)      # the null direction
            nx, ny = -uy, ux
            for i, d in enumerate((0, 32, 32, 62, 62, 94, 94)):
                s = 0 if d == 0 else (1 if i % 2 else -1)
                ox, oy = cx + nx * d * s, cy + ny * d * s
                b.append(
                    f'<line x1="{ox - ux * 160:.1f}" y1="{oy - uy * 160:.1f}" '
                    f'x2="{ox + ux * 160:.1f}" y2="{oy + uy * 160:.1f}" '
                    f'stroke="{SEAL if d == 0 else SOFT}" '
                    f'stroke-width="{2 if d == 0 else 1.3}" stroke-opacity=".8"/>'
                )
            b.append(
                f'<rect x="{px + S - 186}" y="{top + S - 34}" width="180" height="26" '
                f'fill="{PAPER}"/>'
            )
            b.append(
                f'<text class="m lbl" text-anchor="end" x="{px + S - 10}" '
                f'y="{top + S - 14}" fill="{SEAL}">xᵀAx = 0 here</text>'
            )

        else:  # saddle
            a, bb = 34, 26
            th = math.radians(rot)
            ca, sa = math.cos(th), math.sin(th)

            def rp(x, y):
                return (cx + x * ca - y * sa, cy + x * sa + y * ca)

            for sgn in (1, -1):
                tri = [rp(0, 0), rp(-190, sgn * 190 * bb / a), rp(190, sgn * 190 * bb / a)]
                b.append(f'<polygon points="{pts(tri)}" fill="{SEAL}" fill-opacity=".06"/>')
            for k in (-1, 1):
                for sgn in (1, -1):
                    for scale in (1.0, 1.9):
                        cur = []
                        for i in range(61):
                            t = -1.9 + i * (3.8 / 60)
                            if k == 1:
                                x, y = sgn * a * scale * math.cosh(t), bb * scale * math.sinh(t)
                            else:
                                x, y = a * scale * math.sinh(t), sgn * bb * scale * math.cosh(t)
                            cur.append(rp(x, y))
                        b.append(
                            f'<polyline points="{pts(cur)}" fill="none" '
                            f'stroke="{SEAL if k == -1 else SOFT}" stroke-width="1.3" '
                            f'stroke-opacity=".8"/>'
                        )
            for sgn in (1, -1):
                p1, p2 = rp(-190, -sgn * 190 * bb / a), rp(190, sgn * 190 * bb / a)
                b.append(
                    f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" '
                    f'y2="{p2[1]:.1f}" stroke="{LINE2}" stroke-width="1.3" '
                    f'stroke-dasharray="3 4"/>'
                )
            b.append(f'<circle cx="{cx}" cy="{cy}" r="3.4" fill="{INK}"/>')

        b.append("</g>")
        note, seal = {
            "bowl": ("xᵀAx &gt; 0 unless x = 0", False),
            "trough": ("xᵀAx ≥ 0, flat along p₂", False),
            "saddle": ("xᵀAx &lt; 0 somewhere", True),
        }[kind]
        # the last note is right-aligned to its panel, so it does not run up
        # against the middle panel's note
        anchor = ' text-anchor="end"' if kind == "saddle" else ""
        nx = px + S if kind == "saddle" else px
        b.append(
            f'<text class="m lbl" x="{nx}" y="{top + S + 28}"{anchor}'
            f'{f" fill={chr(34)}{SEAL}{chr(34)}" if seal else ""}>{note}</text>'
        )

    b.append(f'<line class="hair" x1="58" y1="{H - 40}" x2="842" y2="{H - 40}"/>')
    b.append(
        f'<text class="m lbl" x="58" y="{H - 14}">'
        f'level sets of f(x) = xᵀAx · convex unless an eigenvalue is negative</text>'
    )
    write("semidefinite", "quadratic-forms.svg", W, H, "\n".join(b))


# ──────────────────────────────── 3 · smote ─────────────────────────────────
def smote():
    W, H = 900, 490
    b = []
    bx0, by0, bx1, by1 = 58, 40, 842, 386
    b.append(
        f'<rect x="{bx0}" y="{by0}" width="{bx1 - bx0}" height="{by1 - by0}" '
        f'fill="none" stroke="{LINE}" stroke-width="1.4"/>'
    )

    random.seed(19)
    for _ in range(150):
        x = random.gauss(520, 150)
        y = random.gauss(186, 72)
        if bx0 + 8 < x < bx1 - 8 and by0 + 8 < y < by1 - 8:
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{FAINT}" opacity=".5"/>')

    minority = [(196, 300), (238, 262), (262, 322), (300, 286), (176, 246),
                (330, 330), (208, 348), (286, 226), (150, 306)]
    a = (238, 262)
    nb = sorted((p for p in minority if p != a),
                key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)[:5]
    r_k = max(math.dist(a, p) for p in nb)

    b.append(
        f'<circle cx="{a[0]}" cy="{a[1]}" r="{r_k:.1f}" fill="{SEAL}" fill-opacity=".045" '
        f'stroke="{SEAL}" stroke-opacity=".45" stroke-width="1.3" stroke-dasharray="4 5"/>'
    )
    for p in nb:
        b.append(
            f'<line x1="{a[0]}" y1="{a[1]}" x2="{p[0]}" y2="{p[1]}" stroke="{SEAL}" '
            f'stroke-width="1.3" stroke-opacity=".55" stroke-dasharray="4 4"/>'
        )
    for p in minority:
        b.append(f'<circle cx="{p[0]}" cy="{p[1]}" r="6.4" fill="{SEAL}"/>')

    bp = nb[0]
    b.append(
        f'<line x1="{a[0]}" y1="{a[1]}" x2="{bp[0]}" y2="{bp[1]}" stroke="{SEAL}" '
        f'stroke-width="2"/>'
    )
    for p, u in zip(nb, [0.62, 0.35, 0.48, 0.71, 0.28]):
        sx, sy = a[0] + u * (p[0] - a[0]), a[1] + u * (p[1] - a[1])
        b.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{6.6 if p == bp else 5.6}" '
            f'fill="{PAPER}" stroke="{SEAL}" stroke-width="1.8"/>'
        )
    b.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="11" fill="none" stroke="{SEAL}" stroke-width="1.6"/>')

    b.append(f'<text class="s" font-size="24" text-anchor="middle" '
             f'x="{a[0] - 4}" y="{a[1] - 24}">a</text>')
    b.append(f'<rect x="{bp[0] - 30}" y="{bp[1] - 10}" width="22" height="24" fill="{PAPER}"/>')
    b.append(f'<text class="s" font-size="24" x="{bp[0] - 26}" y="{bp[1] + 8}">b</text>')

    ax0, ay0 = 424, 296
    b.append(f'<rect x="{ax0 - 14}" y="{ay0 - 32}" width="420" height="84" fill="{PAPER}"/>')
    b.append(
        f'<text class="sr" font-size="25" x="{ax0}" y="{ay0}">'
        f'<tspan class="s">a</tspan> + u(<tspan class="s">b</tspan> − '
        f'<tspan class="s">a</tspan>),&#160;&#160; u ∼ U(0, 1)</text>'
    )
    b.append(f'<text class="m lbl" x="{ax0}" y="{ay0 + 30}">'
             f'the new sample lands on the segment</text>')
    mid = ((a[0] + bp[0]) / 2, (a[1] + bp[1]) / 2)
    b.append(
        f'<path d="M{mid[0] - 8:.0f} {mid[1] + 8:.0f} C 300 360 366 336 {ax0 - 20} {ay0 - 8}" '
        f'fill="none" stroke="{LINE2}" stroke-width="1.3"/>'
    )

    ly = H - 34
    b.append(f'<line class="hair" x1="{bx0}" y1="{ly - 30}" x2="{bx1}" y2="{ly - 30}"/>')
    items = [
        (bx0 + 6, f'<circle cx="{bx0 + 6}" cy="{ly - 5}" r="3.4" fill="{FAINT}" opacity=".6"/>',
         "majority"),
        (bx0 + 190, f'<circle cx="{bx0 + 190}" cy="{ly - 5}" r="6.4" fill="{SEAL}"/>', "minority"),
        (bx0 + 374, f'<circle cx="{bx0 + 374}" cy="{ly - 5}" r="5.6" fill="{PAPER}" '
                    f'stroke="{SEAL}" stroke-width="1.8"/>', "synthesised"),
        (bx0 + 588, f'<circle cx="{bx0 + 588}" cy="{ly - 5}" r="9" fill="none" '
                    f'stroke="{SEAL}" stroke-width="1.3" stroke-dasharray="3 4"/>', "k = 5 nearest"),
    ]
    for x, glyph, label in items:
        b.append(glyph)
        b.append(f'<text class="m lbl" x="{x + 20}" y="{ly}">{label}</text>')
    write("smote", "smote-interpolation.svg", W, H, "\n".join(b))


# ─────────────────────────────── 4 · VIF ────────────────────────────────────
def vif():
    W, H = 900, 470
    x0, x1, ybase, ytop = 128, 828, 380, 76
    vmax = 20

    def X(r2):
        return x0 + (x1 - x0) * (r2 / 0.95)

    def Y(v):
        return ybase - (ybase - ytop) * ((v - 1) / (vmax - 1))

    b = []
    b.append(
        f'<rect x="{x0}" y="{Y(vmax):.1f}" width="{x1 - x0}" height="{Y(5) - Y(vmax):.1f}" '
        f'fill="{SEAL}" fill-opacity=".1"/>'
    )
    b.append(
        f'<rect x="{x0}" y="{Y(5):.1f}" width="{x1 - x0}" height="{Y(1) - Y(5):.1f}" '
        f'fill="{FAINT}" fill-opacity=".05"/>'
    )
    b.append(f'<path class="ax" d="M{x0} {ytop} V{ybase} H{x1}"/>')

    for v in (1, 5, 10, 15, 20):
        y = Y(v)
        b.append(f'<line class="hair" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        b.append(f'<text class="mn lbl" text-anchor="end" x="{x0 - 12}" y="{y + 5.5:.1f}">{v}</text>')
    for r2, lab in ((0, "0"), (0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"),
                    (0.8, "0.8"), (0.95, "0.95")):
        x = X(r2)
        b.append(f'<line class="ax" x1="{x:.1f}" y1="{ybase}" x2="{x:.1f}" y2="{ybase + 7}"/>')
        b.append(
            f'<text class="mn lbl" text-anchor="middle" x="{x:.1f}" y="{ybase + 27}">{lab}</text>'
        )

    curve = []
    r2 = 0.0
    while r2 <= 0.9505:
        curve.append((X(r2), Y(1 / (1 - r2))))
        r2 += 0.0025
    b.append(f'<polyline points="{pts(curve)}" fill="none" stroke="{INK}" stroke-width="2.2"/>')

    xc = X(0.8)
    b.append(
        f'<line x1="{xc:.1f}" y1="{ybase}" x2="{xc:.1f}" y2="{Y(5):.1f}" stroke="{SEAL}" '
        f'stroke-width="1.3" stroke-dasharray="3 4"/>'
    )
    b.append(f'<circle cx="{xc:.1f}" cy="{Y(5):.1f}" r="5.5" fill="{SEAL}"/>')
    b.append(
        f'<text class="m lbl" text-anchor="end" x="{xc - 16:.1f}" y="{Y(5) - 14:.1f}" '
        f'fill="{SEAL}">Rᵢ² = 0.80 → VIFᵢ = 5</text>'
    )

    b.append(f'<text class="m lbl" x="{x0 + 14}" y="{Y(17.6):.1f}" fill="{SEAL}">highly correlated</text>')
    b.append(f'<text class="m lbl" x="{x0 + 14}" y="{Y(3.9):.1f}">moderately correlated</text>')
    b.append(f'<text class="m lbl" x="{x0 + 14}" y="{Y(1) - 12:.1f}">not correlated</text>')

    b.append(
        f'<text class="m lbl" text-anchor="middle" x="{(x0 + x1) / 2:.0f}" y="{ybase + 56}">'
        f'Rᵢ² · how much of Xᵢ the other predictors explain</text>'
    )
    b.append(
        f'<text class="m lbl" transform="translate({x0 - 66},{(ytop + ybase) / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle">VIFᵢ = 1 / (1 − Rᵢ²)</text>'
    )
    b.append(
        f'<text class="sr" font-size="25" x="{x0 - 66}" y="{ytop - 30}">'
        f'var(<tspan class="s">β̂</tspan>ᵢ) = '
        f'σ² / [(n−1)·var(<tspan class="s">X</tspan>ᵢ)] '
        f'× <tspan fill="{SEAL}">VIFᵢ</tspan></text>'
    )
    write("collinearity", "vif-curve.svg", W, H, "\n".join(b))


# ─────────────────────────────── 5 · pickle ─────────────────────────────────
def pickle_fig():
    W, H = 900, 470
    b = []
    # The gap between the columns has to hold `pickle.dump()` at full size —
    # 148px of it — or the label sets straight over the right-hand card.
    L, LW = 58, 276              # source column
    R, RW = 512, 330             # written-out column

    def card(x, y, w, h, seal=False):
        c = SEAL if seal else LINE2
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
                f'stroke="{c}" stroke-opacity="{".55" if seal else "1"}" stroke-width="1.4"/>')

    def tag(x, y, t, fill=FAINT):
        return f'<text class="m" font-size="15" x="{x}" y="{y}" fill="{fill}">{t}</text>'

    def code(x, y, t, fill=INK):
        return (f'<text class="mn" font-size="16" x="{x}" y="{y}" fill="{fill}">{t}</text>')

    # ── in memory
    b.append(tag(L, 46, "in memory"))
    b.append(card(L, 60, LW, 108))
    b.append(code(L + 16, 96, "{&#39;itemA&#39;: [&#39;item&#39;, &#39;A&#39;],"))
    b.append(code(L + 16, 122, "&#160;&#39;itemB&#39;: [1, 3]}"))
    b.append(tag(L + 16, 152, "built-in types"))

    b.append(card(L, 216, LW, 86, seal=True))
    b.append(code(L + 16, 252, "Carseats(pd.DataFrame)"))
    b.append(tag(L + 16, 284, "a class of your own", SEAL))

    # ── both objects feed both writers
    bus = L + LW + 22
    b.append(f'<path d="M{L + LW} 114 H{bus} V259 H{L + LW}" fill="none" '
             f'stroke="{LINE2}" stroke-width="1.4"/>')

    def arrow(y, label):
        return (
            f'<line x1="{bus}" y1="{y}" x2="{R - 22}" y2="{y}" stroke="{SOFT}" stroke-width="1.4"/>'
            f'<path d="M{R - 22} {y - 5} L{R - 8} {y} L{R - 22} {y + 5} Z" fill="{SOFT}"/>'
            f'<text class="mn" font-size="16" text-anchor="middle" x="{(bus + R) / 2:.0f}" '
            f'y="{y - 14}" fill="{INK}">{label}</text>'
        )

    b.append(arrow(114, "pickle.dump()"))
    b.append(arrow(259, "json.dump()"))

    # ── pickle side
    b.append(tag(R, 46, "obj.pickle · binary"))
    b.append(card(R, 60, RW, 108))
    b.append(code(R + 16, 96, "\\x80\\x04\\x95(\\x00\\x00\\x00", SOFT))
    b.append(code(R + 16, 122, "\\x00\\x00}\\x94\\x8c\\x05itemA", SOFT))
    b.append(f'<text class="m" font-size="15" x="{R + 16}" y="152" fill="{FAINT}">'
             f'python only · <tspan fill="{SEAL}">keeps the class</tspan></text>')

    # ── json side
    b.append(tag(R, 202, "obj.json · text"))
    b.append(card(R, 216, RW, 108))
    b.append(code(R + 16, 252, "{&quot;itemA&quot;: [&quot;item&quot;, &quot;A&quot;],"))
    b.append(code(R + 16, 278, "&#160;&quot;itemB&quot;: [1, 3]}"))
    b.append(f'<text class="m" font-size="15" x="{R + 16}" y="308" fill="{FAINT}">'
             f'portable · <tspan fill="{SEAL}">drops the class</tspan></text>')

    # ── the decision the post closes on
    b.append(f'<line class="hair" x1="{L}" y1="{H - 108}" x2="842" y2="{H - 108}"/>')
    b.append(f'<text class="m lbl" x="{L}" y="{H - 74}">must a human read the file?</text>')
    b.append(f'<text class="m lbl" x="{L}" y="{H - 40}">is every object a built-in?</text>')
    b.append(f'<text class="m" font-size="16" x="{R}" y="{H - 74}" fill="{SOFT}">'
             f'yes to both → <tspan fill="{INK}">json</tspan></text>')
    b.append(f'<text class="m" font-size="16" x="{R}" y="{H - 40}" fill="{SOFT}">'
             f'otherwise → <tspan fill="{SEAL}">pickle</tspan></text>')
    write("pickle", "pickle-vs-json.svg", W, H, "\n".join(b))


# ───────────────────────────── 6 · pyradiomics ──────────────────────────────
def pyradiomics():
    """Stacked rather than side-by-side: at 512px of column, four columns of
    mono would set at six pixels. Bands give every stage the full measure."""
    W = 900
    L, Rt = 58, 842
    b = []
    y = 116

    b.append(f'<text class="m lbl" x="{L}" y="40">'
             f'the three sections of features.yaml, over the stage each configures</text>')

    # ── input band: a slice and its mask
    b.append(f'<line x1="{L}" y1="{y - 26}" x2="{Rt}" y2="{y - 26}" stroke="{LINE2}" stroke-width="1.4"/>')
    b.append(f'<text class="m lbl" x="{L}" y="{y - 36}">input</text>')
    for i, (ox, is_mask, name) in enumerate(((0, False, "image.nrrd"), (250, True, "mask.nrrd"))):
        sx, sy = L + ox, y
        b.append(f'<rect x="{sx}" y="{sy}" width="92" height="76" fill="none" '
                 f'stroke="{LINE2}" stroke-width="1.4"/>')
        blob = (f'M{sx + 36} {sy + 25} q 22 -7 29 12 q 6 21 -17 25 '
                f'q -23 4 -24 -16 q -1 -17 12 -21 Z')
        if is_mask:
            b.append(f'<path d="{blob}" fill="{SEAL}" fill-opacity=".16" '
                     f'stroke="{SEAL}" stroke-width="1.6"/>')
        else:
            for j in range(8):
                b.append(f'<line x1="{sx + 7}" y1="{sy + 9 + j * 8.5}" x2="{sx + 85}" '
                         f'y2="{sy + 9 + j * 8.5}" stroke="{FAINT}" '
                         f'stroke-width="{2 if j in (2, 3, 4, 5) else 1}" '
                         f'stroke-opacity="{0.5 if j in (2, 3, 4, 5) else 0.28}"/>')
            b.append(f'<path d="{blob}" fill="{INK}" fill-opacity=".14"/>')
        b.append(f'<text class="mn" font-size="17" x="{sx + 106}" y="{sy + 44}" '
                 f'fill="{INK}">{name}</text>')
    y += 140

    # ── the three configurable stages
    bands = [
        ("setting:", "preprocess",
         ["resample · sitkBSpline", "discretise · binWidth 25", "shift · voxelArrayShift"]),
        ("imageType:", "filter",
         ["Original", "LoG σ = 0.5 1 2", "Wavelet", "LBP3D", "Square", "Logarithm", "Gradient"]),
        ("featureClass:", "describe",
         ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]),
    ]
    for yaml, label, chips in bands:
        # The step arrow rides the right margin: the left of that gutter is
        # occupied by the input plates on the first pass and by chips after.
        ar = Rt - 8
        b.append(f'<path d="M{ar} {y - 76} V{y - 58} M{ar - 7} {y - 66} '
                 f'L{ar} {y - 56} L{ar + 7} {y - 66}" fill="none" stroke="{SOFT}" '
                 f'stroke-width="1.4"/>')
        b.append(f'<line x1="{L}" y1="{y - 26}" x2="{Rt}" y2="{y - 26}" '
                 f'stroke="{LINE2}" stroke-width="1.4"/>')
        b.append(f'<text class="m ttl" x="{L}" y="{y - 36}">{yaml}</text>')
        b.append(f'<text class="m lbl" text-anchor="end" x="{Rt}" y="{y - 36}">{label}</text>')

        # chips wrap across the full measure
        cx, cy = L, y + 22
        for c in chips:
            w = len(c) * 10.4 + 26
            if cx + w > Rt:
                cx, cy = L, cy + 40
            b.append(f'<rect x="{cx}" y="{cy - 22}" width="{w:.0f}" height="32" fill="none" '
                     f'stroke="{LINE}" stroke-width="1.4"/>')
            b.append(f'<text class="mn" font-size="17" x="{cx + 13}" y="{cy}" '
                     f'fill="{INK}">{c}</text>')
            cx += w + 12
        y = cy + 92

    H = y - 4
    b.append(f'<line class="hair" x1="{L}" y1="{y - 62}" x2="{Rt}" y2="{y - 62}"/>')
    b.append(f'<text class="m lbl" x="{L}" y="{y - 32}">'
             f'each image type × each class</text>')
    b.append(f'<text class="m" font-size="16" x="{L + 360}" y="{y - 32}" fill="{SEAL}">→</text>')
    b.append(f'<text class="m" font-size="16" x="{L + 400}" y="{y - 32}" fill="{INK}">'
             f'one row per case · features.csv</text>')
    write("pyradiomics", "radiomics-pipeline.svg", W, H, "\n".join(b))


square_loss()
semidefinite()
smote()
vif()
pickle_fig()
pyradiomics()
