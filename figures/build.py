#!/usr/bin/env python3
"""Build a TikZ figure into a theme-following SVG for a post.

    python3 figures/build.py figures/src/<post>/<name>.tex [--preview DIR]

    figures/src/lstm/rnn-loop.tex  ->  public/img/fig/lstm/rnn-loop.svg

The SVG has no background and no colours of its own: every colour in it is a
CSS variable (--ink, --soft, --fa ...), defined per theme in global.css, so the
figure is redrawn by the page in DAY and in NIGHT. That only works when the
figure is inlined into the page rather than loaded through <img>, which is
what src/plugins/rehype-inline-figures.mjs does.

How the colours get there: preamble.tex defines each colour as an exact
SENTINEL RGB value; after tectonic -> PDF -> pdftocairo -> SVG, every
`rgb(...)` in the file is looked up in SENTINELS below and replaced by its
variable. A colour that is not a sentinel (a mix like fa!20, or plain black)
fails the build rather than shipping a figure that is invisible in one theme.

--preview DIR also writes DIR/<post>--<name>.day.png and .night.png, the figure
rendered on each theme's background, for review.
"""
import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "figures"
FONTS = KIT / "fonts"
OUT = ROOT / "public" / "img" / "fig"

# Fetched on first run (gitignored). All OFL; the same faces the site loads.
FONT_URLS = {
    f"Spectral-{w}.ttf": f"https://github.com/google/fonts/raw/main/ofl/spectral/Spectral-{w}.ttf"
    for w in ("Regular", "Italic", "Medium", "Light", "LightItalic")
} | {
    f"IBMPlexMono-{w}.ttf": f"https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-{w}.ttf"
    for w in ("Regular", "Medium", "Light")
} | {
    "STIXTwoMath-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/stixtwomath/STIXTwoMath-Regular.ttf",
}

# sentinel (R,G,B) in preamble.tex -> CSS variable
SENTINELS = {
    (1, 0, 10): "var(--ink)",
    (2, 0, 10): "var(--soft)",
    (3, 0, 10): "var(--faint)",
    (4, 0, 10): "var(--line2)",
    (5, 0, 10): "var(--fa)",
    (6, 0, 10): "var(--fb)",
    (7, 0, 10): "var(--fc)",
    (8, 0, 10): "var(--fd)",
    (9, 0, 10): "var(--fe)",
    (10, 0, 10): "var(--bg)",
}

# The theme values, for previews only - keep in step with global.css.
THEMES = {
    "day": {"--bg": "#dfe8f1", "--ink": "#101c26", "--soft": "#475666", "--faint": "#788999",
            "--line2": "rgba(16,28,38,.24)", "--fa": "#1f6b9e", "--fb": "#9a6418",
            "--fc": "#3d7a50", "--fd": "#a6454d", "--fe": "#6a58a6"},
    "night": {"--bg": "#08131d", "--ink": "#dce8f2", "--soft": "#8ba2b6", "--faint": "#54687d",
              "--line2": "rgba(220,232,242,.22)", "--fa": "#5fa9e0", "--fb": "#d9a45f",
              "--fc": "#7fbf91", "--fd": "#e08a90", "--fe": "#ae9de0"},
}

# pt -> px. 1.333 is CSS's own; the extra 1.5 brings TikZ's \small (9pt) to
# about the post's body size (17px) when the figure is shown at its natural width.
PX_PER_PT = 1.333 * 1.5

RGB = re.compile(r"rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)")


def fetch_fonts():
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        if not (FONTS / name).exists():
            print(f"fetching {name}")
            urllib.request.urlretrieve(url, FONTS / name)


def to_var(m, where):
    key = tuple(round(float(v) * 2.55) for v in m.groups())
    if key not in SENTINELS:
        sys.exit(f"{where}: colour rgb{key} is not a figkit sentinel - use ink/soft/faint/rule/fa..fe/paper, "
                 "with fill opacity for tints, never a mix like fa!20")
    return SENTINELS[key]


def themed(svg, slug, where):
    svg = RGB.sub(lambda m: to_var(m, where), svg)
    # pdftocairo names things glyph-0-1, clip-3 ...: unique per file, not per page.
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    for i in sorted(ids, key=len, reverse=True):
        svg = re.sub(rf'(id="|#){re.escape(i)}(?=["\)])', rf"\g<1>{slug}-{i}", svg)
    # natural size -> --w, so the figure never renders larger than drawn but
    # still shrinks with the column.
    w = re.search(r'<svg[^>]*\bwidth="([\d.]+)(pt)?"', svg)
    width_px = round(float(w.group(1)) * PX_PER_PT) if w else 640
    svg = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", svg, count=1)
    svg = svg.replace("<svg ", f'<svg class="tfig" style="--w:{width_px}px" ', 1)
    # Opened on its own (an <img>, e.g. the writing page's preview plate) the
    # page's variables are out of reach, so the file carries the two palettes
    # itself. `:root.tfig` matches only when the SVG is the document root; once
    # inlined into a post, :root is <html> and this rule never applies.
    def decl(theme):
        return ";".join(f"{k}:{v}" for k, v in THEMES[theme].items())
    own = (f"<style>:root.tfig{{{decl('day')}}}"
           f"@media (prefers-color-scheme:dark){{:root.tfig{{{decl('night')}}}}}</style>")
    svg = re.sub(r"(<svg[^>]*>)", lambda m: m.group(1) + own, svg, count=1)
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    return svg.strip() + "\n"


def preview(svg, png, theme):
    css = ";".join(f"{k}:{v}" for k, v in THEMES[theme].items())
    html = png.with_suffix(".html")
    html.write_text(
        f'<html><body style="margin:0;{css};background:var(--bg)">'
        f'<div style="padding:28px;display:inline-block">{svg}</div>'
        "<style>.tfig{display:block;width:var(--w)}</style></body></html>"
    )
    w = int(re.search(r"--w:(\d+)px", svg).group(1)) + 56
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={w},2000", f"--screenshot={png}", f"file://{html}"],
                   check=True, capture_output=True)
    html.unlink()
    # trim the empty bottom of the 2000px window
    subprocess.run(["python3", "-c", TRIM, str(png)], check=True)


TRIM = """
import sys
from PIL import Image, ImageChops
p=sys.argv[1]; im=Image.open(p).convert('RGB')
bg=Image.new('RGB',im.size,im.getpixel((2,im.height-2)))
box=ImageChops.difference(im,bg).getbbox()
if box: im.crop((0,0,im.width,min(im.height,box[3]+56))).save(p)
"""


def build(tex, preview_dir=None):
    tex = tex.resolve()
    post, name = tex.parent.name, tex.stem
    fetch_fonts()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "preamble.tex").write_text(
            (KIT / "preamble.tex").read_text().replace("FIGKITFONTS", str(FONTS)))
        (tmp / "fig.tex").write_text(tex.read_text().replace("FIGKIT/preamble.tex", str(tmp / "preamble.tex")))
        r = subprocess.run(["tectonic", "--chatter", "minimal", "fig.tex"], cwd=tmp, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{tex}: tectonic failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        subprocess.run(["pdftocairo", "-svg", "fig.pdf", "fig.svg"], cwd=tmp, check=True)
        svg = themed((tmp / "fig.svg").read_text(), f"{post}-{name}", tex)
    out = OUT / post / f"{name}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    width = re.search(r"--w:(\d+)px", svg).group(1)
    print(f"{out.relative_to(ROOT)}  ({len(svg) // 1024} KB, {width}px)")
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for theme in THEMES:
            preview(svg, (preview_dir / f"{post}--{name}.{theme}.png").resolve(), theme)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="+", type=Path)
    ap.add_argument("--preview", metavar="DIR")
    a = ap.parse_args()
    for t in a.tex:
        build(t, a.preview)
