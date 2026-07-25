"""Export the P as layered artwork for Apple's Icon Composer.

Writes icon/: 01-background.svg, 02-p.svg, flat.svg, flat-1024.png. Every
SVG is 1024x1024 with an identical viewBox so the layers stack in register.

The P's path is taken from work/p UNCHANGED — only scaled and translated
onto the canvas. The plug is negative space in this mark: it is already a
real cutout in the path, which is exactly what the brief asks for and what
survives recolouring, so there is nothing to rebuild and nothing to cover.

There is deliberately NO 03-plug.svg. The plug is not a separate object; it
is the hole. Shipping it as a layer would mean inventing geometry that is
not in the logo.

No shadows, glows, bevels or depth gradients: macOS generates those itself.
"""

import io
import json
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

from png2svg.model import Project
from png2svg.svggen import generate_svg

SRC = json.loads(Path("work/p/project.json").read_text())
D = SRC["model"]["shapes"][0]["d"]

OUT = Path("icon")
CANVAS = 1024.0
MARGIN = 100.0
INK = "#ffffff"
BACKDROP = "#19191a"

# bounds of the mark, from the measured extremes rather than the path
M = json.loads(Path("work/p/analysis/measurements.json").read_text())
x0, x1 = M["x"]["stem_left"], M["bowl"]["xR"]
y0, y1 = M["y"]["top"], M["y"]["stem_bottom"]
avail = CANVAS - 2 * MARGIN
scale = min(avail / (x1 - x0), avail / (y1 - y0))
tx = (CANVAS - (x1 - x0) * scale) / 2.0 - x0 * scale
ty = (CANVAS - (y1 - y0) * scale) / 2.0 - y0 * scale


def place(segs):
    out = []
    for seg in segs:
        if seg[0] == "A":
            out.append(["A", seg[1] * scale, seg[2] * scale, seg[3], seg[4],
                        seg[5], seg[6] * scale + tx, seg[7] * scale + ty])
        elif seg[0] == "Z":
            out.append(["Z"])
        else:
            out.append([seg[0]] + [(v * scale + tx) if i % 2 == 0
                                   else (v * scale + ty)
                                   for i, v in enumerate(seg[1:])])
    return out


def canvas(shapes):
    p = Project(source_path="work/p", width=int(CANVAS), height=int(CANVAS),
                sha256="", background=[0, 0, 0, 0],
                view_box=[0, 0, CANVAS, CANVAS], shapes=shapes)
    p.validate()
    return generate_svg(p)


P_LAYER = {"id": "p", "type": "path", "d": place(D),
           "fills": [{"type": "solid", "color": INK}]}

OUT.mkdir(exist_ok=True)
files = {
    # full bleed, square, no rounding: macOS applies its own corner mask
    "01-background.svg": canvas([{
        "id": "bg", "type": "path",
        "d": [["M", 0, 0], ["L", CANVAS, 0], ["L", CANVAS, CANVAS],
              ["L", 0, CANVAS], ["Z"]],
        "fills": [{"type": "solid", "color": BACKDROP}]}]),
    "02-p.svg": canvas([P_LAYER]),
    "flat.svg": canvas([P_LAYER]),
}
for name, svg in files.items():
    (OUT / name).write_text(svg)
    print(f"  {name:20s} {len(svg):5d} bytes")

png = resvg_py.svg_to_bytes(svg_string=files["flat.svg"], width=1024, height=1024)
img = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
img.save(OUT / "flat-1024.png")
print(f"  {'flat-1024.png':20s} {img.size}")
print(f"\nmark {(x1 - x0) * scale:.0f}x{(y1 - y0) * scale:.0f} of 1024, "
      f"margins >= {MARGIN:.0f}px; path nodes: {len(D)} (unchanged from work/p)")
