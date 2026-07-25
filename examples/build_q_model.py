"""Build work/q (gradient loop mark) using the library primitives only.

Compare this with examples/measure_p.py: the per-image work here is deciding
what the shapes ARE — a gradient-filled loop with a counter, a green cap
drawn over its end, and a shadow where the stroke crosses itself — while
every measurement is a library call.

The loop's left end is left running underneath the green cap rather than
being measured: it is not visible, so there is nothing to measure, and
tucking it under is what keeps the seam from showing.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

from png2svg.compare import foreground_mask
from png2svg.measure import Field, subpixel_contour
from png2svg.model import load_project, save_project
from png2svg.outline import segment_outline, snap_outline, summarise, to_segments
from png2svg.paint import fit_linear_gradient, flat_colour

proj_dir = Path("work/q")
meta = json.loads((proj_dir / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
img = Image.open(proj_dir / "source" / meta["source"]["path"].split("/")[-1])
rgb = np.asarray(img.convert("RGB"))
F = Field(rgb, BG)

TOL = 0.45

mask = ndimage.binary_closing(foreground_mask(img, BG), np.ones((3, 3)))
filled = ndimage.binary_fill_holes(mask)
a = rgb.astype(float)
green = mask & (a[..., 1] > a[..., 0] + 20) & (a[..., 1] > a[..., 2] + 20)


def path_of(region, reverse=False, label=""):
    """Contour -> verified, snapped primitives -> model segments."""
    C = subpixel_contour(F, region, offset=9.0)
    if reverse:
        C = C[::-1]
    prims = segment_outline(C, tol=TOL)
    snapped, notes = snap_outline(prims, contour=C)
    rejected = sum(1 for n in notes if "REJECT" in n)
    print(f"  {label}: {len(C)} pts -> {len(prims)} prims ({summarise(prims)}), "
          f"{len(notes)} constraints, {rejected} rejected")
    return to_segments(snapped)


# Trace the loop ALONE, not the silhouette. Tracing the union would put the
# cap's outline into the loop's path, and the two shapes meet at sharp
# concave junctions that any segmenter will cut the corner on. Instead the
# loop keeps its own clean edges and is extended straight under the cap —
# the hidden end is covered, so it never needs measuring.
BAR_TOP, BAR_BOTTOM = 361, 424       # rows solidly inside the bar
loop_region = (mask & ~green).copy()
loop_region[BAR_TOP:BAR_BOTTOM, int(round(79.0)):140] = True
loop_region = ndimage.binary_fill_holes(ndimage.binary_closing(
    loop_region, np.ones((3, 3))))
outer = path_of(loop_region, label="outer")
hole_mask = ndimage.binary_fill_holes(filled & ~mask)
hole = path_of(hole_mask, reverse=True, label="counter")

# Paint: fitted on loop pixels only, trimming the self-crossing shadow which
# would otherwise drag the ramp and make it look like something exotic.
loop_px = ndimage.binary_erosion(mask & ~green, np.ones((3, 3)), iterations=2)
grad = fit_linear_gradient(rgb, loop_px, n_stops=2, trim=0.12)
print(f"  gradient: axis {grad['axis_deg']:.2f} deg, rms {grad['rms']:.2f}, "
      f"{' -> '.join(s['color'] for s in grad['stops'])}")

# Green cap: a circle. Only the arc that borders the page is real boundary —
# where the cap meets the loop the traced contour follows the junction, not
# the circle, so those points are trimmed away rather than fitted.
gc = subpixel_contour(F, ndimage.binary_fill_holes(green), offset=7.0)
# keep only points that are genuinely cap-against-page: the junction with the
# loop sits where the loop's own pixels are adjacent
loop_any = ndimage.binary_dilation(mask & ~green, np.ones((9, 9)))
free = np.array([p for p in gc if not loop_any[int(round(p[1])), int(round(p[0]))]])
if len(free) < 30:
    free = gc
q = np.array([79.0, 398.0, 40.0])
for _ in range(3):
    def circ_res(p, pts=free):
        return np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2]

    q = optimize.least_squares(circ_res, q).x
    err = np.abs(circ_res(q))
    keep = err <= np.percentile(err, 90)
    if keep.sum() >= 30:
        free = free[keep]
cx, cy, r = (float(v) for v in q)
print(f"  green cap: centre ({cx:.2f}, {cy:.2f}) r {r:.2f}, "
      f"max residual {np.abs(circ_res(q)).max():.3f} over {len(free)} pts")

shapes = [
    {"id": "loop", "type": "path", "d": outer + hole,
     "fills": [{k: v for k, v in grad.items() if k not in ("axis_deg", "rms")}]},
    {"id": "cap", "type": "path",
     "d": [["M", cx - r, cy],
           ["A", r, r, 0, 1, 1, cx + r, cy],
           ["A", r, r, 0, 1, 1, cx - r, cy], ["Z"]],
     "fills": [{"type": "solid", "color": flat_colour(rgb, green, erode=6)}]},
]

proj = load_project(proj_dir)
proj.shapes = shapes
proj.notes = [
    "gradient loop with a counter, plus a green cap painted over its left end",
    "the loop runs under the cap: the hidden end is not measured, it is covered",
    f"paint fitted with trim=0.12 to reject the self-crossing shadow "
    f"(rms {grad['rms']:.2f}; without trimming it reads 6.5 and looks unfittable)",
    f"outline: segment_outline(tol={TOL}) + verified snap_outline",
]
save_project(proj_dir, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes")
