"""Build work/r (ribbon-and-cylinder mark).

The ribbon passes behind the cylinder, so it arrives as three separate
visible pieces. Each is reconstructed from its own contour and then grown
underneath the cylinder before tracing, so no hairline of background can
show along the join; the cylinder is painted last, over the top.

The cylinder's own component holds two shapes — a dark body and a lighter
elliptical top — split here by colour rather than by connectivity.
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

proj_dir = Path("work/r")
meta = json.loads((proj_dir / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
img = Image.open(proj_dir / "source" / meta["source"]["path"].split("/")[-1])
rgb = np.asarray(img.convert("RGB"))
F = Field(rgb, BG)
TOL = 0.5

mask = ndimage.binary_closing(foreground_mask(img, BG), np.ones((3, 3)))
lab, n = ndimage.label(mask)
sizes = ndimage.sum(mask, lab, range(1, n + 1))
comps = [lab == i + 1 for i, s in enumerate(sizes) if s > 300]

a = rgb.astype(float)
# the cylinder is the component whose interior is dark
def darkness(c):
    core = ndimage.distance_transform_edt(c) >= 4
    return float(np.median(a[core].mean(axis=1))) if core.sum() > 50 else 255.0


cyl = min(comps, key=darkness)
ribbons = [c for c in comps if c is not cyl]
blue = cyl & (a[..., 2] > a[..., 0] + 40)
blue = ndimage.binary_opening(blue, np.ones((5, 5)))
body = cyl & ~blue


def path_of(region, label):
    region = ndimage.binary_fill_holes(region)
    C = subpixel_contour(F, region, offset=7.0)
    prims = segment_outline(C, tol=TOL)
    snapped, notes = snap_outline(prims, contour=C)
    rejected = sum(1 for x in notes if "REJECT" in x)
    print(f"  {label:9s}: {len(C):5d} pts -> {len(prims):3d} prims "
          f"({summarise(prims)}), {rejected}/{len(notes)} constraints rejected")
    return to_segments(snapped)


def paint_of(region, label):
    """Gradient if the region varies along an axis, flat otherwise."""
    core = ndimage.distance_transform_edt(region) >= 4
    if core.sum() < 200:
        return {"type": "solid", "color": flat_colour(rgb, region, erode=2)}
    spread = float(a[core].std(axis=0).mean())
    if spread < 4.0:
        col = flat_colour(rgb, region, erode=4)
        print(f"  {label:9s}: flat {col} (spread {spread:.1f})")
        return {"type": "solid", "color": col}
    g = fit_linear_gradient(rgb, region, trim=0.12)
    print(f"  {label:9s}: gradient rms {g['rms']:.2f} axis {g['axis_deg']:.1f} "
          f"{' -> '.join(s['color'] for s in g['stops'])}")
    return {k: v for k, v in g.items() if k not in ("axis_deg", "rms")}


shapes = []
# Ribbons first, grown under the cylinder so the join cannot show a seam.
# The growth is intersected with the cylinder itself, never with a dilated
# copy of it: a ribbon allowed past the cylinder's own edge would hang out
# over the background, which costs far more than the seam it was avoiding.
for i, r in enumerate(ribbons):
    tucked = r | (ndimage.binary_dilation(r, np.ones((9, 9))) & cyl)
    shapes.append({"id": f"ribbon{i}", "type": "path",
                   "d": path_of(tucked, f"ribbon{i}"),
                   "fills": [paint_of(r, f"ribbon{i}")]})
# then the cylinder over the top
shapes.append({"id": "body", "type": "path", "d": path_of(cyl, "cylinder"),
               "fills": [paint_of(body, "body")]})

# The cylinder's top face is an ellipse. Tracing it instead gives a ragged,
# dripping edge: the blue/dark boundary is a soft colour transition inside
# the shape, so subpixel_contour has no background to scan against and falls
# back to the raw ±0.5px traced pixels. Fitting the ellipse the designer drew
# sidesteps that entirely.
edge = blue & ~ndimage.binary_erosion(blue, np.ones((3, 3)))
P = np.stack(np.nonzero(edge)[::-1], 1).astype(float)
q = np.array([P[:, 0].mean(), P[:, 1].mean(),
              np.ptp(P[:, 0]) / 2, np.ptp(P[:, 1]) / 2])
for _ in range(3):
    def ell_res(p, pts=P):
        return (np.hypot((pts[:, 0] - p[0]) / p[2],
                         (pts[:, 1] - p[1]) / p[3]) - 1.0) * min(p[2], p[3])

    q = optimize.least_squares(ell_res, q).x
    err = np.abs(ell_res(q))
    keep = err <= np.percentile(err, 85)
    if keep.sum() > 40:
        P = P[keep]
cx, cy, rx, ry = (float(v) for v in q)
print(f"  {'blue top':9s}: ellipse ({cx:.1f}, {cy:.1f}) rx {rx:.2f} ry {ry:.2f}, "
      f"max residual {np.abs(ell_res(q)).max():.2f}")
shapes.append({
    "id": "top", "type": "path",
    "d": [["M", cx - rx, cy],
          ["A", rx, ry, 0, 0, 1, cx + rx, cy],
          ["A", rx, ry, 0, 0, 1, cx - rx, cy], ["Z"]],
    "fills": [paint_of(blue, "blue top")],
})

proj = load_project(proj_dir)
proj.shapes = shapes
proj.notes = [
    "ribbon passes behind the cylinder and so appears as three pieces;",
    "each is grown under the cylinder before tracing and the cylinder is",
    "painted last, so no background hairline can show along the joins",
    "cylinder component split by colour into dark body + lighter elliptical top",
    f"outlines: segment_outline(tol={TOL}) + verified snap_outline",
]
save_project(proj_dir, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes,",
      len(proj.shapes), "shapes")
