#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.1",
#     "pillow>=12.3.0",
#     "resvg-py>=0.3.3",
#     "scipy>=1.18.0",
#     "typer>=0.26.8",
# ]
# ///
"""TEMPLATE: reconstruct one image.

Copy this next to your work, edit the marked sections, run it with
`uv run --no-project build_<name>.py`. It writes `<project>/project.json`
directly; measurement and model in one pass, since the library does the
measuring.

The shape of the job is always the same:

  1. decide what the SHAPES are, and what covers what        <- your judgment
  2. contour -> segment -> snap -> segments, per shape       <- library
  3. fit the paint of each                                   <- library
  4. hand-measure the constraints the library cannot infer   <- your judgment

Steps 1 and 4 are the reconstruction. Steps 2 and 3 are calls.

Read references/conventions.md first. Scan rays still need background, but
the contour helper now picks a safe start before a nearby component.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# --- locate the skill's bundled png2svg package -----------------------------
def _find_skill() -> Path:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates = []
    if env := os.environ.get("PNG2SVG_SKILL"):
        candidates.append(Path(env).expanduser())
    candidates += [here.parent.parent, here.parent / "png2svg"]
    for base in [cwd, *cwd.parents, Path.home()]:
        candidates += [base / ".claude" / "skills" / "png2svg",
                       base / ".agents" / "skills" / "png2svg",
                       base / "png2svg"]
    for cand in candidates:
        if (cand / "scripts" / "png2svg" / "__init__.py").is_file():
            return cand
    raise SystemExit(
        "could not find the png2svg skill directory.\n"
        "Set PNG2SVG_SKILL=/path/to/skills/png2svg and rerun."
    )


sys.path.insert(0, str(_find_skill() / "scripts"))

from png2svg import primitives as prim                           # noqa: E402
from png2svg.compare import foreground_mask                      # noqa: E402
from png2svg.geom import smooth_polygon                   # noqa: E402,F401
from png2svg.measure import (                                    # noqa: E402
    Field, subpixel_contour,
    edge_samples, fit_line, intersect,        # noqa: F401  hand measurement
    fit_circle, fit_corner_full,              # noqa: F401
)
from png2svg.model import load_project, save_project             # noqa: E402
from png2svg.outline import (                                    # noqa: E402
    segment_outline, snap_outline, summarise, to_segments,
)
from png2svg.paint import (                                      # noqa: E402
    fit_linear_gradient, flat_colour, fit_shared_ramp,  # noqa: F401
    map_ramp,                                           # noqa: F401
)

# ==== EDIT: project =========================================================
PROJECT = Path("work/name")
meta = json.loads((PROJECT / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
img = Image.open(PROJECT / "source" / meta["source"]["path"].split("/")[-1])
rgb = np.asarray(img.convert("RGB"))
F = Field(rgb, BG)

# Tolerance trades segment count against deviation. Sweep it and take the
# knee: 0.25-0.6 for clean sources, near the texture scale for grainy ones.
# If the fit wants far more segments than the artwork plausibly has, it is
# chasing noise.
TOL = 0.4

mask = ndimage.binary_closing(foreground_mask(img, BG), np.ones((3, 3)))


def path_of(region, label, tol=TOL, reverse=False):
    """Contour -> primitives -> verified constraints -> model segments."""
    C = subpixel_contour(F, ndimage.binary_fill_holes(region))
    if reverse:                    # a counter must wind the opposite way
        C = C[::-1]
    prims = segment_outline(C, tol=tol)
    snapped, notes = snap_outline(prims, contour=C)
    rejected = [n for n in notes if "REJECT" in n]
    print(f"  {label:10s}: {len(C):5d} pts -> {len(prims):3d} prims "
          f"({summarise(prims)}), {len(rejected)}/{len(notes)} rejected")
    for n in rejected[:3]:         # rejections describe how the shape is built
        print(f"      {n}")
    return to_segments(snapped)


def fit_primitives(contour, build, p0, trim=0.0):
    """THE OTHER ROUTE (SKILL.md 3b). Use this instead of `path_of` when the
    artwork is overlapping filled primitives and the union's boundary is a
    by-product -- bars, rects, discs, capsules, anything repeated.

    `build(p)` returns [(vertices, radius), ...] as a function of the parameter
    vector; emit all of them with `prim.paths(build(fit.params))`. A radius can
    be one value or one value per corner. Symmetries
    delete parameters rather than merely constraining them: three rounded
    parallelograms with 180-degree symmetry came to eight numbers at 0.080px.

    Tells that you are here rather than in `path_of`: a corner that is an
    intersection rather than a fillet, a spacing that repeats, an edge that is
    an outer boundary in one place and an interior seam in another.
    """
    fit = prim.fit_union(contour, build, p0, trim=trim)
    print(f"  primitives: {fit.summary()}")
    print(f"      worst: {fit.worst_points(3)}")
    return fit


def paint_of(region, label, trim=0.0):
    """Gradient if the region varies along an axis, flat otherwise.

    Pass trim when something is painted ON TOP of the fill; a shadow, a
    glow, a watermark; or it will drag the whole fit toward itself.
    """
    core = ndimage.distance_transform_edt(region) >= 4
    spread = float(np.asarray(rgb, float)[core].std(axis=0).mean()) if core.sum() > 200 else 0.0
    if spread < 4.0:
        col = flat_colour(rgb, region)
        print(f"  {label:10s}: flat {col} (spread {spread:.1f})")
        return {"type": "solid", "color": col}
    g = fit_linear_gradient(rgb, region, trim=trim)
    print(f"  {label:10s}: gradient rms {g['rms']:.2f} axis {g['axis_deg']:.1f} "
          f"{' -> '.join(s['color'] for s in g['stops'])}")
    return {k: v for k, v in g.items() if k not in ("axis_deg", "rms")}


# ==== EDIT: decompose =======================================================
# This is the reconstruction. Everything else is arithmetic.
#
#   - How many shapes, and what covers what? Shapes paint in list order.
#   - WHICH ROUTE? `path_of` traces a boundary that IS the design (letterform,
#     swoosh, blob). `fit_primitives` solves a construction of overlapping
#     primitives whose union boundary is incidental. Picking wrong costs either
#     a pile of nodes or an afternoon -- see SKILL.md section 2.
#   - A hard colour boundary INSIDE one silhouette usually means two
#     overlapping shapes, and the seam is the top shape's edge.
#   - A duplicated shape carries its own gradient in its own local span; the
#     tell is a colour JUMP where two copies meet. Use `fit_shared_ramp`.
#   - When paint follows a bent shape, split it into straight and curved
#     spans, then use `map_ramp(paint, start, end, global_ramp)` on each.
#   - Where one shape hides under another, GROW it into the occluder before
#     tracing (intersect with the occluder itself, never a dilated copy) so
#     no background hairline shows at the join.
#   - A counter is a second subpath in the SAME shape, wound the other way.
#   - Split by colour where connectivity will not do it, but bound the
#     region spatially too: a bright patch elsewhere can match the same test.

regions = {"main": mask}

shapes = []
for name, region in regions.items():
    shapes.append({
        "id": name, "type": "path",
        "d": path_of(region, name),
        "fills": [paint_of(region, name)],
    })

# ==== EDIT: constrain =======================================================
# Here is where a reconstruction goes from good to exact. The library cannot
# know that a cap's radius IS the half-width between two parallel sides, or
# that a bowl is tangent to the flat above it, or that two edges share a
# direction because the artwork sits on a grid. Measure those by hand with
# edge_samples/fit_line/intersect/fit_circle and rebuild the affected
# segments from them.
#
# Every constraint you confirm removes a free parameter that was absorbing
# noise. Every one you invent moves geometry that was already right; so
# check that the constrained fit beats the free one before keeping it.

proj = load_project(PROJECT)
proj.shapes = shapes
proj.notes = [
    f"outlines: segment_outline(tol={TOL}) + verified snap_outline",
    # record what you DECIDED and what you deliberately dropped
]
save_project(PROJECT, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes,",
      len(proj.shapes), "shapes")
print("\nnow: check work/<name> --label r1, read the residual clusters, iterate")
