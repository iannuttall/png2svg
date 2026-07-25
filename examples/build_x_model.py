"""Build work/x (textured 3D app icon) — a "good enough" reconstruction.

This source is a rendered icon: brushed metal, grain, drop shadows, a glow,
engraved text and a scanline waveform. Its interior disagrees with itself by
around 15 levels, so no vector can match it per pixel and chasing deltaE
would be chasing noise.

What IS recoverable is the structure and the large-scale shading, so that is
what this builds: background wash, dark shell, metal plate, screen, slot.
Grain and fine detail are deliberately dropped. Judge the result by
silhouette IoU and the low-frequency deltaE that `check` reports, not by the
per-pixel figures.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from png2svg.measure import Field, subpixel_contour
from png2svg.model import load_project, save_project
from png2svg.outline import segment_outline, snap_outline, summarise, to_segments
from png2svg.paint import fit_linear_gradient

proj_dir = Path("work/x")
meta = json.loads((proj_dir / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
W, H = meta["source"]["width"], meta["source"]["height"]
img = Image.open(proj_dir / "source" / meta["source"]["path"].split("/")[-1])
rgb = np.asarray(img.convert("RGB"))
a = rgb.astype(float)
F = Field(rgb, BG)
TOL = 1.2          # texture-scale: fitting tighter than the grain is pointless

lum = a.mean(2)
sat = a.max(2) - a.min(2)
metal = ndimage.binary_opening((lum > 110) & (sat < 60), np.ones((5, 5)))
screen = ndimage.binary_opening((a[..., 1] > 140) & (a[..., 2] > 140)
                                & (a[..., 0] < 160) & (sat > 40), np.ones((5, 5)))
dark = lum < 45


def largest(m, fill=True):
    m = ndimage.binary_closing(m, np.ones((7, 7)))
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    out = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(out) if fill else out


screen = largest(screen)
# The metal plate lives above the screen. Without that cut, the waveform
# inside the screen — bright and unsaturated, so metal-coloured by the same
# test — joins the plate's component and drags its outline down through the
# middle of the icon.
screen_top = int(np.nonzero(screen.any(axis=1))[0].min())
metal_only = metal.copy()
metal_only[screen_top:] = False
metal = largest(metal_only)
waveform = ndimage.binary_opening(
    (lum > 150) & screen, np.ones((3, 3)))
# the icon silhouette: metal and screen plus the dark shell hugging them,
# which separates the icon's own shadow from the vignette at the canvas edge
seed = metal | screen
icon = largest(seed | (dark & ndimage.binary_dilation(seed, np.ones((81, 81)))))
# the slot is a dark island sitting inside the metal plate
slot = largest(dark & ndimage.binary_erosion(metal, np.ones((21, 21))))


def path_of(region, label, tol=TOL):
    C = subpixel_contour(F, region, offset=9.0)
    prims = segment_outline(C, tol=tol)
    snapped, notes = snap_outline(prims, contour=C)
    print(f"  {label:10s}: {len(C):5d} pts -> {len(prims):3d} prims "
          f"({summarise(prims)})")
    return to_segments(snapped)


def paint_of(region, label, erode=6, trim=0.25):
    g = fit_linear_gradient(rgb, region, erode=erode, trim=trim, max_stops=4)
    print(f"  {label:10s}: gradient rms {g['rms']:.2f} axis {g['axis_deg']:6.1f} "
          f"{' -> '.join(s['color'] for s in g['stops'])}")
    return {k: v for k, v in g.items() if k not in ("axis_deg", "rms")}


shapes = []
# 1. background wash across the whole canvas
bg_region = ~ndimage.binary_dilation(icon, np.ones((41, 41)))
shapes.append({
    "id": "backdrop", "type": "path",
    "d": [["M", 0, 0], ["L", W, 0], ["L", W, H], ["L", 0, H], ["Z"]],
    "fills": [paint_of(bg_region, "backdrop", erode=2)],
})
# 2. dark shell, 3. metal plate, 4. screen
# The shell is a ring, so a linear ramp fitted across it is meaningless —
# it runs from one side, through the hole where the plate sits, to the other.
# A ring wants its own median colour.
from png2svg.paint import flat_colour  # noqa: E402
# sample the dark ring itself, not the glow that bleeds across it
shell_col = flat_colour(rgb, icon & ~seed & dark, erode=4)
print(f"  {'shell':10s}: flat {shell_col} (a ring cannot carry a linear ramp)")
shapes.append({"id": "shell", "type": "path", "d": path_of(icon, "shell"),
               "fills": [{"type": "solid", "color": shell_col}]})
shapes.append({"id": "metal", "type": "path", "d": path_of(metal, "metal"),
               "fills": [paint_of(metal, "metal")]})
# The waveform reaches the screen's top edge, so it is a notch rather than a
# hole and no amount of hole-filling closes it — the traced outline would
# detour around every scanline. Close the mask first so the screen keeps the
# rounded rectangle it actually is.
screen_shape = ndimage.binary_fill_holes(
    ndimage.binary_closing(screen, np.ones((45, 45))))
shapes.append({"id": "screen", "type": "path", "d": path_of(screen_shape, "screen"),
               "fills": [paint_of(screen, "screen")]})
# The waveform is dozens of disconnected scanlines, not one region: tracing
# its largest blob reconstructs a fragment and misreads the rest. It is fine
# detail on a textured source, so it is dropped by design.
if False:
    shapes.append({"id": "waveform", "type": "path",
                   "d": path_of(ndimage.binary_closing(waveform, np.ones((5, 5))),
                                "waveform", 1.5),
                   "fills": [{"type": "solid",
                              "color": flat_colour(rgb, waveform, erode=1)}]})
if slot.sum() > 400:
    shapes.append({"id": "slot", "type": "path", "d": path_of(slot, "slot", 0.6),
                   "fills": [paint_of(slot, "slot", erode=2, trim=0.3)]})

proj = load_project(proj_dir)
proj.shapes = shapes
proj.notes = [
    "TEXTURED SOURCE: interior texture_std ~15, so per-pixel deltaE is not a",
    "meaningful target. Structure and large-scale shading are reconstructed;",
    "grain, brushed-metal streaks, engraved text and the waveform are dropped.",
    "Judge by silhouette_iou and deltaE_lowfreq, not deltaE_mean/p95.",
    f"outlines at tol={TOL}px — fitting tighter than the grain fits noise",
]
save_project(proj_dir, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes,",
      len(proj.shapes), "shapes")
