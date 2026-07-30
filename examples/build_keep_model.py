#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.5.1",
#   "pillow>=12.3.0",
#   "resvg-py>=0.3.3",
#   "scipy>=1.18.0",
#   "typer>=0.26.8",
# ]
# ///
"""Build work/keep from repeated fitted primitives."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "png2svg" / "scripts"))

from png2svg import primitives as prim  # noqa: E402
from png2svg.compare import foreground_mask  # noqa: E402
from png2svg.measure import Field, subpixel_contour  # noqa: E402
from png2svg.model import load_project, save_project  # noqa: E402
from png2svg.paint import flat_colour  # noqa: E402


PROJECT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT / "work" / "keep"
)
meta = json.loads((PROJECT / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
img = Image.open(PROJECT / "source" / Path(meta["source"]["path"]).name)
rgb = np.asarray(img.convert("RGB"))
field = Field(rgb, BG)

mask = ndimage.binary_fill_holes(
    ndimage.binary_closing(foreground_mask(img, BG), np.ones((3, 3)))
)
labels, count = ndimage.label(mask)
if count != 3:
    raise SystemExit(f"expected three components, found {count}")
regions = [labels == i for i in range(1, count + 1)]
contours = [subpixel_contour(field, region) for region in regions]


def rounded_model_path(shapes):
    """Emit stable, readable coordinates while preserving fitted geometry."""
    out = []
    for seg in prim.paths(shapes):
        out.append(
            [seg[0], *(round(float(value), 4) for value in seg[1:])]
            if len(seg) > 1
            else ["Z"]
        )
    return out


# Central vertical plus the two right diagonals. The diagonal inner ends are
# hidden, so their length is deliberately fixed long enough to run under the
# vertical bar; only their visible outer end, direction, width and radius are
# fitted.
RIGHT_ARM_LENGTH = 290.0


def build_right(p):
    (
        vcx,
        vcy,
        vw,
        vh,
        vr,
        upper_x,
        upper_y,
        upper_angle,
        upper_w,
        upper_r,
        lower_x,
        lower_y,
        lower_angle,
        lower_w,
        lower_r,
    ) = p
    shapes = [(prim.rectangle(vcx - vw / 2, vcy - vh / 2, vw, vh), vr)]
    for outer, arm_angle, width, radius in (
        (np.array([upper_x, upper_y]), upper_angle, upper_w, upper_r),
        (np.array([lower_x, lower_y]), lower_angle, lower_w, lower_r),
    ):
        direction = np.array(
            [np.cos(np.radians(arm_angle)), np.sin(np.radians(arm_angle))]
        )
        centre = outer - direction * (RIGHT_ARM_LENGTH / 2)
        shapes.append(
            (
                prim.oriented_rectangle(
                    centre, RIGHT_ARM_LENGTH, width, arm_angle
                ),
                radius,
            )
        )
    return shapes


right0 = np.array(
    [
        626.0,
        592.0,
        87.0,
        539.0,
        7.0,
        826.0,
        397.0,
        -45.7,
        80.0,
        9.0,
        828.0,
        797.0,
        45.7,
        80.0,
        10.0,
    ]
)
right_bounds = (
    [
        620,
        585,
        80,
        525,
        1,
        818,
        388,
        -49,
        72,
        2,
        820,
        788,
        42,
        72,
        2,
    ],
    [
        632,
        602,
        94,
        552,
        14,
        835,
        407,
        -42,
        88,
        18,
        837,
        807,
        49,
        88,
        18,
    ],
)
right_fit = prim.fit_union(
    contours[0], build_right, right0, bounds=right_bounds
)
print("right:", right_fit.summary())
print("  params:", [round(float(x), 5) for x in right_fit.params])
print("  worst:", right_fit.worst_points(5))


# The left group is the mirror construction, but intentionally stops at a
# vertical cut line, leaving the measured 7 px slit beside the centre bar.
# The two diagonal inner ends are clipped there, which explains the sharp
# silhouette intersections reported by analyse.
LEFT_ARM_LENGTH = 340.0


def build_left(p):
    (
        upper_x,
        upper_y,
        upper_angle,
        upper_w,
        upper_r,
        lower_x,
        lower_y,
        lower_angle,
        lower_w,
        lower_r,
        cutx,
        x0,
        hcy,
        hh,
        hr,
    ) = p
    shapes = []
    for outer, arm_angle, width, radius in (
        (
            np.array([upper_x, upper_y]),
            upper_angle,
            upper_w,
            upper_r,
        ),
        (
            np.array([lower_x, lower_y]),
            lower_angle,
            lower_w,
            lower_r,
        ),
    ):
        direction = np.array(
            [np.cos(np.radians(arm_angle)), np.sin(np.radians(arm_angle))]
        )
        centre = outer + direction * (LEFT_ARM_LENGTH / 2)
        verts = prim.clip_halfplane(
            prim.oriented_rectangle(
                centre, LEFT_ARM_LENGTH, width, arm_angle
            ),
            (1.0, 0.0),
            cutx,
        )
        radii = [
            radius if vertex[0] < cutx - 0.1 else 0.0 for vertex in verts
        ]
        shapes.append((verts, radii))
    horizontal = prim.rectangle(x0, hcy - hh / 2, cutx - x0, hh)
    shapes.append((horizontal, [hr, 0.0, 0.0, hr]))
    return shapes


left0 = np.array(
    [
        431.0,
        396.5,
        45.3,
        80.0,
        9.0,
        430.5,
        797.0,
        -45.0,
        80.0,
        10.0,
        575.5,
        344.5,
        593.5,
        85.0,
        4.0,
    ]
)
left_bounds = (
    [
        420,
        388,
        42,
        72,
        2,
        420,
        788,
        -49,
        72,
        2,
        570,
        338,
        586,
        79,
        0,
    ],
    [
        440,
        407,
        49,
        88,
        20,
        440,
        807,
        -42,
        88,
        20,
        580,
        351,
        601,
        92,
        12,
    ],
)
left_fit = prim.fit_union(
    contours[1], build_left, left0, bounds=left_bounds
)
print("left:", left_fit.summary())
print("  params:", [round(float(x), 5) for x in left_fit.params])
print("  worst:", left_fit.worst_points(5))


# The detached right ray is an arrow primitive, mirrored about its horizontal
# centre. Its shoulders are deliberately sharp; the tip and far end carry
# the measured circular rounding.
def build_arrow(p):
    tipx, cy, shoulderx, rightx, halfheight, tipr, top_r, bottom_r = p
    verts = np.array(
        [
            [tipx, cy],
            [shoulderx, cy - halfheight],
            [rightx, cy - halfheight],
            [rightx, cy + halfheight],
            [shoulderx, cy + halfheight],
        ]
    )
    return [(verts, [tipr, 0.0, top_r, bottom_r, 0.0])]


arrow0 = np.array([705.0, 594.0, 746.0, 907.5, 42.5, 5.0, 4.0, 8.0])
arrow_bounds = (
    [700, 588, 738, 900, 38, 0, 0, 0],
    [711, 600, 753, 914, 48, 14, 16, 18],
)
arrow_fit = prim.fit_union(
    contours[2], build_arrow, arrow0, bounds=arrow_bounds
)
print("arrow:", arrow_fit.summary())
print("  params:", [round(float(x), 5) for x in arrow_fit.params])
print("  worst:", arrow_fit.worst_points(5))

right_primitives = build_right(right_fit.params)
left_primitives = build_left(left_fit.params)
arrow_primitives = build_arrow(arrow_fit.params)

colours = [flat_colour(rgb, region) for region in regions]
print("paint:", colours)

shapes = [
    {
        "id": "right-burst",
        "type": "path",
        "d": rounded_model_path(right_primitives),
        "fills": [{"type": "solid", "color": colours[0]}],
    },
    {
        "id": "left-burst",
        "type": "path",
        "d": rounded_model_path(left_primitives),
        "fills": [{"type": "solid", "color": colours[1]}],
    },
    {
        "id": "right-arrow",
        "type": "path",
        "d": rounded_model_path(arrow_primitives),
        "fills": [{"type": "solid", "color": colours[2]}],
    },
]

proj = load_project(PROJECT)
proj.shapes = shapes
proj.notes = [
    "Three disconnected groups reconstructed as repeated filled primitives. "
    "The incidental union boundaries are not traced.",
    f"Right group fit: {right_fit.summary()}; left group fit: "
    f"{left_fit.summary()}; arrow fit: {arrow_fit.summary()}.",
    "The left group is clipped to a measured vertical cut line; the detached "
    "right arrow is mirrored about its horizontal centre.",
    "The reconstruction drops the source's soft shadow, edge glow and "
    "low-level fill texture. The model keeps the crisp geometric structure "
    "and median flat paint.",
]
save_project(PROJECT, proj)
print(
    "model:",
    sum(len(shape["d"]) for shape in shapes),
    "nodes,",
    len(shapes),
    "semantic shapes",
)
