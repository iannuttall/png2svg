"""Extract the plug on its own, as a menu-bar icon.

The plug exists in work/p only as a notch cut out of the P, so every edge of
it is already measured — this rebuilds the same geometry as a filled shape
instead of a hole. The one thing that cannot be measured is its bottom: in
the P the cord runs off and is cut by the bowl, so the end is invented here
and rounded, which reads as intentional where a flat chop reads as a
cropping mistake.

Writes output-plug.svg (rounded cord) and output-plug-flat.svg.
"""

import json
from pathlib import Path

import numpy as np

from png2svg.model import Project, save_project
from png2svg.svggen import generate_svg

M = json.loads(Path("work/p/analysis/measurements.json").read_text())
X, Y, FI = M["x"], M["y"], M["fillets"]
TL, TR = M["taper_left"], M["taper_right"]
PAD = 8.0          # breathing room so the glyph does not touch the edge


def corner(segs, vertex, u_in, u_out, r):
    """Line to the fillet's start, then the arc to its end.

    The sweep comes from the turn itself — the cross product of the incoming
    and outgoing directions — rather than being asserted per corner. Hand-set
    flags invert silently and turn a fillet into a notch, which is exactly
    what a plug's prong shoulders look like when you get it wrong.
    """
    v, a, b = (np.array(t, float) for t in (vertex, u_in, u_out))
    p0, p1 = v - a * r, v + b * r
    segs.append(["L", float(p0[0]), float(p0[1])])
    sweep = 1 if a[0] * b[1] - a[1] * b[0] > 0 else 0
    segs.append(["A", r, r, 0, 0, sweep, float(p1[0]), float(p1[1])])


def build(ending: str):
    """ending: 'cord' | 'stub' | 'none' | 'flat'."""
    DOWN, UP, LEFT, RIGHT = (0, 1), (0, -1), (-1, 0), (1, 0)
    body_top = Y["body_top"]
    cord_w = X["neck_right"] - X["neck_left"]
    taper_end = max(TL["y1"], TR["y1"])
    # In the P the cord runs down to where the bowl's underside cuts it, so
    # that is its full read. Shorter endings stop at the taper instead: the
    # waist is what makes this plug recognisable, so every variant keeps it.
    foot = {"cord": Y["bowl_bottom"],
            "flat": Y["bowl_bottom"],
            "stub": taper_end + cord_w * 0.85,
            "none": taper_end + cord_w * 0.10}[ending]
    round_cord = ending != "flat"
    s = []

    s.append(["M", X["body_left"], body_top + FI["body_tl"]["r"]])
    if ending == "none":
        # No cord at all: the taper is dropped and the body simply rounds off.
        # Keeping the taper and closing it early does not work — the cap's
        # arc starts before the S-curve has finished, so the two sides cross
        # and leave a pair of tails hanging off the bottom.
        y_bot = TL["y0"] + cord_w * 0.25
        r = cord_w * 0.75
        s.append(["L", X["body_left"], y_bot - r])
        s.append(["A", r, r, 0, 0, 0, X["body_left"] + r, y_bot])
        s.append(["L", X["body_right"] - r, y_bot])
        s.append(["A", r, r, 0, 0, 0, X["body_right"], y_bot - r])
        foot = y_bot
    else:
        # left side of the body, down through the taper into the cord
        s.append(["L", X["body_left"], TL["y0"]])
        s.append(["C", X["body_left"], TL["y0"] + TL["h0"],
                  X["neck_left"], TL["y1"] - TL["h1"], X["neck_left"], TL["y1"]])
        if round_cord:
            r = cord_w / 2.0
            s.append(["L", X["neck_left"], foot - r])
            s.append(["A", r, r, 0, 0, 0, X["neck_right"], foot - r])
        else:
            s.append(["L", X["neck_left"], foot])
            s.append(["L", X["neck_right"], foot])
        # back up the right side, out through the taper
        s.append(["L", X["neck_right"], TR["y1"]])
        s.append(["C", X["neck_right"], TR["y1"] - TR["h1"],
                  X["body_right"], TR["y0"] + TR["h0"], X["body_right"], TR["y0"]])
    corner(s, (X["body_right"], body_top), UP, LEFT, FI["body_tr"]["r"])

    # across the top, around both prongs
    for side in ("R", "L"):
        cap = M[f"prong{side}_cap"]
        corner(s, (X[f"prong{side}_right"], body_top), LEFT, UP,
               FI[f"prong{side}_ir"]["r"])
        s.append(["L", X[f"prong{side}_right"], cap["cy"]])
        s.append(["A", cap["r"], cap["r"], 0, 0, 0,
                  X[f"prong{side}_left"], cap["cy"]])
        corner(s, (X[f"prong{side}_left"], body_top), DOWN, LEFT,
               FI[f"prong{side}_il"]["r"])
    corner(s, (X["body_left"], body_top), LEFT, DOWN, FI["body_tl"]["r"])
    s.append(["Z"])
    return s, foot


def write(path: str, ending: str):
    segs, foot = build(ending)
    x0 = X["body_left"] - PAD
    y0 = min(M["prongL_cap"]["cy"] - M["prongL_cap"]["r"],
             M["prongR_cap"]["cy"] - M["prongR_cap"]["r"]) - PAD
    w = (X["body_right"] + PAD) - x0
    h = (foot + PAD) - y0
    shifted = []
    for seg in segs:
        if seg[0] == "A":
            shifted.append(seg[:6] + [seg[6] - x0, seg[7] - y0])
        elif seg[0] == "Z":
            shifted.append(seg)
        else:
            nums = [v - (x0 if i % 2 == 0 else y0) for i, v in enumerate(seg[1:])]
            shifted.append([seg[0]] + nums)
    proj = Project(
        source_path="work/p", width=int(round(w)), height=int(round(h)),
        sha256="", background=[0, 0, 0, 0], view_box=[0, 0, w, h],
        shapes=[{"id": "plug", "type": "path", "d": shifted,
                 "fills": [{"type": "solid", "color": "#000000"}]}],
    )
    proj.validate()
    Path(path).write_text(generate_svg(proj))
    print(f"wrote {path}  viewBox 0 0 {w:.2f} {h:.2f}  "
          f"{len(shifted)} nodes  {len(Path(path).read_text())} bytes")


for name, ending in (("output-plug.svg", "none"),
                     ("output-plug-stub.svg", "stub"),
                     ("output-plug-cord.svg", "cord"),
                     ("output-plug-flat.svg", "flat")):
    write(name, ending)
