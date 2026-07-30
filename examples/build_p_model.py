"""Build work/p: one flat path — the P outline with the plug notched out of
it (the plug opens at the bottom, so it is one contour, not a counter).
Corners are circular fillets, the bowl is two cubics, the plug tapers are
S-cubics between parallel verticals."""

import json
import sys
from pathlib import Path

import numpy as np

from png2svg.model import load_project, save_project

proj_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/p")
M = json.loads((proj_dir / "analysis" / "measurements.json").read_text())
X, Y, FI, B = M["x"], M["y"], M["fillets"], M["bowl"]


class Path_:
    """Accumulates segments; corners are emitted as line + tangent arc."""

    def __init__(self):
        self.segs = []
        self.start = None

    def _pt(self, p):
        return [float(p[0]), float(p[1])]

    def move(self, p):
        self.start = p
        self.segs.append(["M", *self._pt(p)])

    def line(self, p):
        self.segs.append(["L", *self._pt(p)])

    def cubic(self, c1, c2, p):
        self.segs.append(["C", *self._pt(c1), *self._pt(c2), *self._pt(p)])

    def corner(self, vertex, u_in, u_out, r):
        """Fillet of radius r at `vertex`, entered along u_in, left along u_out."""
        v = np.array(vertex, float)
        a = np.array(u_in, float) / np.linalg.norm(u_in)
        b = np.array(u_out, float) / np.linalg.norm(u_out)
        # tangent length for a fillet between two rays meeting at half-angle t
        theta = np.arccos(np.clip(-a @ b, -1, 1)) / 2.0
        t = r / np.tan(theta)
        p0, p1 = v - a * t, v + b * t
        self.line(p0)
        cross = a[0] * b[1] - a[1] * b[0]
        sweep = 1 if cross > 0 else 0
        self.segs.append(["A", float(r), float(r), 0, 0, sweep, *self._pt(p1)])

    def arc(self, p, r, sweep, large=0):
        self.segs.append(["A", float(r), float(r), 0, large, sweep, *self._pt(p)])

    def close(self):
        self.segs.append(["Z"])


DOWN, UP, LEFT, RIGHT = (0, 1), (0, -1), (-1, 0), (1, 0)
P = Path_()

# --- top edge, from just after the stem's top-left fillet -------------------
P.move((X["stem_left"] + FI["stem_tl"]["r"], Y["top"]))
P.line((B["xT"], Y["top"]))

# --- bowl: two cubics, horizontal tangents on the flats, vertical at right --
P.cubic((B["xT"] + B["h1"], Y["top"]), (B["xR"], B["yR"] - B["h2"]),
        (B["xR"], B["yR"]))
P.cubic((B["xR"], B["yR"] + B["h3"]), (B["xB"] + B["h4"], Y["bowl_bottom"]),
        (B["xB"], Y["bowl_bottom"]))

# --- bowl underside, up the neck's right edge -------------------------------
P.corner((X["neck_right"], Y["bowl_bottom"]), LEFT, UP, FI["bowl_bl"]["r"])

# --- right taper (travelling upward, so the S-cubic is reversed) ------------
tr = M["taper_right"]
P.line((X["neck_right"], tr["y1"]))
P.cubic((X["neck_right"], tr["y1"] - tr["h1"]),
        (X["body_right"], tr["y0"] + tr["h0"]), (X["body_right"], tr["y0"]))

# --- plug body top, around both prongs --------------------------------------
P.corner((X["body_right"], Y["body_top"]), UP, LEFT, FI["body_tr"]["r"])
for side in ("R", "L"):
    cap = M[f"prong{side}_cap"]
    P.corner((X[f"prong{side}_right"], Y["body_top"]), LEFT, UP,
             FI[f"prong{side}_ir"]["r"])
    P.line((X[f"prong{side}_right"], cap["cy"]))
    P.arc((X[f"prong{side}_left"], cap["cy"]), cap["r"], 0)
    P.corner((X[f"prong{side}_left"], Y["body_top"]), DOWN, LEFT,
             FI[f"prong{side}_il"]["r"])
P.corner((X["body_left"], Y["body_top"]), LEFT, DOWN, FI["body_tl"]["r"])

# --- left taper (travelling downward) ---------------------------------------
tl = M["taper_left"]
P.line((X["body_left"], tl["y0"]))
P.cubic((X["body_left"], tl["y0"] + tl["h0"]),
        (X["neck_left"], tl["y1"] - tl["h1"]), (X["neck_left"], tl["y1"]))

# --- stem foot and left side -------------------------------------------------
P.corner((X["neck_left"], Y["stem_bottom"]), DOWN, LEFT, FI["stem_br"]["r"])
P.corner((X["stem_left"], Y["stem_bottom"]), LEFT, UP, FI["stem_bl"]["r"])
P.corner((X["stem_left"], Y["top"]), UP, RIGHT, FI["stem_tl"]["r"])
P.close()

shape = {"id": "p-mark", "type": "path", "d": P.segs,
         "fills": [{"type": "solid", "color": M["ink"]}]}

proj = load_project(proj_dir)
proj.shapes = [shape]
proj.notes = [
    "flat 'P' with a plug notched out; the plug opens at the bottom, so the "
    "whole mark is ONE contour rather than a path plus counter",
    f"bowl is two cubics tangent to the flat top (y={Y['top']:.2f}) and flat "
    f"bottom (y={Y['bowl_bottom']:.2f}), vertical tangent at x={B['xR']:.2f}; "
    f"a single ellipse cannot hold it — the upper quarter is ~1px fuller",
    "plug tapers are S-cubics with vertical tangents at both ends",
    f"fillets: stem {FI['stem_tl']['r']:.2f}-{FI['stem_br']['r']:.2f}, "
    f"plug body {FI['body_tl']['r']:.2f}, prong inner "
    f"{FI['prongL_il']['r']:.2f}-{FI['prongR_ir']['r']:.2f}",
]
save_project(proj_dir, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes")
