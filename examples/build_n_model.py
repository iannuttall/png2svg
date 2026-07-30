"""Build work/n model from analysis/measurements.json: four flat polygons
with measured side lines, intersected vertices, and fitted corner cubics."""

import json
import sys
from pathlib import Path

import numpy as np

from png2svg.model import load_project, save_project

proj_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/n")
M = json.loads((proj_dir / "analysis" / "measurements.json").read_text())


def vertex(a, b):
    """Intersection of two sides (steep: x=m*y+c, flat: y=Y)."""
    if "y" in a and "y" not in b:
        return (b["m"] * a["y"] + b["c"], a["y"])
    if "y" in b and "y" not in a:
        return (a["m"] * b["y"] + a["c"], b["y"])
    y = (b["c"] - a["c"]) / (a["m"] - b["m"])
    return (a["m"] * y + a["c"], y)


def travel(side, sign):
    """Unit travel direction along a side. flat: sign=+1 means +x;
    steep: sign=+1 means downward (+y)."""
    if "y" in side:
        return np.array([sign, 0.0])
    d = np.array([side["m"], 1.0]) / np.hypot(side["m"], 1.0)
    return d * sign


def polygon(sides, signs, corners):
    """Clockwise path. sides[i] travelled with signs[i]; corners[i] applies
    at the vertex between sides[i] and sides[i+1] (None = sharp)."""
    n = len(sides)
    verts = [vertex(sides[i], sides[(i + 1) % n]) for i in range(n)]
    segs = []
    for i in range(n):
        v = np.array(verts[i])
        c = corners[i]
        u_in = travel(sides[i], signs[i])
        u_out = travel(sides[(i + 1) % n], signs[(i + 1) % n])
        if c is None:
            segs.append(["L", float(v[0]), float(v[1])])
        else:
            p0 = v - u_in * c["t_in"]
            p3 = v + u_out * c["t_out"]
            p1 = p0 + u_in * c["h"][0]
            p2 = p3 - u_out * c["h"][1]
            segs.append(["L", float(p0[0]), float(p0[1])])
            segs.append(["C", float(p1[0]), float(p1[1]),
                         float(p2[0]), float(p2[1]), float(p3[0]), float(p3[1])])
    segs[0][0] = "M"
    segs.append(["Z"])
    return segs


GREY = M["colours"]["comp1"]["color"]
BLACK = M["colours"]["comp2"]["color"]

h = M["head"]
head = {
    "id": "head", "type": "path",
    "d": polygon(
        [h["top"], h["right"], h["bottom"], h["left"]],
        [+1, +1, -1, -1],
        [h["corner_tr"], None, None, None],
    ),
    "fills": [{"type": "solid", "color": GREY}],
}

v = M["vstroke"]
vstroke = {
    "id": "vstroke", "type": "path",
    "d": polygon(
        [v["top"], v["right"], v["left_lower"], v["left_upper"]],
        [+1, +1, -1, -1],
        [None, None, None, v["corner_tl"]],
    ),
    "fills": [{"type": "solid", "color": BLACK}],
}

w = M["wedge"]
wedge = {
    "id": "wedge", "type": "path",
    "d": polygon(
        [w["right_upper"], w["right_lower"], w["bottom"], w["left"]],
        [+1, +1, -1, -1],
        [w["corner_bend"], w["corner_br"], None, None],
    ),
    "fills": [{"type": "solid", "color": BLACK}],
}

b = M["bar"]
bar = {
    "id": "bar", "type": "path",
    "d": polygon(
        [b["top"], b["right"], b["bottom"], b["left"]],
        [+1, +1, -1, -1],
        [None, b["corner_br"], b["corner_bl"], None],
    ),
    "fills": [{"type": "solid", "color": GREY}],
}

proj = load_project(proj_dir)
proj.shapes = [head, vstroke, wedge, bar]
proj.notes = [
    "flat 'N' monogram: 4 polygons, 2 colours; edges in +0.45 / -0.424 slope families",
    "source carries a diagonal stock-watermark grid: reconstructed CLEAN, so",
    "deltaE along watermark lines is expected and intentional",
]
save_project(proj_dir, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes")
