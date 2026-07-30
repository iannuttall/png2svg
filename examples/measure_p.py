"""Measure work/p (plug 'P' mark).

Straight edges come from line fits. The bowl is two cubic Béziers with
horizontal tangents on the flat top and bottom edges and a vertical tangent
at the right extreme — a single conic cannot hold it, because the upper
quarter is about a pixel fuller than the lower one. The plug is a white
region INSIDE the ink, so its rays start in the plug interior and run
outward. Corner radii come from the vertex-to-arc distance along the corner
bisector, which needs one scan each.

Writes work/p/analysis/measurements.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

from png2svg.compare import foreground_mask
from png2svg.measure import Field, edge_point, edge_samples, fit_line, intersect

proj = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/p")
meta = json.loads((proj / "project.json").read_text())
BG = tuple(meta["source"]["background"][:3])
IMG = proj / "source" / meta["source"]["path"].split("/")[-1]
rgb = np.asarray(Image.open(IMG).convert("RGB"))
F = Field(rgb, BG)

M: dict = {}

# ---------------- straight edges -------------------------------------------
EDGES = {
    "stem_left":    ((82.0, 90.0), (82.0, 440.0), 12.0),
    "top":          ((100.0, 61.5), (265.0, 61.5), 12.0),
    "stem_bottom":  ((100.0, 464.0), (210.0, 464.0), 12.0),
    "neck_left":    ((222.0, 330.0), (222.0, 455.0), 12.0),
    "neck_right":   ((258.0, 330.0), (258.0, 368.0), 10.0),
    "body_left":    ((189.0, 242.0), (189.0, 276.0), 10.0),
    "body_right":   ((294.0, 242.0), (294.0, 280.0), 10.0),
    "prongL_left":  ((207.0, 200.0), (207.0, 226.0), 9.0),
    "prongL_right": ((228.0, 200.0), (228.0, 226.0), 9.0),
    "prongR_left":  ((257.0, 200.0), (257.0, 226.0), 9.0),
    "prongR_right": ((276.0, 200.0), (276.0, 226.0), 9.0),
    "body_top_l":   ((196.0, 231.0), (205.0, 231.0), 8.0),
    "body_top_m":   ((231.0, 231.0), (253.0, 231.0), 9.0),
    "body_top_r":   ((279.0, 231.0), (287.0, 231.0), 8.0),
    "bowl_bottom":  ((264.0, 380.0), (280.0, 380.0), 9.0),
}
L = {}
for name, (p0, p1, off) in EDGES.items():
    pts = edge_samples(F, p0, p1, off, 0.15, 0.85, 24)
    if len(pts) < 6:
        raise SystemExit(f"edge {name}: only {len(pts)} samples")
    c, d, err = fit_line(pts)
    ang = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    vertical = abs(ang - 90) < 45
    # snap to exactly vertical / horizontal: every measured angle is within
    # 0.5 deg of an axis, and the design is plainly axis-aligned
    d = np.array([0.0, 1.0]) if vertical else np.array([1.0, 0.0])
    L[name] = {"c": c, "d": d, "err": err, "pts": pts, "vertical": vertical}
    print(f"  {name:13s} n={len(pts):3d} resid={err:.3f} angle={ang:7.2f} "
          f"at={(c[0] if vertical else c[1]):.3f}")

# the three body-top runs are one line
btp = np.vstack([L[k]["pts"] for k in ("body_top_l", "body_top_m", "body_top_r")])
BODY_TOP = float(btp[:, 1].mean())
print(f"  body_top joint: n={len(btp)} y={BODY_TOP:.3f} spread={np.ptp(btp[:, 1]):.3f}")
for k in ("body_top_l", "body_top_m", "body_top_r"):
    L[k]["c"] = np.array([L[k]["c"][0], BODY_TOP])

X = {k: float(L[k]["c"][0]) for k, v in L.items() if v["vertical"]}
Y = {k: float(L[k]["c"][1]) for k, v in L.items() if not v["vertical"]}
Y["body_top"] = BODY_TOP
M["x"], M["y"] = X, Y

# ---------------- the bowl: two cubics, tangent to the flats ----------------
cx0, cy0, r0 = 272.0, 220.0, 160.0
arc = []
for a in np.radians(np.arange(-86.0, 86.01, 1.0)):
    start = (cx0 + (r0 + 26) * np.cos(a), cy0 + (r0 + 26) * np.sin(a))
    p = edge_point(F, start, (-np.cos(a), -np.sin(a)), 52.0)
    if p:
        arc.append(p)
A = np.array(arc)
Y_TOP, Y_BOT = Y["top"], Y["bowl_bottom"]
ts = np.linspace(0, 1, 260)[:, None]


def bowl_curves(q):
    xT, xB, xR, yR, h1, h2, h3, h4 = q
    c1 = np.array([[xT, Y_TOP], [xT + h1, Y_TOP], [xR, yR - h2], [xR, yR]])
    c2 = np.array([[xR, yR], [xR, yR + h3], [xB + h4, Y_BOT], [xB, Y_BOT]])
    out = []
    for c in (c1, c2):
        out.append((1 - ts) ** 3 * c[0] + 3 * (1 - ts) ** 2 * ts * c[1]
                   + 3 * (1 - ts) * ts ** 2 * c[2] + ts ** 3 * c[3])
    return np.vstack(out)


def bowl_res(q):
    c = bowl_curves(q)
    return np.min(np.linalg.norm(c[None, :, :] - A[:, None, :], axis=2), axis=1)


k = 0.5523 * 160
bfit = optimize.least_squares(bowl_res, [272.0, 272.0, 432.0, 220.0, k, k, k, k])
r = bowl_res(bfit.x)
print("  bowl: n=%d  xT=%.3f xB=%.3f xR=%.3f yR=%.3f  h=(%.2f, %.2f, %.2f, %.2f)"
      % (len(A), *bfit.x))
print("        rms=%.3f max=%.3f" % (np.sqrt((r ** 2).mean()), np.abs(r).max()))
M["bowl"] = dict(zip(("xT", "xB", "xR", "yR", "h1", "h2", "h3", "h4"),
                     [float(v) for v in bfit.x]))
M["bowl"]["y_top"], M["bowl"]["y_bottom"] = Y_TOP, Y_BOT
M["bowl"]["rms"] = float(np.sqrt((r ** 2).mean()))


# ---------------- prong caps: scan outward from inside the prong ------------
def cap_circle(x_l, x_r, y_guess, label):
    """Semicircular cap, constrained tangent to the two prong sides.

    The sides are vertical and the cap must meet them without a kink, so the
    radius IS the half-width and the centre sits midway: only the centre's y
    is free. Fitting r freely gives r slightly greater than the half-width,
    and SVG then cannot place the centre on the chord — it draws a shallower
    arc and the cap lands a pixel low.
    """
    cx = (x_l + x_r) / 2.0
    r = (x_r - x_l) / 2.0
    cy = y_guess + r
    pts = []
    for a in np.radians(np.arange(-176.0, 4.01, 4.0)):
        p = edge_point(F, (cx, cy), (np.cos(a), np.sin(a)), 40.0)
        if p:
            pts.append(p)
    P = np.array(pts)

    def res(q):
        return np.hypot(P[:, 0] - cx, P[:, 1] - q[0]) - r

    fit = optimize.least_squares(res, [cy])
    rr = res(fit.x)
    free = optimize.least_squares(
        lambda q: np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) - q[2], [cx, cy, r])
    print("  %s cap: n=%d cx=%.3f cy=%.3f r=%.3f (tangency-constrained) "
          "rms=%.3f max=%.3f  [free fit r=%.3f rms=%.3f]"
          % (label, len(P), cx, fit.x[0], r, np.sqrt((rr ** 2).mean()),
             np.abs(rr).max(), free.x[2],
             np.sqrt((np.hypot(P[:, 0] - free.x[0], P[:, 1] - free.x[1])
                      - free.x[2]) ** 2).mean() ** 0.5))
    return {"cx": float(cx), "cy": float(fit.x[0]), "r": float(r)}


M["prongL_cap"] = cap_circle(X["prongL_left"], X["prongL_right"], 186.0, "prongL")
M["prongR_cap"] = cap_circle(X["prongR_left"], X["prongR_right"], 186.0, "prongR")


# ---------------- plug tapers: S-cubic between two parallel verticals -------
def s_cubic(x_from, x_to, y_lo, y_hi, direction, label):
    """Cubic with vertical tangents at both ends, fitted between two verticals.

    `direction` is +1 to scan right from the plug interior, -1 to scan left.
    """
    pts = []
    for y in np.arange(y_lo, y_hi, 1.0):
        p = edge_point(F, (240.0, y), (direction, 0), 70.0)
        if p:
            pts.append(p)
    P = np.array(pts)
    tt = np.linspace(0, 1, 220)[:, None]

    def curve(q):
        y0, y1, h0, h1 = q
        p0 = np.array([x_from, y0])
        p1 = np.array([x_from, y0 + h0])
        p2 = np.array([x_to, y1 - h1])
        p3 = np.array([x_to, y1])
        return ((1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * p1
                + 3 * (1 - tt) * tt ** 2 * p2 + tt ** 3 * p3)

    def res(q):
        c = curve(q)
        return np.min(np.linalg.norm(c[None, :, :] - P[:, None, :], axis=2), axis=1)

    fit = optimize.least_squares(res, [y_lo + 6, y_hi - 6, 18.0, 18.0],
                                 bounds=([y_lo - 25, y_lo - 25, 0, 0],
                                         [y_hi + 25, y_hi + 45, 90, 90]))
    rr = res(fit.x)
    print("  %s taper: n=%d y0=%.3f y1=%.3f h0=%.3f h1=%.3f rms=%.3f max=%.3f"
          % (label, len(P), *fit.x, np.sqrt((rr ** 2).mean()), np.abs(rr).max()))
    return dict(zip(("y0", "y1", "h0", "h1"), [float(v) for v in fit.x]))


M["taper_left"] = s_cubic(X["body_left"], X["neck_left"], 268.0, 324.0, -1, "left")
M["taper_right"] = s_cubic(X["body_right"], X["neck_right"], 268.0, 328.0, +1, "right")


# ---------------- corner fillets from the bisector distance -----------------
def fillet(name_a, name_b, u_a, u_b, label):
    """Radius of the fillet between two straight edges.

    u_a / u_b point away from the shared vertex along each edge. For a fillet
    of radius r between edges meeting at half-angle theta, the vertex-to-arc
    distance along the bisector is r/sin(theta) - r, so one scan gives r.
    """
    va = (L[name_a]["c"], L[name_a]["d"])
    vb = (L[name_b]["c"], L[name_b]["d"])
    v = intersect(va, vb)
    ua = np.array(u_a, float) / np.linalg.norm(u_a)
    ub = np.array(u_b, float) / np.linalg.norm(u_b)
    bis = (ua + ub)
    bis /= np.linalg.norm(bis)
    theta = np.arccos(np.clip(ua @ ub, -1, 1)) / 2.0
    # which side of the vertex is background?
    inward_is_bg = F.at(*(v + bis * 9)) < F.at(*(v - bis * 9))
    probe = v + bis * 34 if inward_is_bg else v - bis * 34
    d_scan = -bis if inward_is_bg else bis
    p = edge_point(F, tuple(probe), tuple(d_scan), 68.0)
    if p is None:
        raise SystemExit(f"fillet {label}: no boundary found")
    d = float(np.hypot(p[0] - v[0], p[1] - v[1]))
    s = np.sin(theta)
    r = d * s / (1.0 - s)
    print("  fillet %-11s vertex=(%.2f, %.2f) d=%.3f r=%.3f" % (label, *v, d, r))
    return {"vx": float(v[0]), "vy": float(v[1]), "r": float(r)}


DOWN, UP, LEFT, RIGHT = (0, 1), (0, -1), (-1, 0), (1, 0)
M["fillets"] = {
    "stem_tl": fillet("top", "stem_left", RIGHT, DOWN, "stem_tl"),
    "stem_bl": fillet("stem_bottom", "stem_left", RIGHT, UP, "stem_bl"),
    "stem_br": fillet("stem_bottom", "neck_left", LEFT, UP, "stem_br"),
    "bowl_bl": fillet("bowl_bottom", "neck_right", RIGHT, UP, "bowl_bl"),
    "body_tl": fillet("body_top_l", "body_left", RIGHT, DOWN, "body_tl"),
    "body_tr": fillet("body_top_r", "body_right", LEFT, DOWN, "body_tr"),
    "prongL_il": fillet("body_top_l", "prongL_left", LEFT, UP, "prongL_il"),
    "prongL_ir": fillet("body_top_m", "prongL_right", RIGHT, UP, "prongL_ir"),
    "prongR_il": fillet("body_top_m", "prongR_left", LEFT, UP, "prongR_il"),
    "prongR_ir": fillet("body_top_r", "prongR_right", RIGHT, UP, "prongR_ir"),
}

# ---------------- ink colour ------------------------------------------------
mask = foreground_mask(Image.open(IMG), BG)
core = ndimage.distance_transform_edt(mask) >= 6
med = np.median(rgb[core], axis=0).astype(int)
M["ink"] = "#{:02x}{:02x}{:02x}".format(*med)
print("  ink:", M["ink"], " core px", int(core.sum()))


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, float)):
        return round(float(o), 4)
    return o


(proj / "analysis").mkdir(exist_ok=True)
(proj / "analysis" / "measurements.json").write_text(json.dumps(clean(M), indent=2) + "\n")
print("wrote measurements.json")
