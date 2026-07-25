"""Measure work/n (flat 'N' monogram): precise side lines, vertices, and
rounded-corner cubics. Writes work/n/analysis/measurements.json."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg.measure import Field, edge_point, fit_corner_full, fit_line_x_of_y

proj = Path("work/n")
rgb = np.asarray(Image.open(proj / "source" / "in.webp").convert("RGB"))
F = Field(rgb, (255, 255, 255))


def steep_side(m_pred, c_pred, ys, from_right, off=25.0):
    """Fit x = m*y + c for a steep edge. from_right: scan leftward into fg."""
    pts = []
    for y in ys:
        x0 = m_pred * y + c_pred + (off if from_right else -off)
        d = (-1, 0) if from_right else (1, 0)
        p = edge_point(F, (x0, y), d, off * 2)
        if p:
            pts.append(p)
    m, c, err = fit_line_x_of_y(pts)
    return {"m": m, "c": c, "err": round(err, 3), "n": len(pts)}


def flat_side(y_pred, xs, from_below):
    """Mean y of a horizontal edge. from_below: scan upward into fg."""
    pts = []
    for x in xs:
        y0 = y_pred + (18 if from_below else -18)
        d = (0, -1) if from_below else (0, 1)
        p = edge_point(F, (x, y0), d, 36)
        if p:
            pts.append(p[1])
    return {"y": float(np.mean(pts)), "spread": round(float(np.ptp(pts)), 3), "n": len(pts)}


def corner_samples(box, scans):
    """Collect boundary points near a corner from a list of scan specs."""
    pts = []
    for p0, d, tmax in scans:
        p = edge_point(F, p0, d, tmax)
        if p and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]:
            pts.append(p)
    return pts


M: dict = {}

# ---------------- grey head (parallelogram, TR corner rounded) --------------
M["head"] = {
    "top": flat_side(371.9, range(615, 726, 8), False),
    "right": steep_side(0.485, 782.5 - 0.485 * 468.5, range(408, 461, 4), True),
    "bottom": flat_side(468.1, range(695, 776, 8), True),
    "left": steep_side(0.856, 684.5 - 0.856 * 468.5, range(380, 456, 5), False),
}
tr_v = None  # computed in build; corner fit needs vertex -> do here inline
top_y = M["head"]["top"]["y"]
r = M["head"]["right"]
vx = r["m"] * top_y + r["c"]
sam = corner_samples(
    (vx - 60, top_y - 1, vx + 30, top_y + 70),
    [((x, 350), (0, 1), 40) for x in np.arange(vx - 55, vx + 15, 4)]
    + [((r["m"] * y + r["c"] + 25, y), (-1, 0), 50) for y in np.arange(top_y + 4, top_y + 60, 4)],
)
u_in = (1.0, 0.0)  # clockwise: along top edge, then down the right edge
dv = np.array([r["m"], 1.0]) / np.hypot(r["m"], 1.0)
(fit, err) = fit_corner_full((vx, top_y), u_in, tuple(dv), sam)
M["head"]["corner_tr"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                          "err": round(err, 3)}
print("head:", json.dumps(M["head"], indent=1, default=float))

# ---------------- black v-stroke (TL corner rounded, V-notch left) ----------
M["vstroke"] = {
    "top": flat_side(371.9, range(965, 1061, 8), False),
    "right": steep_side(-0.4254, 1070.5 + 0.4254 * 371.5, range(400, 671, 12), True),
    "left_lower": steep_side(0.431, 934.5 - 0.431 * 691.5, range(548, 681, 8), False),
    "left_upper": steep_side(-0.4254, 865.8 + 0.4254 * 531.6, range(462, 525, 4), False),
}
lu = M["vstroke"]["left_upper"]
top_y = M["vstroke"]["top"]["y"]
vx = lu["m"] * top_y + lu["c"]
sam = corner_samples(
    (vx - 30, top_y - 1, vx + 65, top_y + 75),
    [((x, 350), (0, 1), 40) for x in np.arange(vx - 10, vx + 60, 4)]
    + [((lu["m"] * y + lu["c"] - 25, y), (1, 0), 50) for y in np.arange(top_y + 6, top_y + 70, 4)],
)
dv = np.array([-lu["m"], -1.0]) / np.hypot(lu["m"], 1.0)  # travelling up the edge
(fit, err) = fit_corner_full((vx, top_y), tuple(dv), (1.0, 0.0), sam)
M["vstroke"]["corner_tl"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                             "err": round(err, 3)}
print("vstroke:", json.dumps(M["vstroke"], indent=1, default=float))

# ---------------- black wedge (bend + BR rounded) ----------------------------
M["wedge"] = {
    "right_upper": steep_side(0.451, 577.68 - 0.451 * 417.42, range(432, 561, 8), True, 20),
    "right_lower": steep_side(-0.4285, 646.14 + 0.4285 * 578.35, range(588, 731, 8), True, 20),
    "bottom": flat_side(756.1, range(445, 536, 8), True),
    "left": steep_side(-0.4246, 433.5 + 0.4246 * 756.5, range(440, 741, 12), False),
}
ru, rl = M["wedge"]["right_upper"], M["wedge"]["right_lower"]
bend_y = (rl["c"] - ru["c"]) / (ru["m"] - rl["m"])
bend_x = ru["m"] * bend_y + ru["c"]
sam = corner_samples(
    (bend_x - 45, bend_y - 40, bend_x + 10, bend_y + 40),
    [((max(ru["m"] * y + ru["c"], rl["m"] * y + rl["c"]) + 22, y), (-1, 0), 45)
     for y in np.arange(bend_y - 35, bend_y + 35, 3)],
)
d_in = np.array([ru["m"], 1.0]) / np.hypot(ru["m"], 1.0)
d_out = np.array([rl["m"], 1.0]) / np.hypot(rl["m"], 1.0)
(fit, err) = fit_corner_full((bend_x, bend_y), tuple(d_in), tuple(d_out), sam, t0=15.0)
M["wedge"]["corner_bend"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                             "err": round(err, 3)}
bot_y = M["wedge"]["bottom"]["y"]
vx = rl["m"] * bot_y + rl["c"]
sam = corner_samples(
    (vx - 45, bot_y - 60, vx + 40, bot_y + 1),
    [((x, 775), (0, -1), 40) for x in np.arange(vx - 40, vx + 20, 3)]
    + [((rl["m"] * y + rl["c"] + 22, y), (-1, 0), 45) for y in np.arange(bot_y - 55, bot_y - 4, 3)],
)
(fit, err) = fit_corner_full((vx, bot_y), tuple(d_out), (-1.0, 0.0), sam)
M["wedge"]["corner_br"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                           "err": round(err, 3)}
print("wedge:", json.dumps(M["wedge"], indent=1, default=float))

# ---------------- grey bar (BR + BL rounded) ---------------------------------
M["bar"] = {
    "top": flat_side(508.9, range(675, 796, 8), False),
    "right": steep_side(0.4566, 806.5 - 0.4566 * 508.5, range(525, 721, 10), True),
    "bottom": flat_side(756.1, range(795, 876, 8), True),
    "left": steep_side(0.488, 783.5 - 0.488 * 756.5, range(525, 741, 10), False),
}
rr, ll = M["bar"]["right"], M["bar"]["left"]
bot_y = M["bar"]["bottom"]["y"]
vx = rr["m"] * bot_y + rr["c"]
sam = corner_samples(
    (vx - 45, bot_y - 60, vx + 40, bot_y + 1),
    [((x, 775), (0, -1), 40) for x in np.arange(vx - 40, vx + 15, 3)]
    + [((rr["m"] * y + rr["c"] + 22, y), (-1, 0), 45) for y in np.arange(bot_y - 55, bot_y - 4, 3)],
)
d_in = np.array([rr["m"], 1.0]) / np.hypot(rr["m"], 1.0)
(fit, err) = fit_corner_full((vx, bot_y), tuple(d_in), (-1.0, 0.0), sam)
M["bar"]["corner_br"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                         "err": round(err, 3)}
vx = ll["m"] * bot_y + ll["c"]
sam = corner_samples(
    (vx - 30, bot_y - 45, vx + 45, bot_y + 1),
    [((x, 775), (0, -1), 40) for x in np.arange(vx - 8, vx + 40, 3)]
    + [((ll["m"] * y + ll["c"] - 22, y), (1, 0), 45) for y in np.arange(bot_y - 40, bot_y - 4, 3)],
)
d_out = -np.array([ll["m"], 1.0]) / np.hypot(ll["m"], 1.0)  # travelling up the left edge
(fit, err) = fit_corner_full((vx, bot_y), (-1.0, 0.0), tuple(d_out), sam, t0=12.0)
M["bar"]["corner_bl"] = {"t_in": fit[0], "t_out": fit[1], "h": [fit[2], fit[3]],
                         "err": round(err, 3)}
print("bar:", json.dumps(M["bar"], indent=1, default=float))

# ---------------- colours (median of eroded interiors, watermark-robust) -----
from scipy import ndimage
from png2svg.compare import foreground_mask

mask = foreground_mask(Image.open(proj / "source" / "in.webp"), (255, 255, 255))
lab, n = ndimage.label(ndimage.binary_fill_holes(mask))
colours = {}
for i in range(1, n + 1):
    comp = lab == i
    if comp.sum() < 1000:
        continue
    interior = ndimage.distance_transform_edt(comp) >= 5
    med = np.median(rgb[interior], axis=0).astype(int)
    ys, xs = np.nonzero(comp)
    colours[f"comp{i}"] = {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "color": "#{:02x}{:02x}{:02x}".format(*med),
    }
M["colours"] = colours
print("colours:", json.dumps(colours, indent=1))


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return round(float(o), 4)
    if isinstance(o, float):
        return round(o, 4)
    return o


(proj / "analysis" / "measurements.json").write_text(json.dumps(clean(M), indent=2) + "\n")
print("wrote measurements.json")
