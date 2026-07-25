"""Deterministic first-pass analysis of a source PNG.

Produces analysis/features.json + overlay images: connected components,
traced boundary contours segmented into lines / circular arcs / leftover
curves, corner locations, and a paint-classification probe per component
(flat / linear / angular / complex).

This is a PROPOSAL layer: coordinates are mask-derived (±0.5px). Precise
geometry should be refined with png2svg.measure on the regions this reports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from .compare import composite_over, foreground_mask

# ------------------------------------------------------------------ contours

_MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def trace_boundary(mask: np.ndarray) -> np.ndarray:
    """Ordered outer boundary of a connected component (Moore tracing).

    Returns Nx2 float array of (x, y) in SVG-ish space (pixel index + 0.5).
    """
    ys, xs = np.nonzero(mask)
    sy = ys.min()
    sx = xs[ys == sy].min()
    start = (sy, sx)
    contour = [start]
    prev_dir = 2  # pretend we arrived moving East: clockwise trace goes E first
    cur = start
    H, W = mask.shape
    for _ in range(mask.sum() * 4 + 8):
        found = False
        for k in range(8):
            # clockwise search starting just after the came-from neighbour
            d = (prev_dir + 5 + k) % 8
            ny, nx = cur[0] + _MOORE[d][0], cur[1] + _MOORE[d][1]
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                cur = (ny, nx)
                prev_dir = d
                contour.append(cur)
                found = True
                break
        if not found:  # isolated pixel
            break
        if cur == start and len(contour) > 2:
            break
    pts = np.array([(x + 0.5, y + 0.5) for y, x in contour[:-1]], dtype=float)
    return pts


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker simplification of an open polyline."""
    if len(points) < 3:
        return points
    p0, p1 = points[0], points[-1]
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    if seg_len == 0:
        d = np.linalg.norm(points - p0, axis=1)
    else:
        rel = points - p0
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
    i = int(np.argmax(d))
    if d[i] > epsilon:
        left = rdp(points[: i + 1], epsilon)
        right = rdp(points[i:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([p0, p1])


# ------------------------------------------------------------------ segments


def _fit_line(pts: np.ndarray):
    centre = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - centre)
    direction = vt[0]
    rel = pts - centre
    err = np.abs(direction[0] * rel[:, 1] - direction[1] * rel[:, 0])
    t = (pts - centre) @ direction
    return {
        "kind": "line",
        "p0": (centre + direction * t.min()).round(2).tolist(),
        "p1": (centre + direction * t.max()).round(2).tolist(),
        "max_err": round(float(err.max()), 3),
    }


def _fit_arc(pts: np.ndarray):
    def res(p):
        return np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2]

    p0 = [pts[:, 0].mean(), pts[:, 1].mean(), np.ptp(pts, axis=0).max()]
    try:
        out = optimize.least_squares(res, p0, max_nfev=200)
    except Exception:
        return None
    cx, cy, r = out.x
    if not (1.0 < r < 1e5):
        return None
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return {
        "kind": "arc",
        "centre": [round(float(cx), 2), round(float(cy), 2)],
        "radius": round(float(r), 2),
        "angle_first": round(float(angles[0]), 1),
        "angle_last": round(float(angles[-1]), 1),
        "p0": pts[0].round(2).tolist(),
        "p1": pts[-1].round(2).tolist(),
        "max_err": round(float(np.abs(res(out.x)).max()), 3),
    }


def segment_contour(
    contour: np.ndarray,
    epsilon: float = 0.8,
    line_min: float = 45.0,
    arc_tol: float = 1.2,
    corner_span: float = 75.0,
):
    """Classify the closed contour into line / arc / curve segments.

    RDP-simplify, then treat long simplified edges as straight lines and
    chains of short edges as curved regions (an RDP chord on radius-r arc is
    ~2*sqrt(2*r*eps) long, so gentle arcs and corner roundings both fall
    below line_min). Curved regions get a circle fit; short ones are corner
    roundings. Pointwise turn angles can't do this: squircle corners and
    r>200px cap arcs both hide below any usable per-point threshold.
    """
    n = len(contour)
    if n < 8:
        return [], 0
    # cut the closed loop at the point farthest from the centroid (always on
    # the hull, so never mid-way through a straight edge's interior stairstep)
    cut = int(np.argmax(np.linalg.norm(contour - contour.mean(axis=0), axis=1)))
    rolled = np.roll(contour, -cut, axis=0)
    closed = np.vstack([rolled, rolled[:1]])
    simp = rdp(closed, epsilon)

    # map simplified vertices back to indices in `rolled`
    idx_map = []
    j = 0
    for v in simp:
        while j < len(closed) and not np.array_equal(closed[j], v):
            j += 1
        idx_map.append(min(j, n))

    edges = []  # (i0, i1, straight?)
    for k in range(len(simp) - 1):
        length = float(np.linalg.norm(simp[k + 1] - simp[k]))
        edges.append([idx_map[k], idx_map[k + 1], length >= line_min])

    # merge consecutive curved edges into regions
    merged = []
    for e in edges:
        if merged and not e[2] and not merged[-1][2]:
            merged[-1][1] = e[1]
        else:
            merged.append(list(e))
    # wrap-around: if first and last are both curved, join them
    wrapped = None
    if len(merged) > 1 and not merged[0][2] and not merged[-1][2]:
        wrapped = (merged[-1][0], merged[0][1])
        merged = merged[1:-1]

    segments = []
    n_corners = 0
    regions = [(m[0], m[1], m[2]) for m in merged]
    if wrapped:
        regions.append((wrapped[0], wrapped[1] + n, False))
    for i0, i1, straight in regions:
        pts = rolled.take(range(i0, i1 + 1), axis=0, mode="wrap")
        if len(pts) < 4:
            continue
        if straight:
            segments.append(_fit_line(pts))
            continue
        # discriminate corner roundings from real arcs by traversed path
        # length: a closed/near-closed arc has tiny endpoint span but long path
        span = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        arc = _fit_arc(pts)
        if arc and arc["max_err"] <= arc_tol and span > corner_span:
            segments.append(arc)
        elif span <= corner_span:
            n_corners += 1
            segments.append({
                "kind": "corner",
                "p0": pts[0].round(2).tolist(),
                "p1": pts[-1].round(2).tolist(),
                "span": round(span, 1),
                "arc_radius": arc["radius"] if arc and arc["max_err"] <= arc_tol else None,
            })
        else:
            segments.append({
                "kind": "curve",
                "p0": pts[0].round(2).tolist(),
                "p1": pts[-1].round(2).tolist(),
                "n_points": len(pts),
                "arc_err": arc["max_err"] if arc else None,
            })
    return _merge_collinear(segments), n_corners


def _merge_collinear(segments: list[dict], tol_deg: float = 2.0) -> list[dict]:
    """Merge consecutive line segments that continue in the same direction."""
    out: list[dict] = []
    for s in segments:
        if s["kind"] == "line" and out and out[-1]["kind"] == "line":
            prev = out[-1]
            v1 = np.subtract(prev["p1"], prev["p0"])
            v2 = np.subtract(s["p1"], s["p0"])
            gap = float(np.linalg.norm(np.subtract(s["p0"], prev["p1"])))
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            if gap < 3.0 and cosang > math.cos(math.radians(tol_deg)):
                prev["p1"] = s["p1"]
                prev["max_err"] = max(prev["max_err"], s["max_err"])
                continue
        out.append(s)
    return out


# -------------------------------------------------------------- paint probes


def classify_paint(rgb: np.ndarray, comp_mask: np.ndarray, arc_centres):
    """Classify a component's fill: flat / linear / angular / complex."""
    interior = ndimage.distance_transform_edt(comp_mask) >= 3.0
    if interior.sum() < 50:
        interior = comp_mask
    vals = rgb[interior].astype(float)
    ys, xs = np.nonzero(interior)
    med = np.median(vals, axis=0)

    if vals.std(axis=0).max() < 2.5:
        return {
            "kind": "flat",
            "color": "#{:02x}{:02x}{:02x}".format(*(int(v) for v in med)),
            "residual": round(float(vals.std(axis=0).max()), 2),
        }

    # linear probe: fit a plane per channel, take the dominant gradient dir
    A = np.column_stack([xs, ys, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    grad = coef[:2]  # 2x3: d/dx, d/dy per channel
    direction = grad[:, np.argmax(np.abs(grad).sum(axis=0))]
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        d = direction / norm
        t = xs * d[0] + ys * d[1]
        order = np.argsort(t)
        t_sorted, v_sorted = t[order], vals[order]
        bins = np.linspace(t_sorted[0], t_sorted[-1], 48)
        prof = [
            v_sorted[(t_sorted >= a) & (t_sorted < b)].mean(axis=0)
            for a, b in zip(bins, bins[1:])
            if ((t_sorted >= a) & (t_sorted < b)).any()
        ]
        prof = np.array(prof)
        pred = np.array([np.interp(t, bins[: len(prof)], prof[:, c]) for c in range(3)]).T
        resid = np.abs(vals - pred).mean()
        if resid < 3.0:
            return {
                "kind": "linear",
                "direction": [round(float(d[0]), 4), round(float(d[1]), 4)],
                "residual": round(float(resid), 2),
            }

    # angular probe around detected arc centres: constant along radius,
    # varying with angle
    for cx, cy in arc_centres:
        ang = np.arctan2(ys - cy, xs - cx)
        rad = np.hypot(xs - cx, ys - cy)
        if rad.max() < 20:
            continue
        nb = 72
        abins = np.floor((ang + np.pi) / (2 * np.pi) * nb).astype(int).clip(0, nb - 1)
        radial_spread = []
        angular_means = np.zeros((nb, 3))
        counts = np.zeros(nb)
        for b in range(nb):
            sel = abins == b
            if sel.sum() < 20:
                continue
            radial_spread.append(vals[sel].std(axis=0).max())
            angular_means[b] = vals[sel].mean(axis=0)
            counts[b] = sel.sum()
        if len(radial_spread) < 8:
            continue
        used = counts > 0
        across = angular_means[used].std(axis=0).max()
        along = float(np.median(radial_spread))
        if along < 6.0 and across > 3 * along:
            return {
                "kind": "angular",
                "centre": [round(float(cx), 2), round(float(cy), 2)],
                "radial_spread": round(along, 2),
                "angular_spread": round(float(across), 2),
            }

    return {"kind": "complex", "note": "not flat/linear/angular; needs judgment"}


# --------------------------------------------------------------------- main


def analyse_image(img: Image.Image, background) -> tuple[dict, Image.Image]:
    rgb = composite_over(img, tuple(background[:3]))
    mask = foreground_mask(img, tuple(background[:3]))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((3, 3))))
    lab, n = ndimage.label(mask)

    overlay = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    features = {"background": list(background), "components": []}

    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 40:
            continue
        ys, xs = np.nonzero(comp)
        contour = trace_boundary(comp)
        segments, n_corners = segment_contour(contour)
        arc_centres = [s["centre"] for s in segments if s["kind"] == "arc"]
        paint = classify_paint(rgb, comp, arc_centres)
        features["components"].append({
            "id": i,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area_px": int(comp.sum()),
            "paint": paint,
            "n_contour_points": len(contour),
            "n_corners": n_corners,
            "segments": segments,
        })
        # overlay: lines red, arc chords blue, curves yellow, corners white
        for s in segments:
            colour = {"line": (255, 60, 60), "arc": (60, 120, 255),
                      "curve": (255, 220, 40), "corner": (255, 255, 255)}[s["kind"]]
            draw.line([tuple(s["p0"]), tuple(s["p1"])], fill=colour, width=2)

    return features, overlay
