"""Segment a subpixel contour into the primitives a designer would have drawn.

`curves.fit_bezier_chain` fits anything, which is exactly its weakness on
designed artwork: a shape made of straight runs joined by round corners comes
back as a hundred cubics chasing contour noise. This module prefers structure
— it takes the longest straight run it can, then the longest circular arc,
and only falls back to Béziers where the outline is genuinely free-form.

Tolerance is the one dial that matters, and it is a deliberate choice rather
than something to infer: it trades segment count against deviation, and the
knee sits where the artwork's own structure runs out. Sweep it, look at both
numbers, and pick — 0.25 to 0.6 px suits most sources. `estimate_noise`
reports the contour's local noise as a floor, but it measures smoothness
between neighbouring samples, not how far the artwork departs from a
primitive, so it cannot choose the tolerance for you.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .curves import corner_indices, fit_run


def estimate_noise(pts: np.ndarray, window: int = 13) -> float:
    """Typical deviation of the contour from locally straight.

    Fits a line over short sliding windows and takes the median of their
    maximum residuals. Windows that straddle a corner or a curve inflate,
    which is why this is a median: on a designed outline most windows sit on
    straight or gently curved stretches and report the measurement noise.
    """
    n = len(pts)
    if n < window * 2:
        return 0.15
    res = []
    for i in range(0, n, max(window // 2, 1)):
        seg = pts.take(range(i, i + window), axis=0, mode="wrap")
        h = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        if h[-1] < 1e-9:
            continue
        # quadratic, not linear: a straight fit counts the window's curvature
        # as noise, so a curved outline would set its own tolerance far too
        # loose while a boxy one sets it far too tight
        A = np.stack([np.ones_like(h), h, h ** 2], axis=1)
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        res.append(float(np.linalg.norm(seg - A @ coef, axis=1).max()))
    return float(np.median(res)) if res else 0.15


def _line_error(seg: np.ndarray) -> float:
    d = seg[-1] - seg[0]
    L = np.linalg.norm(d)
    if L < 1e-9:
        return 0.0
    nrm = np.array([-d[1], d[0]]) / L
    return float(np.abs((seg - seg[0]) @ nrm).max())


def _fit_arc(seg: np.ndarray):
    """Circle through a span; returns ((cx, cy, r), max radial residual)."""
    def res(q):
        return np.hypot(seg[:, 0] - q[0], seg[:, 1] - q[1]) - q[2]

    mid = seg[len(seg) // 2]
    chord = seg[-1] - seg[0]
    guess_r = max(np.linalg.norm(chord), 1.0)
    nrm = np.array([-chord[1], chord[0]])
    if np.linalg.norm(nrm) > 1e-9:
        nrm = nrm / np.linalg.norm(nrm)
    p0 = [mid[0] + nrm[0] * guess_r, mid[1] + nrm[1] * guess_r, guess_r]
    try:
        out = optimize.least_squares(res, p0, max_nfev=200)
    except Exception:
        return None, np.inf
    return tuple(float(v) for v in out.x), float(np.abs(res(out.x)).max())


def _longest_fit(pts: np.ndarray, start: int, tol: float, kind: str,
                 min_len: int) -> tuple[int, object]:
    """Furthest index reachable from `start` while `kind` stays within tol."""
    n = len(pts)
    lo, hi = start + min_len, n - 1
    best_end, best_fit = -1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        seg = pts[start:mid + 1]
        if kind == "line":
            err, fit = _line_error(seg), None
        else:
            fit, err = _fit_arc(seg)
        if err <= tol:
            best_end, best_fit = mid, fit
            lo = mid + 1
        else:
            hi = mid - 1
    return best_end, best_fit


def segment_run(pts: np.ndarray, tol: float, min_line: int = 5,
                min_arc: int = 6) -> list[dict]:
    """Segment one corner-to-corner run into typed primitives.

    The minimum spans are deliberately small. Contour points sit about a
    pixel apart, so a 4px fillet is only six of them; demanding eighteen
    makes every small corner unfittable, drops the rest of its run into the
    free-form fallback, and fragments a clean outline into dozens of cubics.
    Nonsense short fits are rejected by tolerance and the radius sanity
    check instead.
    """
    out: list[dict] = []
    i, n = 0, len(pts)
    while i < n - 1:
        remaining = n - 1 - i
        if remaining < 3:
            out.append({"kind": "line", "p0": pts[i], "p1": pts[-1]})
            break
        line_end, _ = _longest_fit(pts, i, tol, "line", min(min_line, remaining))
        arc_end, arc_fit = _longest_fit(pts, i, tol, "arc", min(min_arc, remaining))
        # A nearly-straight run fits a circle of enormous radius, which then
        # beats the line on span and litters the model with 3-million-unit
        # arcs. Such a fit is a line that has not admitted it yet.
        if arc_fit is not None:
            chord = np.linalg.norm(pts[arc_end] - pts[i]) if arc_end > i else 1.0
            if arc_fit[2] > 40.0 * max(chord, 1.0):
                arc_end, arc_fit = -1, None
        # prefer whichever covers more, but only take the arc when it covers
        # meaningfully more — the simpler primitive wins a near tie
        take_arc = arc_fit is not None and arc_end > i and (
            arc_end - i) > 1.15 * max(line_end - i, 0)
        if take_arc:
            cx, cy, r = arc_fit
            c = np.array([cx, cy])
            span = pts[i:arc_end + 1]
            # the traversal itself gives the sweep: unwrap the angles of the
            # points the arc was fitted to. Deriving it from the endpoints
            # alone cannot tell a minor arc from a major one.
            ang = np.unwrap(np.arctan2(span[:, 1] - cy, span[:, 0] - cx))
            sweep_angle = float(ang[-1] - ang[0])
            out.append({"kind": "arc", "p0": pts[i], "p1": pts[arc_end],
                        "c": c, "r": r, "i0": i, "i1": arc_end,
                        "sweep_angle": sweep_angle})
            i = arc_end
        elif line_end > i:
            out.append({"kind": "line", "p0": pts[i], "p1": pts[line_end],
                        "i0": i, "i1": line_end})
            i = line_end
        else:
            # genuinely free-form from here to the end of the run
            cur = pts[i]
            for seg in fit_run(pts[i:], tol):
                if seg[0] == "L":
                    out.append({"kind": "line", "p0": cur, "p1": seg[1]})
                    cur = seg[1]
                else:
                    out.append({"kind": "cubic", "p0": cur, "c1": seg[1],
                                "c2": seg[2], "p1": seg[3]})
                    cur = seg[3]
            break
    return out


def segment_outline(pts: np.ndarray, tol: float | None = None,
                    corner_deg: float = 35.0, noise_factor: float = 2.5) -> list[dict]:
    """Split a closed subpixel contour into lines, arcs and cubics.

    `tol` defaults to `noise_factor` times the contour's measured noise —
    below that a fit is chasing the measurement, not the artwork.
    """
    pts = np.asarray(pts, float)
    if tol is None:
        tol = max(estimate_noise(pts) * noise_factor, 0.12)
    cuts = corner_indices(pts, corner_deg) or [0]
    prims: list[dict] = []
    for k in range(len(cuts)):
        i0, i1 = cuts[k], cuts[(k + 1) % len(cuts)]
        run = pts[i0:i1 + 1] if i1 > i0 else np.vstack([pts[i0:], pts[:i1 + 1]])
        if len(run) >= 2:
            prims.extend(segment_run(run, tol))
    return prims


def summarise(prims: list[dict]) -> str:
    counts: dict[str, int] = {}
    for p in prims:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Constraint snapping
#
# A fitted outline is a set of independent primitives that happen to sit near
# each other. A drawn one has structure: edges share directions, corners share
# radii, arcs meet lines tangentially. Recovering that structure is what takes
# a reconstruction from "close" to exact, because every constraint removes a
# free parameter that was otherwise absorbing measurement noise.
#
# Nothing here is applied blindly: `snap_outline` reports what it changed so
# the caller can check that each constraint improved the fit rather than
# assuming it.
# ---------------------------------------------------------------------------


def _line_dir(p: dict) -> np.ndarray:
    d = p["p1"] - p["p0"]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def cluster_angles(prims: list[dict], angle_tol: float = 1.2) -> dict:
    """Group line directions into families and snap each to its shared angle.

    Edges that repeat a direction to within a degree are a design grid, not a
    coincidence. Families within `angle_tol` of an axis snap to it exactly.
    Returns {index: snapped angle in degrees}.
    """
    lines = [(i, p) for i, p in enumerate(prims) if p["kind"] == "line"]
    if not lines:
        return {}
    entries = []
    for i, p in lines:
        d = _line_dir(p)
        ang = np.degrees(np.arctan2(d[1], d[0])) % 180.0
        entries.append([i, ang, float(np.linalg.norm(p["p1"] - p["p0"]))])
    entries.sort(key=lambda e: e[1])
    families, cur = [], [entries[0]]
    for e in entries[1:]:
        if e[1] - cur[-1][1] <= angle_tol:
            cur.append(e)
        else:
            families.append(cur)
            cur = [e]
    families.append(cur)
    # the 0/180 wrap is one family, not two
    if len(families) > 1 and (families[0][0][1] + 180.0 - families[-1][-1][1]) <= angle_tol:
        families[0] = families[-1] + [[i, a - 180.0, w] for i, a, w in families[0]]
        families.pop()
    out = {}
    for fam in families:
        w = np.array([e[2] for e in fam])
        mean = float(np.sum(np.array([e[1] for e in fam]) * w) / w.sum())
        for axis in (0.0, 90.0, 180.0, -90.0):
            if abs(mean - axis) <= angle_tol:
                mean = axis
                break
        for i, _, _ in fam:
            out[i] = mean % 180.0
    return out


def cluster_radii(prims: list[dict], radius_tol: float = 0.75) -> dict:
    """Group arc radii that agree closely and set each group to its mean."""
    arcs = [(i, p["r"]) for i, p in enumerate(prims) if p["kind"] == "arc"]
    if not arcs:
        return {}
    arcs.sort(key=lambda e: e[1])
    groups, cur = [], [arcs[0]]
    for e in arcs[1:]:
        if e[1] - cur[-1][1] <= radius_tol:
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)
    out = {}
    for g in groups:
        mean = float(np.mean([r for _, r in g]))
        for i, _ in g:
            out[i] = mean
    return out


def _intersect(a0, da, b0, db):
    m = np.array([da, -db]).T
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t, _ = np.linalg.solve(m, b0 - a0)
    return a0 + t * da


def _prim_points(p: dict, n: int = 160) -> np.ndarray:
    """Dense samples of one primitive, for scoring it against the contour."""
    if p["kind"] == "line":
        return np.linspace(p["p0"], p["p1"], n)
    if p["kind"] == "cubic":
        from .curves import _bezier
        return _bezier(np.array([p["p0"], p["c1"], p["c2"], p["p1"]]),
                       np.linspace(0, 1, n))
    c, r = p["c"], p["r"]
    a0 = np.arctan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
    sweep = p.get("sweep_angle", 0.0)
    t = np.linspace(a0, a0 + sweep, n)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)


def _deviation(prim: dict, contour: np.ndarray) -> float:
    d = _prim_points(prim)
    return float(np.min(np.linalg.norm(d[:, None] - contour[None], axis=2),
                        axis=1).max())


def snap_outline(prims: list[dict], contour: np.ndarray | None = None,
                 angle_tol: float = 1.2, radius_tol: float = 0.75,
                 allow: float = 0.35) -> tuple[list[dict], list[str]]:
    """Apply direction families, shared radii and line/arc tangency.

    Pass `contour` and every constraint is verified before it is kept: a
    change that pushes a primitive further from the measured boundary than
    `allow` is reverted. Constraints are a hypothesis about how the artwork
    was drawn, and a wrong one — two nearly parallel edges "filleted" into a
    corner that was never there — moves geometry by tens of pixels. Accepting
    only the ones that fit better is what makes snapping safe to run blind.

    Returns the rebuilt primitives and a list of what was changed or rejected.
    """
    prims = [dict(p) for p in prims]
    notes: list[str] = []
    angles = cluster_angles(prims, angle_tol)
    radii = cluster_radii(prims, radius_tol)

    # anchor each line by its midpoint and give it the family direction
    for i, ang in angles.items():
        p = prims[i]
        before = np.degrees(np.arctan2(*_line_dir(p)[::-1])) % 180.0
        mid = (p["p0"] + p["p1"]) / 2.0
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        p["anchor"], p["dir"] = mid, d
        if abs(before - ang) > 0.02:
            notes.append(f"line {i}: {before:.2f} -> {ang:.2f} deg")
    for i, r in radii.items():
        if abs(prims[i]["r"] - r) > 0.02:
            notes.append(f"arc {i}: r {prims[i]['r']:.3f} -> {r:.3f}")
        prims[i]["r"] = r

    n = len(prims)
    for i, p in enumerate(prims):
        if p["kind"] != "arc":
            continue
        prev, nxt = prims[(i - 1) % n], prims[(i + 1) % n]
        if prev["kind"] != "line" or nxt["kind"] != "line":
            continue
        # a fillet between two lines is fully determined by them and r:
        # its centre is where the two inward offset lines meet
        a0, da = prev["anchor"], prev["dir"]
        b0, db = nxt["anchor"], nxt["dir"]
        best = None
        for sa in (1, -1):
            for sb in (1, -1):
                na = np.array([-da[1], da[0]]) * sa * p["r"]
                nb = np.array([-db[1], db[0]]) * sb * p["r"]
                c = _intersect(a0 + na, da, b0 + nb, db)
                if c is None:
                    continue
                d = np.linalg.norm(c - p["c"])
                if best is None or d < best[0]:
                    best = (d, c)
        if best is None or best[0] > 3.0 * p["r"] + 5.0:
            continue
        c = best[1]
        keep = dict(p)
        p["c"] = c
        # tangent points are the feet of the perpendiculars from the centre
        p["p0"] = a0 + da * float((c - a0) @ da)
        p["p1"] = b0 + db * float((c - b0) @ db)
        ang0 = np.arctan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
        ang1 = np.arctan2(p["p1"][1] - c[1], p["p1"][0] - c[0])
        delta = (ang1 - ang0 + np.pi) % (2 * np.pi) - np.pi
        if keep.get("sweep_angle", delta) * delta < 0:
            delta += 2 * np.pi * (1 if delta < 0 else -1)
        p["sweep_angle"] = float(delta)
        if contour is not None:
            before, after = _deviation(keep, contour), _deviation(p, contour)
            if after > max(before, 0.0) + allow:
                prims[i] = keep
                notes.append(f"arc {i}: tangency REJECTED "
                             f"({before:.2f} -> {after:.2f}px)")
                continue
        prev["p1"], nxt["p0"] = p["p0"], p["p1"]
        notes.append(f"arc {i}: tangency enforced against lines "
                     f"{(i - 1) % n} and {(i + 1) % n}")

    # line/line joins become true intersections
    for i, p in enumerate(prims):
        if p["kind"] != "line":
            continue
        nxt = prims[(i + 1) % n]
        if nxt["kind"] != "line" or "dir" not in p or "dir" not in nxt:
            continue
        v = _intersect(p["anchor"], p["dir"], nxt["anchor"], nxt["dir"])
        # Two nearly parallel edges intersect a long way off. Moving the join
        # there turns a soft corner into a spike, so cap how far a vertex may
        # travel by the length of the shorter edge — a real corner's
        # intersection sits close to the traced one.
        reach = 0.5 * min(np.linalg.norm(p["p1"] - p["p0"]),
                          np.linalg.norm(nxt["p1"] - nxt["p0"]))
        if v is None or np.linalg.norm(v - p["p1"]) >= min(40.0, max(reach, 2.0)):
            continue
        keep_a, keep_b = p["p1"].copy(), nxt["p0"].copy()
        p["p1"], nxt["p0"] = v, v
        if contour is not None:
            if max(_deviation(p, contour), _deviation(nxt, contour)) > allow * 2:
                p["p1"], nxt["p0"] = keep_a, keep_b
                notes.append(f"join {i}/{(i + 1) % n}: intersection REJECTED")
    return prims, notes


def to_segments(prims: list[dict], close: bool = True) -> list[list]:
    """Convert primitives into model path segments."""
    if not prims:
        return []
    segs: list[list] = [["M", float(prims[0]["p0"][0]), float(prims[0]["p0"][1])]]
    for p in prims:
        if p["kind"] == "line":
            segs.append(["L", float(p["p1"][0]), float(p["p1"][1])])
        elif p["kind"] == "cubic":
            segs.append(["C", float(p["c1"][0]), float(p["c1"][1]),
                         float(p["c2"][0]), float(p["c2"][1]),
                         float(p["p1"][0]), float(p["p1"][1])])
        else:
            sweep_angle = p.get("sweep_angle")
            if sweep_angle is None:
                a0 = np.arctan2(p["p0"][1] - p["c"][1], p["p0"][0] - p["c"][0])
                a1 = np.arctan2(p["p1"][1] - p["c"][1], p["p1"][0] - p["c"][0])
                sweep_angle = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            large = 1 if abs(sweep_angle) > np.pi else 0
            sweep = 1 if sweep_angle > 0 else 0
            segs.append(["A", float(p["r"]), float(p["r"]), 0, large, sweep,
                         float(p["p1"][0]), float(p["p1"][1])])
    if close:
        segs.append(["Z"])
    return segs
