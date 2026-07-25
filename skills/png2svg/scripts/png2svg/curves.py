"""Fit a minimal chain of Béziers to an ordered run of boundary points.

This is the general form of the one-off curve fits that otherwise get
hand-written per image: give it a subpixel contour and it returns the
fewest line and cubic segments that stay within tolerance, with matched
tangents across joins so the outline stays smooth where the artwork is
smooth and breaks only where there is a real corner.

The algorithm is Schneider's: parameterise by chord length, solve for the
two handle lengths in least squares, improve the parameterisation with a
Newton step, and split at the worst point when the error is still too big.
"""

from __future__ import annotations

import numpy as np


def _bezier(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    p0, p1, p2, p3 = ctrl
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 0 else u


def _fit_one(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    """Least-squares cubic through pts[0]/pts[-1] with the given end tangents."""
    p0, p3 = pts[0], pts[-1]
    a1 = 3 * (1 - u) ** 2 * u
    a2 = 3 * (1 - u) * u ** 2
    # p1 = p0 + alpha1*t1 and p2 = p3 + alpha2*t2, so the p0/p3 contributions
    # include the middle basis terms too — dropping them makes every fit wrong
    base = ((1 - u) ** 3 + a1)[:, None] * p0 + (a2 + u ** 3)[:, None] * p3
    rhs = pts - base
    # normal equations for the two scalar handle lengths
    c11 = float(np.sum(a1 * a1) * (t1 @ t1))
    c12 = float(np.sum(a1 * a2) * (t1 @ t2))
    c22 = float(np.sum(a2 * a2) * (t2 @ t2))
    x1 = float(np.sum(a1[:, None] * rhs @ t1))
    x2 = float(np.sum(a2[:, None] * rhs @ t2))
    det = c11 * c22 - c12 * c12
    if abs(det) < 1e-12:
        seg = np.linalg.norm(p3 - p0) / 3.0
        alpha1 = alpha2 = seg
    else:
        alpha1 = (x1 * c22 - c12 * x2) / det
        alpha2 = (c11 * x2 - x1 * c12) / det
    span = np.linalg.norm(p3 - p0)
    if alpha1 < 1e-6 or alpha2 < 1e-6:
        alpha1 = alpha2 = span / 3.0
    # runaway handles fit noise, not shape
    alpha1 = min(alpha1, span * 1.5)
    alpha2 = min(alpha2, span * 1.5)
    return np.array([p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3])


def _max_error(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray):
    """Largest deviation, measured at the fitted parameters.

    Comparing against a fixed dense sampling instead would put a resolution
    floor under the result — half the sample spacing — and any tolerance
    below that floor would split forever.
    """
    err = np.linalg.norm(_bezier(ctrl, u) - pts, axis=1)
    return float(err.max()), int(err.argmax())


def _reparameterise(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = ctrl
    d1 = 3 * (p1 - p0), 3 * (p2 - p1), 3 * (p3 - p2)
    out = u.copy()
    for i, t in enumerate(u):
        b = _bezier(ctrl, np.array([t]))[0]
        q1 = ((1 - t) ** 2 * d1[0] + 2 * (1 - t) * t * d1[1] + t ** 2 * d1[2])
        q2 = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
        num = (b - pts[i]) @ q1
        den = q1 @ q1 + (b - pts[i]) @ q2
        if abs(den) > 1e-12:
            out[i] = np.clip(t - num / den, 0.0, 1.0)
    return np.sort(out)


def _tangent(pts: np.ndarray, i: int, span: int, forward: bool) -> np.ndarray:
    """Tangent at pts[i], pointing forward or backward along the run.

    A plain secant is biased by half the curve's turning over the span, which
    is around a degree on a typical arc — small, but it is exactly the error
    the handle solve cannot compensate for, so it puts a floor under every
    fit. Fitting a local quadratic in chord length and taking its derivative
    at the endpoint removes that bias.
    """
    n = len(pts)
    if forward:
        seg = pts[i:min(i + span + 1, n)]
    else:
        seg = pts[max(i - span, 0):i + 1][::-1]
    if len(seg) < 3:
        v = seg[-1] - seg[0] if len(seg) > 1 else np.array([1.0, 0.0])
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.array([1.0, 0.0])
    h = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    if h[-1] < 1e-9:
        return np.array([1.0, 0.0])
    A = np.stack([h, h ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, seg - seg[0], rcond=None)
    v = coef[0]
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        v = seg[-1] - seg[0]
        norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.array([1.0, 0.0])


def fit_run(pts: np.ndarray, tol: float, t1=None, t2=None, depth: int = 0) -> list:
    """Fit one run of points, splitting recursively until within tolerance."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return [("L", pts[-1])]
    if t1 is None:
        t1 = _tangent(pts, 0, min(5, len(pts) - 1), True)
    if t2 is None:
        t2 = _tangent(pts, len(pts) - 1, min(5, len(pts) - 1), False)
    # a straight run needs no curve at all
    chord = pts[-1] - pts[0]
    L = np.linalg.norm(chord)
    if L > 1e-9:
        nrm = np.array([-chord[1], chord[0]]) / L
        if np.abs((pts - pts[0]) @ nrm).max() <= tol:
            return [("L", pts[-1])]
    u = _chord_params(pts)
    ctrl = _fit_one(pts, u, t1, t2)
    err, idx = _max_error(pts, u, ctrl)
    # Newton reparameterisation converges quickly but not in three steps;
    # stopping early splits curves that a single cubic would have held
    for _ in range(16):
        if err <= tol:
            break
        u2 = _reparameterise(pts, u, ctrl)
        ctrl2 = _fit_one(pts, u2, t1, t2)
        err2, idx2 = _max_error(pts, u2, ctrl2)
        if err2 >= err * 0.999:      # no longer improving
            break
        u, ctrl, err, idx = u2, ctrl2, err2, idx2
    if err <= tol or depth >= 12:
        return [("C", ctrl[1], ctrl[2], ctrl[3])]
    idx = int(np.clip(idx, 2, len(pts) - 3))
    centre = _tangent(pts, idx, 3, True)
    left = fit_run(pts[:idx + 1], tol, t1, -centre, depth + 1)
    right = fit_run(pts[idx:], tol, centre, t2, depth + 1)
    return left + right


def corner_indices(pts: np.ndarray, angle_deg: float = 35.0, span: int = 6) -> list[int]:
    """Indices where the tangent turns sharply enough to be a real corner."""
    n = len(pts)
    out = []
    for i in range(n):
        a = _tangent(pts, i, span, False)
        b = _tangent(pts, i, span, True)
        turn = np.degrees(np.arccos(np.clip(-a @ b, -1, 1)))
        if turn > angle_deg:
            out.append((i, turn))
    # keep the sharpest index in each cluster of adjacent candidates
    picked, group = [], []
    for i, turn in out:
        if group and i - group[-1][0] > span:
            picked.append(max(group, key=lambda z: z[1]))
            group = []
        group.append((i, turn))
    if group:
        picked.append(max(group, key=lambda z: z[1]))
    # the contour is closed, so a cluster straddling the seam appears twice:
    # once near index 0 and once near index n-1. Keep the sharper.
    if len(picked) > 1 and (picked[0][0] + n - picked[-1][0]) <= span:
        drop = picked[0] if picked[0][1] < picked[-1][1] else picked[-1]
        picked = [p for p in picked if p is not drop]
    return [i for i, _ in picked]


def fit_bezier_chain(pts, tol: float = 0.3, corner_deg: float = 35.0,
                     closed: bool = True) -> list:
    """Fit an ordered contour with the fewest line/cubic segments.

    Returns segments as ("L", p) or ("C", c1, c2, p); the caller supplies the
    opening move to pts[0]. Corners are detected first so that deliberate
    breaks are preserved rather than smoothed over.
    """
    pts = np.asarray(pts, float)
    cuts = corner_indices(pts, corner_deg)
    if not cuts:
        cuts = [0]
    segs = []
    for k in range(len(cuts)):
        i0 = cuts[k]
        i1 = cuts[(k + 1) % len(cuts)]
        run = pts[i0:i1 + 1] if i1 > i0 else np.vstack([pts[i0:], pts[:i1 + 1]])
        if len(run) >= 2:
            segs.extend(fit_run(run, tol))
    if not closed:
        segs = fit_run(pts, tol)
    return segs
