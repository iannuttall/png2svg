"""Subpixel measurement utilities for reconstruction analysis.

Coordinate convention: pixel index j covers SVG-space [j, j+1], so a
0.5-coverage crossing at interpolated index m corresponds to SVG m + 0.5.
Functions here return SVG-space coordinates directly.

Coverage is normalised by the full-strength foreground plateau found along
each scan ray, so dark and light shapes measure identically (a fixed
colour-distance threshold would bias dark edges outward by ~0.4 px).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

INDEX_TO_SVG = 0.5


class Field:
    """Bilinear-interpolated background-distance field over an RGB image."""

    def __init__(self, rgb: np.ndarray, background: tuple[float, float, float]):
        self.dist = np.linalg.norm(
            rgb.astype(np.float64) - np.array(background, dtype=np.float64), axis=-1
        )

    def at(self, x: float, y: float) -> float:
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        d = self.dist
        return float(
            d[y0, x0] * (1 - fx) * (1 - fy)
            + d[y0, x0 + 1] * fx * (1 - fy)
            + d[y0 + 1, x0] * (1 - fx) * fy
            + d[y0 + 1, x0 + 1] * fx * fy
        )


def edge_cross(
    field: Field,
    p0: tuple[float, float],
    direction: tuple[float, float],
    t_max: float,
    step: float = 0.1,
) -> float | None:
    """Distance t along the ray p0 + t*direction at which coverage crosses 50%.

    The ray must start in background and run into the foreground. Coverage is
    normalised by the plateau value measured past the transition. Returns t in
    index space; add INDEX_TO_SVG projected on the ray for SVG space (the
    caller helpers below handle this).
    """
    ts = np.arange(0.0, t_max, step)
    vals = np.array([field.at(p0[0] + t * direction[0], p0[1] + t * direction[1]) for t in ts])
    above = np.where(vals > 60.0)[0]
    if len(above) == 0:
        return None
    i0 = above[0]
    plateau_lo = min(i0 + int(2.5 / step), len(vals) - 1)
    plateau_hi = min(i0 + int(8.0 / step), len(vals))
    if plateau_hi - plateau_lo < 3:
        return None
    steady = float(np.median(vals[plateau_lo:plateau_hi]))
    half = steady / 2.0
    for i in range(max(i0 - int(3.0 / step), 0), plateau_lo):
        if vals[i] < half <= vals[i + 1]:
            return float(ts[i] + (half - vals[i]) / (vals[i + 1] - vals[i]) * step)
    return None


def edge_point(
    field: Field,
    p0: tuple[float, float],
    direction: tuple[float, float],
    t_max: float,
) -> tuple[float, float] | None:
    """SVG-space point where the ray crosses the 50%-coverage boundary."""
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    t = edge_cross(field, p0, tuple(d), t_max)
    if t is None:
        return None
    return (
        p0[0] + t * d[0] + INDEX_TO_SVG,
        p0[1] + t * d[1] + INDEX_TO_SVG,
    )


def edge_samples(
    field: Field,
    p0: tuple[float, float],
    p1: tuple[float, float],
    offset: float = 14.0,
    t0: float = 0.15,
    t1: float = 0.85,
    count: int = 28,
) -> np.ndarray:
    """Boundary samples along the nominal straight edge p0 -> p1.

    Rays are cast perpendicular to the edge from whichever side reads as
    background, so this works for outer silhouettes and counters alike.
    `offset` must land clear of the edge in that background; endpoints are
    skipped (t0/t1) because corners bend the boundary. Returns an (n, 2)
    array of SVG-space points, which may be shorter than `count` where a
    ray found no clean transition.
    """
    a, b = np.array(p0, float), np.array(p1, float)
    v = b - a
    length = float(np.linalg.norm(v))
    d = v / length
    nrm = np.array([-d[1], d[0]])
    mid = a + v / 2
    side = 1.0 if field.at(*(mid + nrm * offset)) < field.at(*(mid - nrm * offset)) else -1.0
    pts = []
    for t in np.linspace(t0, t1, count):
        start = a + d * (t * length) + nrm * (side * offset)
        hit = edge_point(field, tuple(start), tuple(-nrm * side), offset * 2.4)
        if hit is not None:
            pts.append(hit)
    return np.array(pts, dtype=np.float64)


def subpixel_contour(
    field: Field,
    mask: np.ndarray,
    offset: float = 9.0,
    smooth: int = 3,
) -> np.ndarray:
    """Refine a traced mask boundary to an ordered subpixel contour.

    `edge_samples` needs the endpoints of a straight edge; this is its
    free-form counterpart and the starting point for any curved work. The
    integer boundary is traced, a local tangent is estimated over +/-`smooth`
    points, and each point is re-found along its own normal by scanning from
    the background side inward.

    Where no clean transition is found — typically at a sharp concave corner,
    where the estimated normal points along the boundary rather than across
    it — the raw traced point is kept instead. Dropping those would tear a
    hole in the contour and any curve fitted across the gap would be
    nonsense. Returns an (n, 2) array in SVG space.
    """
    from .analyse import trace_boundary

    raw = np.asarray(trace_boundary(mask), dtype=np.float64)
    n = len(raw)
    if n < 8:
        return raw
    # signed area gives the winding, and thus which normal points outward
    area = 0.5 * np.sum(raw[:, 0] * np.roll(raw[:, 1], -1)
                        - np.roll(raw[:, 0], -1) * raw[:, 1])
    sign = 1.0 if area > 0 else -1.0
    out = []
    for i in range(n):
        a = raw[(i - smooth) % n]
        b = raw[(i + smooth) % n]
        t = b - a
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            continue
        t /= norm
        nrm = np.array([t[1], -t[0]]) * sign      # outward
        start = raw[i] + nrm * offset
        hit = edge_point(field, tuple(start), tuple(-nrm), offset * 2.4)
        if hit is None:
            hit = (raw[i][0] + INDEX_TO_SVG, raw[i][1] + INDEX_TO_SVG)
        out.append(hit)
    return np.array(out, dtype=np.float64)


def fit_line(points) -> tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line fit (no steep/shallow bias).

    Returns (point_on_line, unit_direction, max_abs_residual). Unlike
    `fit_line_x_of_y` this handles every orientation, so it is the right
    default for polygon edges of unknown slope.
    """
    pts = np.asarray(points, dtype=np.float64)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    return centre, d, float(np.abs((pts - centre) @ nrm).max())


def intersect(line_a, line_b) -> np.ndarray:
    """Intersection of two (point, direction) lines. Raises if parallel."""
    (c1, d1), (c2, d2) = line_a, line_b
    t, _ = np.linalg.solve(np.array([d1, -d2], dtype=np.float64).T,
                           np.asarray(c2, float) - np.asarray(c1, float))
    return np.asarray(c1, float) + t * np.asarray(d1, float)


def fit_circle(points: list[tuple[float, float]]):
    """Least-squares circle fit; returns ((cx, cy, r), max_abs_residual)."""
    pts = np.asarray(points, dtype=np.float64)

    def res(p):
        return np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2]

    p0 = [pts[:, 0].mean(), pts[:, 1].mean(), np.std(pts, axis=0).sum()]
    out = optimize.least_squares(res, p0)
    return tuple(out.x), float(np.abs(res(out.x)).max())


def fit_line_x_of_y(points: list[tuple[float, float]]):
    """Fit x = m*y + c; returns (m, c, max_abs_residual)."""
    pts = np.asarray(points, dtype=np.float64)
    A = np.vstack([pts[:, 1], np.ones(len(pts))]).T
    (m, c), *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    return float(m), float(c), float(np.abs(A @ [m, c] - pts[:, 0]).max())


def fit_corner_cubic(
    p0: tuple[float, float],
    u_in: tuple[float, float],
    p3: tuple[float, float],
    u_out: tuple[float, float],
    samples: list[tuple[float, float]],
):
    """Fit handle lengths (h_in, h_out) of a corner cubic to boundary samples.

    p0/p3 are the tangent endpoints; u_in is the travel direction into the
    corner, u_out the travel direction out. Returns ((h_in, h_out), max_err).
    """
    p0a, p3a = np.array(p0, float), np.array(p3, float)
    u0, u3 = np.array(u_in, float), np.array(u_out, float)
    u0 /= np.linalg.norm(u0)
    u3 /= np.linalg.norm(u3)
    S = np.asarray(samples, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, 200)[:, None]

    def loss(h):
        p1 = p0a + u0 * h[0]
        p2 = p3a - u3 * h[1]
        curve = (
            (1 - ts) ** 3 * p0a
            + 3 * (1 - ts) ** 2 * ts * p1
            + 3 * (1 - ts) * ts**2 * p2
            + ts**3 * p3a
        )
        return np.min(np.linalg.norm(curve[None, :, :] - S[:, None, :], axis=2), axis=1)

    out = optimize.least_squares(loss, [15.0, 15.0], bounds=(0.0, 100.0))
    return (float(out.x[0]), float(out.x[1])), float(np.abs(loss(out.x)).max())


def fit_corner_full(
    vertex: tuple[float, float],
    u_in: tuple[float, float],
    u_out: tuple[float, float],
    samples: list[tuple[float, float]],
    t0: float = 40.0,
):
    """Fit a corner cubic with free tangent lengths AND handle lengths.

    Returns ((t_in, t_out, h_in, h_out), max_err). p0 = vertex - u_in*t_in,
    p3 = vertex + u_out*t_out, handles along the edge directions.
    """
    v = np.array(vertex, float)
    u0 = np.array(u_in, float) / np.linalg.norm(u_in)
    u3 = np.array(u_out, float) / np.linalg.norm(u_out)
    S = np.asarray(samples, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, 240)[:, None]

    def loss(p):
        t_in, t_out, h_in, h_out = p
        p0 = v - u0 * t_in
        p3 = v + u3 * t_out
        p1 = p0 + u0 * h_in
        p2 = p3 - u3 * h_out
        curve = (
            (1 - ts) ** 3 * p0
            + 3 * (1 - ts) ** 2 * ts * p1
            + 3 * (1 - ts) * ts**2 * p2
            + ts**3 * p3
        )
        return np.min(np.linalg.norm(curve[None, :, :] - S[:, None, :], axis=2), axis=1)

    out = optimize.least_squares(
        loss, [t0, t0, t0 * 0.45, t0 * 0.45], bounds=(0.0, 120.0)
    )
    return tuple(float(x) for x in out.x), float(np.abs(loss(out.x)).max())
