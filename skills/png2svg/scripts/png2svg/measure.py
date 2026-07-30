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
        self.height, self.width = self.dist.shape
        ring = np.concatenate(
            (
                self.dist[:2].ravel(),
                self.dist[-2:].ravel(),
                self.dist[:, :2].ravel(),
                self.dist[:, -2:].ravel(),
            )
        )
        median = float(np.median(ring))
        mad = float(np.median(np.abs(ring - median)))
        # A border-derived background can differ slightly from the page
        # interior after compression or a faint page gradient. Admit the
        # image's lower quartile only when it stays close to the border; a
        # low-contrast foreground farther away must remain foreground.
        page_low = float(np.percentile(self.dist, 25))
        page_background = page_low if page_low <= median + 12.0 else median
        self.background_level = page_background
        self.background_limit = max(
            median + max(10.0, 5.0 * 1.4826 * mad),
            page_background + 5.0,
        )

    def at(self, x: float, y: float) -> float:
        if not self.contains(x, y):
            raise ValueError(f"sample ({x:g}, {y:g}) is outside the image")
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        d = self.dist
        return float(
            d[y0, x0] * (1 - fx) * (1 - fy)
            + d[y0, x0 + 1] * fx * (1 - fy)
            + d[y0 + 1, x0] * (1 - fx) * fy
            + d[y0 + 1, x0 + 1] * fx * fy
        )

    def contains(self, x: float, y: float) -> bool:
        """Whether bilinear interpolation is defined at `(x, y)`."""
        return 0.0 <= x < self.width - 1 and 0.0 <= y < self.height - 1


def _background_offset(
    field: Field,
    origin: np.ndarray,
    outward: np.ndarray,
    maximum: float,
    *,
    minimum: float = 1.0,
    step: float = 0.5,
) -> float | None:
    """Farthest background point before another foreground shape.

    A fixed scan offset is unsafe when two components are close: it can put
    the ray inside the neighbour, so the first crossing belongs to the wrong
    shape. Starting near the boundary and walking outward finds the connected
    background corridor instead. The fixed sampling grid makes the result
    deterministic.
    """
    found = None
    entered_background = False
    first_excess = None
    distances = np.arange(minimum, maximum + step * 0.25, step)
    for distance in distances:
        point = origin + outward * distance
        if not field.contains(float(point[0]), float(point[1])):
            break
        value = field.at(float(point[0]), float(point[1]))
        is_background = value <= field.background_limit
        if is_background:
            entered_background = True
            found = float(distance)
        elif entered_background:
            # This is the next component (or the image border), not more room
            # in which to start the current component's scan.
            break
        else:
            excess = max(0.0, value - field.background_level)
            if first_excess is None:
                first_excess = excess
            if distance >= 2.5 and excess > max(2.0, first_excess * 0.6):
                # On the correct side a blurred edge falls quickly toward
                # background. A foreground plateau means this is the inside.
                return None
        if not entered_background and distance >= 4.5:
            # A traced edge is never this far from its connected background.
            return None
    return found


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
    baseline = float(np.median(vals[: max(2, int(0.8 / step))]))
    minimum_contrast = max(
        3.0, field.background_limit - field.background_level
    )
    threshold = baseline + minimum_contrast

    # Find connected foreground runs instead of sampling a fixed window
    # 2.5-8px inside. The fixed window fails on thin strokes because it has
    # already crossed out the other side. Nearby weak runs are grouped with
    # a stronger plateau so resampling overshoot cannot steal the crossing.
    # An isolated weak run is retained for genuine low-contrast thin strokes.
    cursor = 0
    background_needed = max(3, int(0.5 / step))
    weak_candidate: tuple[int, float, int] | None = None
    while True:
        candidates = np.where(vals[cursor:] > threshold)[0]
        if len(candidates) == 0:
            if weak_candidate is None:
                return None
            i0, steady, _ = weak_candidate
            break
        i0 = cursor + int(candidates[0])
        search_end = min(i0 + int(10.0 / step), len(vals))
        end = search_end
        next_cursor = i0 + 1
        background_run = 0
        for j in range(i0, search_end):
            if vals[j] <= threshold:
                background_run += 1
                if background_run >= background_needed:
                    end = j - background_run + 1
                    next_cursor = j + 1
                    break
            else:
                background_run = 0
        run = vals[i0:end]
        if len(run) >= 3:
            steady = float(np.percentile(run, 90))
            strength = steady - baseline
            if strength >= minimum_contrast:
                if weak_candidate is not None:
                    _, _, weak_end = weak_candidate
                    if (i0 - weak_end) * step > 4.0:
                        i0, steady, _ = weak_candidate
                        break
                if strength >= minimum_contrast * 3.0:
                    break
                if weak_candidate is None:
                    weak_candidate = (i0, steady, end)
                else:
                    weak_candidate = (
                        weak_candidate[0],
                        weak_candidate[1],
                        end,
                    )
        if next_cursor >= len(vals):
            if weak_candidate is None:
                return None
            i0, steady, _ = weak_candidate
            break
        if weak_candidate is not None and (next_cursor - weak_candidate[2]) * step > 4.0:
            i0, steady, _ = weak_candidate
            break
        cursor = next_cursor

    half = baseline + (steady - baseline) / 2.0
    peak_index = i0 + int(np.argmax(run))
    for i in range(max(i0 - int(3.0 / step), 0), peak_index):
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
    `offset` is the maximum search distance. The actual start stays in the
    connected background corridor, so a nearby component cannot steal the
    scan. Endpoints are skipped (t0/t1) because corners bend the boundary.
    Returns an (n, 2) array, which may be shorter than `count` where a ray
    found no clean transition.
    """
    a, b = np.array(p0, float), np.array(p1, float)
    v = b - a
    length = float(np.linalg.norm(v))
    d = v / length
    nrm = np.array([-d[1], d[0]])
    mid = a + v / 2
    plus = _background_offset(field, mid, nrm, offset)
    minus = _background_offset(field, mid, -nrm, offset)
    if plus is None and minus is None:
        return np.empty((0, 2), dtype=np.float64)
    side = 1.0 if (plus or 0.0) >= (minus or 0.0) else -1.0
    pts = []
    for t in np.linspace(t0, t1, count):
        edge = a + d * (t * length)
        actual = _background_offset(field, edge, nrm * side, offset)
        if actual is None:
            continue
        start = edge + nrm * (side * actual)
        hit = edge_point(field, tuple(start), tuple(-nrm * side), actual + 10.0)
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
    the background side inward. `offset` is a maximum: each scan starts at
    the farthest point in the connected background corridor before another
    component or the image edge.

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
        actual = _background_offset(field, raw[i], nrm, offset)
        hit = None
        if actual is not None:
            start = raw[i] + nrm * actual
            hit = edge_point(field, tuple(start), tuple(-nrm), actual + 10.0)
            # A traced pixel is at most about one pixel from its real edge.
            # A farther result belongs to unrelated geometry along a bad
            # corner normal, so keep the traced point instead.
            raw_svg = raw[i] + INDEX_TO_SVG
            if hit is not None and np.linalg.norm(np.asarray(hit) - raw_svg) > 2.0:
                hit = None
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
