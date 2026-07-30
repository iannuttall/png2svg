"""Geometry helpers for building path segment lists."""

from __future__ import annotations

import math


def rounded_polygon(
    points: list[tuple[float, float]], radius: float | list[float]
) -> list[list]:
    """Segment list for a polygon with circular corner rounding.

    Points are the sharp vertices in drawing order; winding determines arc
    sweep automatically. radius may be one value or one per vertex.
    """
    n = len(points)
    radii = [radius] * n if isinstance(radius, (int, float)) else list(radius)
    if n < 3:
        raise ValueError("polygon needs at least 3 points")
    if len(radii) != n:
        raise ValueError(f"expected one radius or {n} corner radii")
    if any(float(r) < 0 for r in radii):
        raise ValueError("radii must be >= 0")
    segs: list[list] = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        v = points[i]
        p_next = points[(i + 1) % n]
        r = radii[i]
        u_in = _norm((v[0] - p_prev[0], v[1] - p_prev[1]))
        u_out = _norm((p_next[0] - v[0], p_next[1] - v[1]))
        cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
        dot = max(-1.0, min(1.0, u_in[0] * u_out[0] + u_in[1] * u_out[1]))
        turn = math.acos(dot)
        if r <= 0 or abs(cross) < 1e-9:
            segs.append(["L", v[0], v[1]])
            continue
        t = r * math.tan(turn / 2)
        a = (v[0] - u_in[0] * t, v[1] - u_in[1] * t)
        b = (v[0] + u_out[0] * t, v[1] + u_out[1] * t)
        sweep = 1 if cross > 0 else 0
        segs.append(["L", a[0], a[1]])
        segs.append(["A", r, r, 0, 0, sweep, b[0], b[1]])
    # every vertex emits an L first, so the path always opens with one
    segs[0][0] = "M"
    segs.append(["Z"])
    return segs


def _norm(v: tuple[float, float]) -> tuple[float, float]:
    d = math.hypot(*v)
    if d < 1e-12:
        raise ValueError("polygon has a zero-length edge")
    return (v[0] / d, v[1] / d)


def smooth_polygon(
    points: list[tuple[float, float]], corners: list[dict]
) -> list[list]:
    """Polygon with smoothed (squircle-style) corners as single cubics.

    corners[i] applies at points[i]: {"t_in": float, "t_out": float,
    "h_in": float, "h_out": float}. t_* are tangent-point distances from the
    vertex along the incoming/outgoing edges; h_* are cubic handle lengths
    along the edge directions (h=0 gives a chamfer, larger h a rounder turn).
    """
    n = len(points)
    segs: list[list] = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        v = points[i]
        p_next = points[(i + 1) % n]
        c = corners[i]
        u_in = _norm((v[0] - p_prev[0], v[1] - p_prev[1]))
        u_out = _norm((p_next[0] - v[0], p_next[1] - v[1]))
        p0 = (v[0] - u_in[0] * c["t_in"], v[1] - u_in[1] * c["t_in"])
        p3 = (v[0] + u_out[0] * c["t_out"], v[1] + u_out[1] * c["t_out"])
        p1 = (p0[0] + u_in[0] * c["h_in"], p0[1] + u_in[1] * c["h_in"])
        p2 = (p3[0] - u_out[0] * c["h_out"], p3[1] - u_out[1] * c["h_out"])
        segs.append(["L", p0[0], p0[1]])
        segs.append(["C", p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]])
    segs[0][0] = "M"
    segs.append(["Z"])
    return segs


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    if abs(a) < 1e-14:
        return [] if abs(b) < 1e-14 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    root = math.sqrt(max(0.0, disc))
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def _on_arc(angle: float, start: float, delta: float) -> bool:
    if delta >= 0:
        travelled = (angle - start) % (2.0 * math.pi)
        return travelled <= delta + 1e-12
    travelled = (start - angle) % (2.0 * math.pi)
    return travelled <= -delta + 1e-12


def _arc_parameters(p0: tuple[float, float], seg: list):
    """SVG endpoint arc converted to centre parameterisation."""
    rx, ry, rotation, large_arc, sweep, x1, y1 = seg[1:]
    rx, ry = abs(float(rx)), abs(float(ry))
    p1 = (float(x1), float(y1))
    if rx < 1e-14 or ry < 1e-14 or (
        abs(p0[0] - p1[0]) < 1e-14 and abs(p0[1] - p1[1]) < 1e-14
    ):
        return None
    phi = math.radians(float(rotation) % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    xp = cp * dx + sp * dy
    yp = -sp * dx + cp * dy
    scale = xp * xp / (rx * rx) + yp * yp / (ry * ry)
    if scale > 1.0:
        factor = math.sqrt(scale)
        rx *= factor
        ry *= factor
    numerator = max(
        0.0,
        rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp,
    )
    denominator = rx * rx * yp * yp + ry * ry * xp * xp
    coefficient = 0.0 if denominator < 1e-28 else math.sqrt(numerator / denominator)
    if bool(large_arc) == bool(sweep):
        coefficient = -coefficient
    cxp = coefficient * rx * yp / ry
    cyp = -coefficient * ry * xp / rx
    cx = cp * cxp - sp * cyp + (p0[0] + p1[0]) / 2.0
    cy = sp * cxp + cp * cyp + (p0[1] + p1[1]) / 2.0

    ux, uy = (xp - cxp) / rx, (yp - cyp) / ry
    vx, vy = (-xp - cxp) / rx, (-yp - cyp) / ry
    start = math.atan2(uy, ux)
    delta = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if not sweep and delta > 0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0:
        delta += 2.0 * math.pi
    return cx, cy, rx, ry, phi, start, delta


def path_bounds(segments: list[list], stroke_width: float = 0.0) -> tuple[float, ...]:
    """Exact bounds of absolute M/L/Q/C/A/Z path segments.

    Bezier derivative roots and elliptical-arc cardinal extrema are solved
    analytically. `stroke_width` expands the result by half the width; callers
    with sharp miter joins should add their own larger safety margin.
    """
    if not segments:
        raise ValueError("path is empty")
    points: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None

    def add(point):
        points.append((float(point[0]), float(point[1])))

    for seg in segments:
        cmd = seg[0]
        if cmd == "M":
            current = (float(seg[1]), float(seg[2]))
            start = current
            add(current)
            continue
        if current is None:
            raise ValueError("path must start with M")
        if cmd == "L":
            current = (float(seg[1]), float(seg[2]))
            add(current)
        elif cmd == "Q":
            p0 = current
            p1 = (float(seg[1]), float(seg[2]))
            p2 = (float(seg[3]), float(seg[4]))
            add(p2)
            for axis in (0, 1):
                denominator = p0[axis] - 2.0 * p1[axis] + p2[axis]
                if abs(denominator) > 1e-14:
                    t = (p0[axis] - p1[axis]) / denominator
                    if 0.0 < t < 1.0:
                        u = 1.0 - t
                        add((
                            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
                        ))
            current = p2
        elif cmd == "C":
            p0 = current
            p1 = (float(seg[1]), float(seg[2]))
            p2 = (float(seg[3]), float(seg[4]))
            p3 = (float(seg[5]), float(seg[6]))
            add(p3)
            roots: set[float] = set()
            for axis in (0, 1):
                a = -p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]
                b = 2 * (p0[axis] - 2 * p1[axis] + p2[axis])
                c = p1[axis] - p0[axis]
                roots.update(_quadratic_roots(a, b, c))
            for t in roots:
                if 0.0 < t < 1.0:
                    u = 1.0 - t
                    add((
                        u**3 * p0[0] + 3 * u * u * t * p1[0]
                        + 3 * u * t * t * p2[0] + t**3 * p3[0],
                        u**3 * p0[1] + 3 * u * u * t * p1[1]
                        + 3 * u * t * t * p2[1] + t**3 * p3[1],
                    ))
            current = p3
        elif cmd == "A":
            endpoint = (float(seg[-2]), float(seg[-1]))
            params = _arc_parameters(current, seg)
            add(endpoint)
            if params is not None:
                cx, cy, rx, ry, phi, angle0, delta = params
                cp, sp = math.cos(phi), math.sin(phi)
                candidates = (
                    math.atan2(-ry * sp, rx * cp),
                    math.atan2(-ry * sp, rx * cp) + math.pi,
                    math.atan2(ry * cp, rx * sp),
                    math.atan2(ry * cp, rx * sp) + math.pi,
                )
                for angle in candidates:
                    if _on_arc(angle, angle0, delta):
                        add((
                            cx + rx * cp * math.cos(angle) - ry * sp * math.sin(angle),
                            cy + rx * sp * math.cos(angle) + ry * cp * math.sin(angle),
                        ))
            current = endpoint
        elif cmd == "Z":
            if start is not None:
                current = start
                add(start)
        else:
            raise ValueError(f"unsupported path command {cmd!r}")

    pad = max(0.0, float(stroke_width)) / 2.0
    xs, ys = zip(*points)
    return (
        min(xs) - pad,
        min(ys) - pad,
        max(xs) + pad,
        max(ys) + pad,
    )
