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
