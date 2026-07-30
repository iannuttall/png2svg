"""Fit a decomposition into overlapping rounded primitives, as one system.

The other fitting path in this library traces a boundary and fits segments to
it. That is the wrong shape of tool when the artwork is a handful of
overlapping filled primitives, because the union's boundary is an *output* of
the construction, not the thing the designer positioned. Tracing it spends
nodes on corners that are mere intersections and loses every constraint that
made the artwork regular.

Here you declare the decomposition as a function of a parameter vector and
solve for the parameters against every contour point at once:

    def build(p):
        ang, cx, cy, a, b, g, k, r = p
        ...
        return [(verts_of_rect_1, r), (verts_of_rect_2, r), ...]

    fit = fit_union(contour, build, p0)

The signed distance to the union is zero on the true boundary, so the
residual is a real distance in pixels and reads directly against the edge
targets in SKILL.md.

Why this is usually the better deal: symmetries collapse parameters. A mark
built from three rounded parallelograms with 180-degree symmetry has 8 free
numbers, and the offset between primitives and each one's width fall out of
the symmetry rather than being fitted. 8 numbers beat ~40 traced nodes, and
every one of them is a number a designer typed.

Convexity: `rounded_convex_sdf` requires a convex polygon. The radius may be
one value or one value per corner. Non-convex artwork stays a union of convex
pieces, which is how it was drawn anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import optimize


class PrimitiveError(ValueError):
    """A primitive's parameters do not describe a real shape."""


def _inward_normals(verts: np.ndarray) -> np.ndarray:
    """Unit inward normal per edge i (from verts[i] to verts[i+1])."""
    n = len(verts)
    e = np.roll(verts, -1, axis=0) - verts
    lengths = np.linalg.norm(e, axis=1)
    if np.any(lengths < 1e-12):
        raise PrimitiveError("polygon has a zero-length edge")
    d = e / lengths[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    # Signed area picks the winding, which picks which normal points inward.
    # In screen coordinates (y down) a clockwise polygon has POSITIVE signed
    # area, and (-dy, dx) already points inward for it. Getting this backwards
    # silently turns the SDF into an unsigned distance -- which still fits a
    # boundary perfectly and only shows up when you rasterise. Hence
    # `test_raster_and_ink_bounds_agree_with_the_geometry`.
    if _signed_area(verts) < 0:
        nrm = -nrm
    return nrm


def _signed_area(verts: np.ndarray) -> float:
    return 0.5 * float(np.sum(verts[:, 0] * np.roll(verts[:, 1], -1)
                              - np.roll(verts[:, 0], -1) * verts[:, 1]))


def _radii(radius, count: int) -> np.ndarray:
    """Normalise a scalar or per-corner radius list."""
    if np.isscalar(radius):
        out = np.full(count, float(radius), dtype=np.float64)
    else:
        out = np.asarray(radius, dtype=np.float64)
        if out.shape != (count,):
            raise PrimitiveError(
                f"expected one radius or {count} corner radii, got {out.size}"
            )
    if np.any(~np.isfinite(out)) or np.any(out < 0):
        raise PrimitiveError("radii must be finite and >= 0")
    return out


def _validate_convex(verts: np.ndarray) -> float:
    """Validate a strictly convex polygon and return its turn sign."""
    if len(verts) < 3:
        raise PrimitiveError("polygon needs at least 3 vertices")
    edges = np.roll(verts, -1, axis=0) - verts
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        raise PrimitiveError("polygon has a zero-length edge")
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    nonzero = turns[np.abs(turns) > 1e-10]
    if len(nonzero) != len(turns) or np.any(np.sign(nonzero) != np.sign(nonzero[0])):
        raise PrimitiveError("rounded polygon must be strictly convex")
    return float(np.sign(nonzero[0]))


def _rounded_parts(verts, radius):
    """Tangent points and arc centres for a rounded convex polygon."""
    v = np.asarray(verts, dtype=np.float64)
    turn_sign = _validate_convex(v)
    radii = _radii(radius, len(v))
    edges = np.roll(v, -1, axis=0) - v
    lengths = np.linalg.norm(edges, axis=1)
    directions = edges / lengths[:, None]
    tangent = np.zeros(len(v), dtype=np.float64)
    incoming = np.roll(directions, 1, axis=0)
    outgoing = directions
    for i in range(len(v)):
        dot = float(np.clip(incoming[i] @ outgoing[i], -1.0, 1.0))
        turn = math.acos(dot)
        tangent[i] = radii[i] * math.tan(turn / 2.0)
    for i, length in enumerate(lengths):
        if tangent[i] + tangent[(i + 1) % len(v)] > length + 1e-9:
            raise PrimitiveError(
                f"corner radii overlap on edge {i}; reduce one or both radii"
            )
    before = v - incoming * tangent[:, None]
    after = v + outgoing * tangent[:, None]
    inward = _inward_normals(v)
    centres = before + np.roll(inward, 1, axis=0) * radii[:, None]
    return v, radii, incoming, outgoing, before, after, centres, turn_sign


def rectangle(x: float, y: float, width: float, height: float) -> np.ndarray:
    """Clockwise vertices of an axis-aligned rectangle."""
    if width <= 0 or height <= 0:
        raise PrimitiveError("rectangle width and height must be positive")
    return np.array(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
        dtype=np.float64,
    )


def oriented_rectangle(
    centre: tuple[float, float],
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """Clockwise rectangle about `centre`, with its long axis at `angle_deg`."""
    if length <= 0 or width <= 0:
        raise PrimitiveError("rectangle length and width must be positive")
    angle = math.radians(angle_deg)
    along = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    across = np.array([-along[1], along[0]], dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    a = along * (length / 2.0)
    b = across * (width / 2.0)
    return np.array([c - a - b, c + a - b, c + a + b, c - a + b])


def clip_halfplane(verts, normal: tuple[float, float], limit: float) -> np.ndarray:
    """Clip a convex polygon to `dot(point, normal) <= limit`.

    This is useful for truncated bars and arrow stems without adding a
    logo-specific helper. Vertex order is preserved.
    """
    polygon = np.asarray(verts, dtype=np.float64)
    if len(polygon) < 3:
        raise PrimitiveError("polygon needs at least 3 vertices")
    nrm = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(nrm))
    if norm < 1e-12:
        raise PrimitiveError("half-plane normal must be non-zero")
    nrm /= norm
    bound = float(limit) / norm
    out: list[np.ndarray] = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da = float(a @ nrm - bound)
        db = float(b @ nrm - bound)
        a_inside = da <= 1e-12
        b_inside = db <= 1e-12
        if a_inside:
            out.append(a)
        if a_inside != b_inside:
            t = da / (da - db)
            out.append(a + (b - a) * t)

    # A clipping line through an existing vertex can add that vertex twice.
    # Remove only adjacent numerical duplicates so downstream fillet code
    # never receives a zero-length edge.
    cleaned: list[np.ndarray] = []
    for point in out:
        if not cleaned or float(np.linalg.norm(point - cleaned[-1])) > 1e-10:
            cleaned.append(point)
    if (
        len(cleaned) > 1
        and float(np.linalg.norm(cleaned[0] - cleaned[-1])) <= 1e-10
    ):
        cleaned.pop()

    if len(cleaned) < 3:
        raise PrimitiveError("half-plane removes the whole polygon")
    return np.asarray(cleaned, dtype=np.float64)


def paths(shapes) -> list[list]:
    """Emit one compound path from `(vertices, radius)` primitives."""
    from .geom import rounded_polygon

    out: list[list] = []
    for verts, radius in shapes:
        _rounded_parts(verts, radius)
        out.extend(rounded_polygon(np.asarray(verts).tolist(), radius))
    return out


def erode_convex(verts, radius: float) -> np.ndarray:
    """Shrink a convex polygon by `radius`; vertices become fillet centres.

    Each edge moves inward by radius and adjacent offset edges are
    intersected, so the result's vertices are exactly the centres of the
    corner arcs of the rounded shape.
    """
    v = np.asarray(verts, dtype=np.float64)
    if len(v) < 3:
        raise PrimitiveError("polygon needs at least 3 vertices")
    nrm = _inward_normals(v)
    e = np.roll(v, -1, axis=0) - v
    d = e / np.linalg.norm(e, axis=1)[:, None]
    base = v + nrm * radius                       # a point on each offset line
    out = np.empty_like(v)
    for i in range(len(v)):
        j = (i - 1) % len(v)
        m = np.stack([d[j], -d[i]], axis=1)
        if abs(np.linalg.det(m)) < 1e-12:
            raise PrimitiveError(f"edges {j} and {i} are parallel; cannot fillet")
        t, _ = np.linalg.solve(m, base[i] - base[j])
        out[i] = base[j] + t * d[j]
    # A radius larger than the shape can carry turns the offset lines inside
    # out, and intersecting them pairwise still yields four tidy points --
    # eroding a 10x10 square by 6 gives a 2x2 square with the winding intact,
    # so a winding check does NOT catch it. Test containment instead: every
    # eroded vertex must satisfy every offset half-plane.
    for i in range(len(v)):
        if np.min((out - base[i]) @ nrm[i]) < -1e-9:
            raise PrimitiveError(
                f"radius {radius:g} is too large for this polygon "
                f"(edge {i}'s inward offset overshoots the core)")
    return out


def convex_sdf(P, verts) -> np.ndarray:
    """Signed distance from points P to a convex polygon (negative inside)."""
    P = np.asarray(P, dtype=np.float64)
    v = np.asarray(verts, dtype=np.float64)
    nrm = _inward_normals(v)
    d2 = np.full(len(P), np.inf)
    inside = np.ones(len(P), dtype=bool)
    for i in range(len(v)):
        a = v[i]
        e = v[(i + 1) % len(v)] - a
        w = P - a
        t = np.clip((w @ e) / (e @ e), 0.0, 1.0)
        d2 = np.minimum(d2, ((w - t[:, None] * e) ** 2).sum(1))
        inside &= (w @ nrm[i]) >= 0.0
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def _point_segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    edge = b - a
    t = np.clip(((P - a) @ edge) / (edge @ edge), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * edge), axis=1)


def _angle_delta(angle: np.ndarray | float, start: float, direction: float):
    """Positive angular distance from start in the chosen direction."""
    return np.mod(direction * (angle - start), 2.0 * math.pi)


def _mixed_rounded_sdf(P, verts, radius) -> np.ndarray:
    """Exact boundary distance and sign for independent circular fillets."""
    P = np.asarray(P, dtype=np.float64)
    (
        v,
        radii,
        incoming,
        outgoing,
        before,
        after,
        centres,
        turn_sign,
    ) = _rounded_parts(verts, radius)
    distances = np.full(len(P), np.inf)

    # Flats run from the tangent after one corner to the tangent before the
    # next. A zero radius simply puts both tangent points on the vertex.
    for i in range(len(v)):
        a = after[i]
        b = before[(i + 1) % len(v)]
        distances = np.minimum(distances, _point_segment_distance(P, a, b))

    # Arc distance is radial where the point's angle lies within the fillet,
    # otherwise it is the distance to the nearer tangent endpoint.
    direction = turn_sign
    for i, r in enumerate(radii):
        if r <= 0:
            continue
        c = centres[i]
        start = math.atan2(before[i, 1] - c[1], before[i, 0] - c[0])
        end = math.atan2(after[i, 1] - c[1], after[i, 0] - c[0])
        span = float(_angle_delta(end, start, direction))
        vectors = P - c
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        on_arc = _angle_delta(angles, start, direction) <= span + 1e-12
        radial = np.abs(np.linalg.norm(vectors, axis=1) - r)
        endpoints = np.minimum(
            np.linalg.norm(P - before[i], axis=1),
            np.linalg.norm(P - after[i], axis=1),
        )
        distances = np.minimum(distances, np.where(on_arc, radial, endpoints))

    # Start with the sharp polygon. At each rounded corner, only the wedge
    # between its two tangent points differs: that wedge is inside exactly
    # where it is inside the fillet circle.
    normals = _inward_normals(v)
    inside = np.ones(len(P), dtype=bool)
    for i in range(len(v)):
        inside &= ((P - v[i]) @ normals[i]) >= -1e-12
    for i, r in enumerate(radii):
        if r <= 0:
            continue
        toward_vertex_on_incoming = ((P - before[i]) @ incoming[i]) >= -1e-12
        toward_vertex_on_outgoing = ((P - after[i]) @ outgoing[i]) <= 1e-12
        in_corner_wedge = toward_vertex_on_incoming & toward_vertex_on_outgoing
        inside &= ~in_corner_wedge | (
            np.linalg.norm(P - centres[i], axis=1) <= r + 1e-12
        )
    return np.where(inside, -distances, distances)


def rounded_convex_sdf(P, verts, radius: float | list[float]) -> np.ndarray:
    """Signed distance to a convex polygon with circular corner rounding.

    A scalar radius uses the eroded-core Minkowski construction. A radius
    list uses the exact line and circular-arc boundary with an analytic
    inside test. Both paths return real pixel distances.
    """
    if np.isscalar(radius):
        value = float(radius)
        if value < 0:
            raise PrimitiveError("radius must be >= 0")
        if value == 0:
            return convex_sdf(P, verts)
        return convex_sdf(P, erode_convex(verts, value)) - value
    return _mixed_rounded_sdf(P, verts, radius)


def union_sdf(P, shapes) -> np.ndarray:
    """Signed distance to a union of (vertices, radius) rounded convex shapes."""
    if not shapes:
        raise PrimitiveError("union needs at least one shape")
    return np.min(np.stack([rounded_convex_sdf(P, v, r) for v, r in shapes]), axis=0)


@dataclass
class UnionFit:
    """Result of `fit_union`. `residual` is signed distance in pixels."""

    params: np.ndarray
    residual: np.ndarray
    kept: np.ndarray = field(repr=False)
    contour: np.ndarray = field(repr=False)
    n_trimmed: int = 0

    @property
    def mean(self) -> float:
        return float(np.abs(self.residual).mean())

    @property
    def rms(self) -> float:
        return float(np.sqrt((self.residual ** 2).mean()))

    @property
    def p95(self) -> float:
        return float(np.percentile(np.abs(self.residual), 95))

    @property
    def max(self) -> float:
        return float(np.abs(self.residual).max())

    def summary(self) -> str:
        return (f"mean {self.mean:.4f}  rms {self.rms:.4f}  "
                f"p95 {self.p95:.4f}  max {self.max:.4f}  "
                f"n={len(self.residual)}" + (f"  trimmed {self.n_trimmed}"
                                             if self.n_trimmed else ""))

    def worst(self, k: int = 8) -> list[tuple[int, float]]:
        """Indices into the *kept* contour and their signed residuals."""
        order = np.argsort(-np.abs(self.residual))[:k]
        return [(int(i), float(self.residual[i])) for i in order]

    def worst_points(
        self, k: int = 8
    ) -> list[tuple[int, tuple[float, float], float]]:
        """Original contour index, coordinate, and residual for worst points."""
        source = np.flatnonzero(self.kept)
        order = np.argsort(-np.abs(self.residual))[:k]
        return [
            (
                int(source[i]),
                (float(self.contour[source[i], 0]), float(self.contour[source[i], 1])),
                float(self.residual[i]),
            )
            for i in order
        ]


def fit_union(contour, build, p0, *, bounds=None, trim: float = 0.0,
              passes: int = 2, max_nfev: int | None = None) -> UnionFit:
    """Solve a parametric decomposition against a subpixel contour.

    `build(p)` must return a list of `(vertices, radius)` describing the
    primitives as a function of the parameter vector -- the same list you
    would hand to `union_sdf`. Emit the matching path with
    `geom.rounded_polygon(vertices, radius)` so the model and the fit cannot
    drift apart.

    `trim` discards that fraction of worst-fitting points and refits, which
    is worth reaching for when the union has sharp spikes: rasterising a
    narrow tip rounds it, so those points sit ~1px inside the model through no
    fault of the parameters and would otherwise bend every one of them. Check
    what got trimmed with `worst()` before believing it -- a cluster of
    trimmed points anywhere other than a tip means the decomposition is wrong,
    not the data.

    Deterministic: no RNG, and `least_squares` is a fixed iteration from p0.
    """
    C = np.asarray(contour, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 2:
        raise PrimitiveError("contour must be an (n, 2) array")
    p = np.asarray(p0, dtype=np.float64)
    keep = np.ones(len(C), dtype=bool)
    kw = {"x_scale": "jac"}
    if bounds is not None:
        kw["bounds"] = bounds
    if max_nfev is not None:
        kw["max_nfev"] = max_nfev

    for it in range(max(1, passes) if trim > 0 else 1):
        sub = C[keep]

        def residual(q):
            return union_sdf(sub, build(q))

        out = optimize.least_squares(residual, p, **kw)
        p = out.x
        if trim <= 0.0:
            break
        err = np.abs(union_sdf(C, build(p)))
        cut = float(np.percentile(err, 100.0 * (1.0 - trim)))
        nxt = err <= cut
        if nxt.sum() < max(8, 0.5 * len(C)):
            break
        keep = nxt

    return UnionFit(
        params=p,
        residual=union_sdf(C[keep], build(p)),
        kept=keep,
        contour=C,
        n_trimmed=int((~keep).sum()),
    )


def raster(shapes, width: int, height: int, inset: float = 0.0) -> np.ndarray:
    """Boolean mask of a union of rounded convex shapes, at pixel centres.

    `inset` erodes the result, which is what you want before fitting paint:
    the source rings for 2-4px at every seam, and an un-inset mask feeds those
    blended pixels straight into the gradient fit.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    P = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1).astype(np.float64)
    return (union_sdf(P, shapes) <= -inset).reshape(height, width)


def ink_bounds(shapes):
    """Exact (x0, y0, x1, y1) of a rounded-primitive union.

    Scalar radii use the eroded core. Mixed radii test the line endpoints and
    every cardinal angle that lies on a corner arc. Nothing is sampled.
    """
    if not shapes:
        raise PrimitiveError("union is empty")
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for verts, r in shapes:
        if np.isscalar(r):
            value = float(r)
            core = erode_convex(verts, value) if value > 0 else np.asarray(verts, float)
            lo = np.minimum(lo, core.min(axis=0) - value)
            hi = np.maximum(hi, core.max(axis=0) + value)
            continue
        (
            v,
            radii,
            _incoming,
            _outgoing,
            before,
            after,
            centres,
            turn_sign,
        ) = _rounded_parts(verts, r)
        candidates = [before, after]
        for i, radius in enumerate(radii):
            if radius <= 0:
                continue
            c = centres[i]
            start = math.atan2(before[i, 1] - c[1], before[i, 0] - c[0])
            end = math.atan2(after[i, 1] - c[1], after[i, 0] - c[0])
            span = float(_angle_delta(end, start, turn_sign))
            for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
                if float(_angle_delta(angle, start, turn_sign)) <= span + 1e-12:
                    candidates.append(
                        np.array(
                            [[
                                c[0] + radius * math.cos(angle),
                                c[1] + radius * math.sin(angle),
                            ]]
                        )
                    )
        points = np.concatenate(candidates, axis=0)
        lo = np.minimum(lo, points.min(axis=0))
        hi = np.maximum(hi, points.max(axis=0))
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
