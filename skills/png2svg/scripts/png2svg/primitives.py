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

Convexity: `rounded_convex_sdf` requires a convex polygon and one radius,
because it works by eroding the polygon by r and inflating the result. That
is not a real restriction -- non-convex artwork is a union of convex pieces,
which is how it was drawn anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def rounded_convex_sdf(P, verts, radius: float) -> np.ndarray:
    """Signed distance to a convex polygon with circular corner rounding.

    Erode by radius, then inflate: the rounded shape is the Minkowski sum of
    the eroded polygon with a disc, and a Minkowski sum with a disc of radius
    r has signed distance `sdf(core) - r`.
    """
    if radius < 0:
        raise PrimitiveError("radius must be >= 0")
    if radius == 0:
        return convex_sdf(P, verts)
    return convex_sdf(P, erode_convex(verts, radius)) - radius


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

    return UnionFit(params=p, residual=union_sdf(C[keep], build(p)),
                    kept=keep, n_trimmed=int((~keep).sum()))


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
    """Exact (x0, y0, x1, y1) of a union -- the viewBox for a cropped export.

    Solved, not sampled. A rounded convex polygon is the Minkowski sum of its
    eroded core with a disc of radius r, and a Minkowski sum with a disc grows
    the bounding box by exactly r on every side. So the answer is the core's
    bbox pushed out by r -- no marching, no resolution parameter, and exact
    even where the extreme point sits mid-arc rather than at a vertex.
    """
    if not shapes:
        raise PrimitiveError("union is empty")
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for verts, r in shapes:
        core = erode_convex(verts, r) if r > 0 else np.asarray(verts, float)
        lo = np.minimum(lo, core.min(axis=0) - r)
        hi = np.maximum(hi, core.max(axis=0) + r)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
