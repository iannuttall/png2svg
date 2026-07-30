"""Recover a paint from pixels: flat colour, or a linear gradient's real
construction — axis, stop positions and stop colours.

Design tools build gradients as a few stops interpolated in sRGB, so the
recoverable truth is a handful of numbers, not a sampled curve. Sampling
9-13 stops along the axis reproduces the appearance but hides the
construction and costs bytes; fitting the knots finds what the designer
actually set, and the giveaway is that the answer comes out round — stops
landing on 0.5, an axis through the artwork's centre, channels repeating
across stops.

Everything here returns paint dicts that drop straight into a model shape.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import ndimage, optimize


def _core_pixels(rgb: np.ndarray, mask: np.ndarray, erode: int = 4):
    """Interior pixels only — edge pixels are blends with the background."""
    core = ndimage.distance_transform_edt(mask) >= erode
    if core.sum() < 30:
        core = mask
    ys, xs = np.nonzero(core)
    return np.stack([xs, ys], 1).astype(np.float64), rgb[core].astype(np.float64)


def flat_colour(rgb: np.ndarray, mask: np.ndarray, erode: int = 4) -> str:
    """Median interior colour, as #rrggbb.

    The median rather than the mean: it is unmoved by watermarks, JPEG
    ringing and stray overlays that would drag an average off the true fill.
    """
    _, colours = _core_pixels(rgb, mask, erode)
    med = np.median(colours, axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*np.clip(med, 0, 255))


def gradient_axis(points: np.ndarray, colours: np.ndarray,
                  coarse: float = 3.0, fine: float = 0.05) -> float:
    """Axis along which colour varies, in degrees.

    Found by minimising the colour variance *across* the axis: on the true
    axis every band perpendicular to it is a single colour, so the summed
    within-band variance bottoms out there.
    """
    def score(theta: float) -> float:
        u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
        t = points @ u
        b = np.floor(t - t.min()).astype(int)
        nb = int(b.max()) + 1
        total = 0.0
        for ch in range(3):
            s1 = np.bincount(b, colours[:, ch], nb)
            s2 = np.bincount(b, colours[:, ch] ** 2, nb)
            cnt = np.bincount(b, None, nb)
            ok = cnt > 20
            if not ok.any():
                return np.inf
            var = s2[ok] / cnt[ok] - (s1[ok] / cnt[ok]) ** 2
            total += float((var * cnt[ok]).sum() / cnt[ok].sum())
        return total

    best = min(((t, score(t)) for t in np.arange(-90.0, 90.0, coarse)),
               key=lambda z: z[1])
    lo, hi = best[0] - coarse, best[0] + coarse
    best = min(((t, score(t)) for t in np.arange(lo, hi, fine)), key=lambda z: z[1])
    return float(best[0])


def _basis(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation weights, clamped outside the knots."""
    n, k = len(t), len(knots)
    B = np.zeros((n, k))
    tc = np.clip(t, knots[0], knots[-1])
    idx = np.clip(np.searchsorted(knots, tc, side="right") - 1, 0, k - 2)
    lo, hi = knots[idx], knots[idx + 1]
    span = np.where(hi > lo, hi - lo, 1.0)
    w = (tc - lo) / span
    B[np.arange(n), idx] = 1.0 - w
    B[np.arange(n), idx + 1] = w
    return B


def _fit_stops(t: np.ndarray, colours: np.ndarray, n_stops: int,
               limits: tuple[float, float] | None = None):
    """Least-squares stop colours and positions for a given stop count."""
    lo, hi = limits if limits else (float(t.min()), float(t.max()))
    init = np.linspace(float(t.min()), float(t.max()), n_stops)

    def solve(knots):
        B = _basis(t, knots)
        c, *_ = np.linalg.lstsq(B, colours, rcond=None)
        return c, B @ c

    def resid(free):
        knots = np.concatenate([[free[0]], np.sort(free[1:-1]), [free[-1]]])
        if np.any(np.diff(knots) <= 1e-6):
            return np.full(colours.size, 1e3)
        _, pred = solve(knots)
        return (pred - colours).ravel()

    # Knots stay inside the observed range. Outside it there is no data, so
    # the solver would happily push an endpoint far away and compensate with
    # an extreme colour — a gradient that looks right across the shape but
    # reports stop colours the designer never chose. A true endpoint beyond
    # the artwork is unobservable anyway, and clamping at the edge of the
    # data is indistinguishable from it.
    bounds = ([lo] * n_stops, [hi] * n_stops)
    out = optimize.least_squares(resid, init, bounds=bounds, max_nfev=200)
    knots = np.concatenate([[out.x[0]], np.sort(out.x[1:-1]), [out.x[-1]]])
    c, pred = solve(knots)
    rms = float(np.sqrt(((pred - colours) ** 2).mean()))
    return knots, c, rms


def _round_channels(colours: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Pull saturated channels to their true value.

    A channel pinned at 255 reads back around 254.6 from a lossy source, and
    a fitted stop inherits that bias. Where the interior's mode for a channel
    is 0 or 255 and the fit lands within a couple of levels, take the mode.
    """
    out = np.array(colours, dtype=float)
    for ch in range(3):
        vals = samples[:, ch].astype(int)
        counts = np.bincount(np.clip(vals, 0, 255), minlength=256)
        mode = int(counts.argmax())
        if mode in (0, 255):
            near = np.abs(out[:, ch] - mode) <= 2.5
            out[near, ch] = mode
    return out


def fit_linear_gradient(rgb: np.ndarray, mask: np.ndarray, *,
                        n_stops: int | None = None, erode: int = 4,
                        max_stops: int = 5, target_rms: float = 1.6,
                        snap_offsets: bool = True, trim: float = 0.0,
                        passes: int = 3) -> dict:
    """Fit a linear gradient and return it as a model paint.

    With `n_stops` unset, the fewest stops that reach `target_rms` win — the
    aim is the construction, not the closest possible curve. The result
    carries `rms` and `axis_deg` so the caller can judge the fit and check
    whether the numbers look designed.

    `trim` discards that fraction of worst-fitting pixels and refits, up to
    `passes` times. Anything painted *on top of* the gradient — a shadow
    where a shape crosses itself, an overlay, a watermark — otherwise drags
    the whole fit toward itself, and the damage is easy to misread as the
    gradient being something more exotic than it is. On a logo with a
    self-crossing shadow, `trim=0.12` moved the fit from rms 6.5 to 1.5 and
    turned an apparently unfittable paint into a plain two-stop ramp. Raise
    the trimmed pixels as their own shape rather than pretending they are
    part of the gradient.
    """
    points, colours = _core_pixels(rgb, mask, erode)
    theta = gradient_axis(points, colours)
    u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    t = points @ u

    # Colours are fitted on interior pixels, but the knots may range over the
    # whole shape. Erosion removes exactly the corners where a gradient
    # reaches its extremes, so bounding the knots by the eroded extent would
    # report the colour a few pixels in as if it were the end stop.
    ys, xs = np.nonzero(mask)
    t_full = np.stack([xs, ys], 1).astype(np.float64) @ u
    limits = (float(t_full.min()), float(t_full.max()))

    def fit_all(tt, cc):
        candidates = [n_stops] if n_stops else list(range(2, max_stops + 1))
        chosen = None
        for k in candidates:
            knots, cols, rms = _fit_stops(tt, cc, k, limits)
            if chosen is None or rms < chosen[2]:
                chosen = (knots, cols, rms)
            if n_stops is None and rms <= target_rms:
                break
        return chosen

    knots, cols, rms = fit_all(t, colours)
    if trim > 0.0:
        tt, cc = t, colours
        for _ in range(max(passes, 1)):
            pred = _basis(tt, knots) @ cols
            err = np.linalg.norm(pred - cc, axis=1)
            keep = err <= np.percentile(err, 100.0 * (1.0 - trim))
            if keep.sum() < 50:
                break
            tt, cc = tt[keep], cc[keep]
            knots, cols, rms = fit_all(tt, cc)
        colours = cc

    cols = _round_channels(cols, colours)
    span = knots[-1] - knots[0]
    offsets = (knots - knots[0]) / (span if span > 1e-9 else 1.0)
    if snap_offsets:
        for i, o in enumerate(offsets):
            for nice in (0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0):
                if abs(o - nice) <= 0.02:
                    offsets[i] = nice
                    break

    origin = points.mean(axis=0)
    base = origin - (origin @ u) * u
    p0, p1 = base + knots[0] * u, base + knots[-1] * u
    return {
        "type": "linear",
        "x1": float(p0[0]), "y1": float(p0[1]),
        "x2": float(p1[0]), "y2": float(p1[1]),
        "stops": [
            {"offset": float(round(o, 5)),
             "color": "#{:02x}{:02x}{:02x}".format(
                 *np.clip(np.round(c), 0, 255).astype(int))}
            for o, c in zip(offsets, cols)
        ],
        "axis_deg": float(theta),
        "rms": float(rms),
    }


def fit_shared_ramp(rgb: np.ndarray, pieces, *, n_stops: int = 2,
                    erode: int = 9, exclude: np.ndarray | None = None) -> dict:
    """One ramp shared by several shapes, each in its own local coordinates.

    A duplicated element carries its gradient with it, so N copies show N
    ramps that are the *same* ramp in each copy's own span. Fitting them
    separately wastes the constraint and, worse, each copy usually exposes
    only part of its ramp -- so a per-copy fit extrapolates from a narrow
    window and the copies disagree about the end stops.

    The tell that you are in this situation is a colour DISCONTINUITY where
    two copies meet: one ends saturated exactly where its neighbour restarts
    light. A single gradient across the pair cannot produce that, and
    `fit_linear_gradient` over the union will report non-monotonic stops as it
    tries.

    `pieces` is a list of `(mask, t0, t1)`: the shape's own pixels and the two
    values of the gradient axis that bound *that copy*. Only vertical ramps
    are handled -- t0/t1 are y. Pass `exclude` (a boolean mask) to keep a
    watermark or overlay out of the fit.

    Returns `{"stops": [...], "rms": float, "n": int}`; feed the stops to a
    linear paint per shape with that shape's own y1/y2.
    """
    us, cs = [], []
    for mask, t0, t1 in pieces:
        keep = ndimage.distance_transform_edt(mask) >= erode
        if exclude is not None:
            keep = keep & ~exclude
        if keep.sum() < 50:
            continue
        ys = np.nonzero(keep)[0] + 0.5
        us.append(np.clip((ys - t0) / (t1 - t0), 0.0, 1.0))
        cs.append(rgb[keep].astype(np.float64))
    if not us:
        raise ValueError("no piece had enough interior pixels; lower `erode`")
    u = np.concatenate(us)
    c = np.concatenate(cs)

    knots = np.linspace(0.0, 1.0, n_stops)
    B = _basis(u, knots)
    cols, *_ = np.linalg.lstsq(B, c, rcond=None)
    cols = _round_channels(cols, c)
    rms = float(np.sqrt((((B @ cols) - c) ** 2).sum(1)).mean())
    return {
        "stops": [
            {"offset": float(round(o, 5)),
             "color": "#{:02x}{:02x}{:02x}".format(
                 *np.clip(np.round(col), 0, 255).astype(int))}
            for o, col in zip(knots, cols)
        ],
        "rms": rms,
        "n": int(len(u)),
    }


def _ramp_stop(stop) -> tuple[float, str, float]:
    if isinstance(stop, dict):
        return (
            float(stop["offset"]),
            stop["color"],
            float(stop.get("opacity", 1.0)),
        )
    if len(stop) == 2:
        return float(stop[0]), stop[1], 1.0
    return float(stop[0]), stop[1], float(stop[2])


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError("ramp colours must use #rrggbb")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def sample_ramp(ramp, position: float) -> tuple[str, float]:
    """sRGB colour and opacity at one position in a shared 0..1 ramp."""
    stops = sorted((_ramp_stop(stop) for stop in ramp), key=lambda stop: stop[0])
    return _sample_sorted_ramp(stops, float(position), side="right")


def _sample_sorted_ramp(
    stops: list[tuple[float, str, float]],
    position: float,
    *,
    side: str,
) -> tuple[str, float]:
    """Sample one side of a possibly discontinuous, already sorted ramp."""
    if len(stops) < 2:
        raise ValueError("ramp needs at least two stops")
    exact = [stop for stop in stops if abs(position - stop[0]) < 1e-12]
    if exact:
        chosen = exact[0] if side == "left" else exact[-1]
        return chosen[1], chosen[2]
    if position <= stops[0][0]:
        return stops[0][1], stops[0][2]
    if position >= stops[-1][0]:
        return stops[-1][1], stops[-1][2]
    for left, right in zip(stops, stops[1:]):
        if right[0] == left[0]:
            continue
        if position <= right[0]:
            amount = (position - left[0]) / (right[0] - left[0])
            a, b = _rgb(left[1]), _rgb(right[1])
            rgb = tuple(round(x + (y - x) * amount) for x, y in zip(a, b))
            opacity = left[2] + (right[2] - left[2]) * amount
            return "#{:02x}{:02x}{:02x}".format(*rgb), float(opacity)
    raise AssertionError("unreachable")


def ramp_segment(ramp, start: float, end: float) -> list[dict]:
    """Minimal local stops for the slice `start..end` of a shared ramp.

    This is the key primitive for paint that follows a bent shape. Each
    straight or curved piece gets local offsets 0..1, while all colours come
    from one global distance-along-the-shape ramp.
    """
    start, end = float(start), float(end)
    if abs(end - start) < 1e-12:
        raise ValueError("ramp segment start and end must differ")
    global_stops = sorted((_ramp_stop(stop) for stop in ramp), key=lambda s: s[0])
    low, high = sorted((start, end))
    forward = end > start
    colour, opacity = _sample_sorted_ramp(
        global_stops, start, side="right" if forward else "left"
    )
    entries = [(start, colour, opacity)]
    interior = [stop for stop in global_stops if low < stop[0] < high]
    if not forward:
        interior.reverse()
    entries.extend(interior)
    colour, opacity = _sample_sorted_ramp(
        global_stops, end, side="left" if forward else "right"
    )
    entries.append((end, colour, opacity))
    out = []
    for position, color, opacity in entries:
        stop = {
            "offset": float(round((position - start) / (end - start), 8)),
            "color": color,
        }
        if opacity != 1.0:
            stop["opacity"] = float(round(opacity, 8))
        out.append(stop)
    return out


def map_ramp(paint: dict, start: float, end: float, ramp) -> dict:
    """Copy a linear, radial, or conic paint onto a shared ramp slice."""
    if "stops" not in paint:
        raise ValueError("only gradient paints can be mapped to a ramp")
    result = deepcopy(paint)
    result["stops"] = ramp_segment(ramp, start, end)
    return result
