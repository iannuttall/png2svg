# Worked reconstructions

The source repository publishes three examples. Each one uses artwork Ian
owns or has permission to distribute.

| source | result | what it teaches |
|---|---|---|
| Keep | IoU 0.9966, edge 0.146px, 57 nodes | overlapping primitives, clipping and mixed corner radii |
| IN monogram | IoU 0.9976, edge 0.099px, 26 nodes | fitted polygons and artifact removal |
| P Auto | IoU 0.9988, edge 0.067px, 33 nodes | tangency, fillets and a connected cutout |

Read the closest example before choosing a reconstruction route. The scripts
live under `examples/` in the source repository.

## Keep uses overlapping fitted primitives

**Final result:** IoU 0.9966, edge mean 0.146px, Delta E mean 0.72, 57 model
nodes across 3 shapes and a 1,284-byte compact SVG.

The mark has three disconnected groups. The large right group combines a
central vertical rectangle with two diagonal arms. The left group combines
two clipped diagonal arms with one horizontal bar. A detached arrow forms
the third group.

`analyse` found repeated spacing and corners much tighter than the usual
rounding. Those measurements point to overlapping primitives because the
sharp corners are intersections between shapes. Tracing the visible union
would turn those incidental intersections into permanent path nodes.

The build script declares each construction as a function of a short
parameter vector:

```python
def build_right(p):
    vertical = primitives.rectangle(...)
    upper = primitives.oriented_rectangle(...)
    lower = primitives.oriented_rectangle(...)
    return [(vertical, vr), (upper, upper_r), (lower, lower_r)]

fit = primitives.fit_union(contour, build_right, p0, bounds=bounds)
```

Hidden diagonal ends run beneath the central bar. Their precise length is
unobservable, so the script fixes it long enough to guarantee overlap and
fits only the visible outer endpoint, angle, width and radius.

The left diagonals stop at a measured vertical cut line:

```python
arm = primitives.oriented_rectangle(...)
arm = primitives.clip_halfplane(arm, (1.0, 0.0), cut_x)
```

Corners on that cut stay sharp while the visible outer corners remain
rounded. A radius list such as `[r, r, 0, 0]` passes unchanged through the
fitter, raster mask, crop bounds and path emitter.

The arrow uses the same mixed-radius representation. Its shoulders are
sharp, while its tip and far corners use independent circular rounding.
Describing the symmetry with a centre and half-height removes a parameter
and keeps the two sides tied together.

The model deliberately drops the source's soft shadow, edge glow and
low-level texture. Adding traced shadow shapes barely changes silhouette
overlap and makes the SVG harder to edit. The model notes record that choice
so the remaining colour residual is expected.

See `examples/build_keep_model.py`.

## IN measures four fitted polygons

**Final result:** IoU 0.9976, edge mean 0.099px, Delta E mean 0.60, 26 model
nodes across 4 shapes and a 710-byte compact SVG.

The monogram contains four flat polygons in two colours. Its sides collapse
into two slope families, roughly `+0.45` and `-0.424`. Measuring every edge
independently first reveals that shared design grid.

Each edge is sampled away from its corners, fitted as a line and intersected
with the next fitted edge. This avoids trusting pixel-level contour vertices,
which are least reliable exactly where the direction changes.

Six corners use fitted cubic smoothing. The scripts recover the tangent
points and handle lengths from local boundary samples, then build the final
polygons from those measured lines and cubics.

The source contains a diagonal watermark grid. It fragments automatic
outline analysis and makes flat paint look more complex than it is. The
reconstruction removes the watermark and samples colour from eroded interior
regions. Delta E p95 remains higher along the discarded grid, which is
expected and documented in the model notes.

See `examples/measure_n.py` and `examples/build_n_model.py`.

## P Auto keeps the plug as a real cutout

**Final result:** IoU 0.9988, edge mean 0.067px, Delta E mean 0.71, 33 nodes
in one path and an 852-byte compact SVG.

The first useful fact is topological. The plug opens through the bottom of
the P, so the white region connects to the outside. The entire mark is one
contour with a notch. Treating the plug as a background-coloured shape would
break when the logo is placed over another colour.

The bowl is tangent to flat top and bottom runs. Its upper quarter is about
one pixel fuller than its lower quarter, so a single ellipse leaves a
systematic residual. Two cubic segments hold the measured shape with
horizontal tangents at the flats and a vertical tangent at the right
extreme.

The prong caps expose a common arc mistake. A free circle fit returned a
radius slightly larger than half the measured prong width. SVG could not
place that circle on the chord without moving the cap. Constraining the
radius to the half-width improved the fit and kept the arc exactly tangent.

The plug body and neck sides are parallel vertical runs. Each taper is one
cubic with vertical tangents at both ends. Small body and stem fillets come
from local bisector measurements.

The plug is background inside dark ink, so its measurement rays start within
the light cutout and travel outward into the mark. Reversing that direction
returns no useful transition.

An early check reported IoU 0.9938 and edge mean 0.344px even though the
measured model was correct. Rendering at 4x and downsampling moved the same
model to IoU 0.9983 and edge mean 0.092px. This is the renderer quantisation
case described in [conventions.md](conventions.md).

See `examples/measure_p.py` and `examples/build_p_model.py`.
