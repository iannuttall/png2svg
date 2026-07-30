# Measurement conventions

Every rule here was paid for with a failed reconstruction. Violating one
costs hours and the symptom is rarely obvious; a model that looks right and
scores badly.

## 1. The half-pixel shift

Pixel index `j` covers SVG coordinate range `[j, j+1]`, so a crossing found
at interpolated index `m` is at SVG `m + 0.5`. Every subpixel measurement
taken in index space needs `+0.5` in **both** axes.

`measure.edge_point` applies this for you (`INDEX_TO_SVG`). Hand-rolled
scans; colour-seam interpolation, centroid estimates, anything you compute
from raw array indices; must add it themselves.

Symptom when missed: the whole model renders half a pixel up and left, edge
distances sit at a suspiciously uniform ~0.7px, and no local fix helps.

## 2. Normalise coverage, never threshold colour

An edge pixel's value is a coverage blend of foreground and background. To
find the 50%-coverage crossing you must divide by the **local full-strength
foreground level** measured along that same scan ray (the plateau past the
transition), not compare against a fixed colour distance.

A fixed threshold biases dark edges ~0.4px outward relative to light ones,
because a dark shape reaches any given distance-from-background sooner. On a
logo with both dark and light regions this produces an error you cannot fit
away; different edges of the same shape disagree.

`measure.edge_cross` handles the normalisation. It also requires the ray to
**start in background and run into the shape**; a ray starting inside the
foreground returns `None`.

The start must also be in the background connected to the edge being
measured. A fixed 9px offset can land inside a nearby component and make the
first crossing belong to that neighbour. `subpixel_contour` and
`edge_samples` now walk outward from the edge and use the farthest background
point before another foreground shape or the image border. Their `offset`
argument is a maximum, not a required clearance. If a hand-written scan still
fails, apply the same rule rather than increasing its offset.

The foreground plateau must come from the first connected run along the ray.
A fixed sample window several pixels inside crosses straight through a
1-3px stroke and measures background on the far side. `edge_cross` ends the
run after consecutive background samples and takes its high percentile, so a
thin stroke and a thick fill use the same coverage rule.

Resampling can also put one or more weak echo runs up to 4px before the real
edge. `edge_cross` groups nearby weak runs with a stronger plateau and uses
the stronger run for normalization. If no stronger run follows, it retains
the weak one so a real thin, low-contrast stroke is not erased. The ray must
still extend far enough to reach the real edge; if it ends inside the echo
band, increase `t_max`.

The same asymmetry bites in the comparison step, so `compare.foreground_mask`
uses an adaptive per-pixel 50%-coverage contour for reference and render
alike. Edge metrics additionally run on closed + hole-filled masks, because a
strong internal colour seam otherwise reads as a spurious boundary.

## 3. Corners: three candidates, and how to tell them apart

Do not assume which kind you have. There are three, they look alike at a
glance, and each of the first two has been the right answer on real artwork:

1. **Squircle cubics.** Design tools with corner *smoothing* on use cubic
   Béziers whose tangent lengths and handle lengths are independent. The
   signature: the flat run ends earlier than a circular fit predicts, yet
   mid-corner curvature behaves like a *larger* radius.
2. **Circular arcs in final space.** A plain corner radius, tangent to both
   edges, same radius at every corner of the shape. This is the default in
   most tools, and on a parallelogram it produces a *different apparent
   radius* at the sharp corners than at the shallow ones if you fit circles
   without accounting for the corner angle.
3. **Elliptical arcs**, when a rounded rectangle was drawn and then sheared or
   scaled non-uniformly; the shear turns each circular fillet into an arc of
   one common ellipse.

**The discriminator between 2 and 3 is to un-shear and refit.** Take the
contour, apply the inverse of the shear that makes the slanted edges vertical,
and fit circles at the (now 90°) corners. If it is case 3 you get one radius
across every corner; if it is case 2 you get nonsense. On the swipe-s mark the
un-sheared fit gave 0.96-1.14px residuals with two disagreeing radii (7 and
38), while circular-in-final-space gave 0.19-0.34px with one radius. Rejected
in ten minutes, and it would have been an afternoon of wrong geometry
otherwise.

One polygon may genuinely use different circular radii at different corners.
Pass a radius list with one value per vertex. `rounded_convex_sdf`,
`fit_union`, `raster`, `ink_bounds` and `primitives.paths` all accept the same
form. A sharp corner does not need a separate fitting path: use radius zero
for that corner.

Fit the cubic with `fit_corner_full(vertex, u_in, u_out, samples)`, which
solves tangent lengths and handle lengths together; `fit_corner_cubic` fits
handles when you already know the tangent points.

**Size the sample window from the radius, and never use one window for two
corner angles.** A shallow corner's whole fillet sits within ~0.5r of the
vertex; a sharp one's sits 1.3r-2.1r away. One fixed window reports garbage
for the other; a 55px window returned a 73px radius for a 17px fillet, and
identical tangent lengths for eight corners of different sizes. Guess r, set
the window to ~1.1x the tangent length, refit, repeat. Nothing in the output
flags a bad window; only an absurd radius does.

Samples out on the straight sides blow the fit up (a tangent length of 71px
with 1.2px error is the tell), and scans too short to reach the boundary
silently return nothing.

**Shallow corners cannot arbitrate a shared radius**; their fillets are too
flat, so a free fit is worth about ±2px. Let the sharp corners set the radius
and check the shallow ones against it, not the other way round.

## 4. Constrain tangency

Where an arc meets a flat edge, force the arc's pole onto that edge rather
than letting a free fit land near it. If the top and bottom tangents of a cap
disagree by under a pixel, split the cap into two quarter-arcs with
independent vertical radii so each pole lands exactly on its flat edge.

This matters more than it sounds. A round cap between two parallel sides has
radius **exactly** the half-width; fit it freely and you get something like
9.61 where the half-width is 9.56. Emit an SVG arc of radius 9.61 between two
points 19.12 apart and the renderer cannot put the centre on the chord, so it
draws a *shallower* arc: the cap lands a full pixel low. The constrained fit
is also the better fit (rms 0.056 against 0.190 in the P mark), which is the
tell that the constraint was the designer's, not yours.

Whenever a constrained fit beats the free one, believe the constraint.

## 5. Overlap hidden edges boldly

Where one shape tucks under another, extend the hidden edge 1-2px past the
covering edge. Abutting edges leave background slivers and mask pinholes:
two antialiased edges that meet composite to `1-(1-a)(1-b)`, which is less
than full alpha, so the seam shows.

The same effect is why conic fans stroke each wedge with its own gradient.
See [model.md](model.md).

## 6. Read gradient stops off the image

Two approaches depending on the gradient:

**Sampled stops**; median of a 5x5 patch at 9-13 positions along the axis,
margins ~3px from any seam. This tolerates compression and overlays, but
costs bytes.

**Fitted stops**; when the gradient is piecewise-linear in sRGB (most design
tools), recover the real construction instead:

1. Find the axis by scanning angles and minimising cross-axis colour
   variance (bin pixels by projection, sum within-bin variance).
2. Fit line segments per channel along that axis; the knee is where one
   channel's slope goes flat and another's starts.
3. Read stop colours off the segment ends.

Saturated channels read ~0.4 low in lossy sources; take the **mode**, not
the mean, or you will chase a phantom `254` that is really `255`. Once you
have candidate stops, refit the stop *positions* against all interior pixels
with the colours fixed; offsets that land on 0.5 or 0/1 are the designer's,
not a coincidence.

### A duplicated shape carries its gradient with it

If a shape was copied, each copy has the **same ramp in its own local span**,
not one ramp spanning both and not two unrelated ramps. Fitting a single
gradient across the pair is the mistake that is easy to make and hard to read
afterwards: `fit_linear_gradient` returns non-monotonic stops, sometimes with
a stop borrowed from a neighbouring colour entirely, and the rms only looks
mildly bad.

**The tell is a colour discontinuity where two copies meet**; one ends
saturated exactly where its neighbour restarts light. No single gradient can
do that.

Fit them together with `paint.fit_shared_ramp`, passing each copy's own mask
and its own axis bounds. Pooling matters because each copy usually exposes
only *part* of its ramp, so a per-copy fit extrapolates from a narrow window
and the copies disagree about the end stops.

Pass each **individual shape's** mask, not the whole region's mask twice with
different bounds; the latter sends every pixel through both spans and the
ramp collapses (rms 13.4 against 1.7).

### Paint that follows a bent shape

There is no portable SVG fill that follows distance along an arbitrary path.
Reconstruct it as local pieces: a linear paint for each straight run and a
conic paint for each turn. Assign every piece a start and end position in one
global 0..1 travel ramp, then call `paint.map_ramp`.

The spans may run backwards. A top bar travelling right to left might use
`0.24 -> 0.00`, followed by a left cap using `0.24 -> 0.45`.
`map_ramp` keeps only the global knots inside each span and converts them to
local offsets. Adjacent spans therefore meet at the exact same colour without
copying 9 to 13 sampled stops into every paint. Duplicate-offset hard stops
use the correct side of the discontinuity even when a span is reversed.

## 7. Counters: winding or fill-rule

A second subpath cuts a hole either by running the **opposite way round**
from the outer path (the nonzero rule then cancels it), or by setting
`"fill_rule": "evenodd"` on the shape, which cuts regardless of direction.

Winding is the default and keeps the path portable. Reach for `evenodd` when
the consumer expects it, or when the two subpaths come from separate traces
whose directions you would otherwise have to reason about.

Either way the hole must be a real hole. A background-coloured shape laid on
top looks identical until the artwork is recoloured or composited onto
anything else, at which point the "hole" turns opaque; which is exactly how
it fails in a macOS tinted icon variant.

## 8. Straight-edged logos: fit lines, intersect for vertices

Never trust traced vertices; RDP output is a ±0.5px proposal and its corner
points are the least reliable part of it. Instead:

1. Sample each edge's boundary at 20-30 points, skipping the ends (corners
   bend the boundary): `measure.edge_samples`.
2. Fit each edge with total least squares: `measure.fit_line` (not
   `fit_line_x_of_y`, which assumes a steep edge).
3. Take vertices from `measure.intersect` of adjacent edge lines.
4. Check the fitted **angles** for structure. Families that repeat to within
   ~0.05° are the design grid. Snap those; leave genuinely free edges free.
5. Re-check every sample against the final polygon. Deviations should stay
   under ~0.2px; a single edge above that means you snapped something that
   was not actually regular.

Where a frame is provably regular (a hexagon pair, a shared centre), fit it
as **one** constrained system from all its edge samples at once rather than
edge by edge; the constraint is linear in (centre, apothem) when the edge
normals are known.

### A symmetry deletes parameters rather than constraining them

Write the construction down as a function of a parameter vector, and
quantities you were about to measure stop being parameters at all. On a mark
of three rounded parallelograms with 180° symmetry, the offset between them
and each one's width both fell out of the symmetry: eight numbers fixed the
whole silhouette to a mean residual of 0.080px. See
[examples.md](examples.md) for the construction.

Use `primitives.fit_union` for this rather than rolling your own solve; it
takes the contour and a `build(p)` returning `(vertices, radius)` pairs, and
minimises signed distance in pixels, directly comparable to the edge targets.

**Measure the redundant quantity anyway, before you derive it.** Two
independent estimates of that offset agreeing to 0.02px is what proved the
decomposition. Deriving it from the start would have hidden a wrong guess.

## 9. A constraint is a hypothesis; verify it

Snapping edges to a shared direction, radii to a shared value, an arc to
tangency: each of these is a claim about how the artwork was drawn. Right
ones improve the fit *and* remove a free parameter. Wrong ones move geometry
by tens of pixels; a "fillet" invented between two nearly parallel edges
that never met at a corner at all.

Only measurement separates them, so `snap_outline` takes the contour and
re-scores every change, keeping what fits better and reverting the rest. On
a real mark it proposed 19 constraints and rejected 8; applying them blind
gave edge max 22.6px, verified it gave 1.0px, with an identical fit
underneath.

The same rule applies to constraints you enforce by hand. When a constrained
fit beats the free one, the constraint was the designer's. When it does not,
you invented it.

Two shapes that meet also need care: vertices may not travel far when a join
is recomputed. Two nearly parallel edges intersect a long way off, and
moving the join there turns a soft corner into a spike.

## 10. Anything painted on top corrupts the fit beneath

A shadow where a shape crosses itself, a glow, a watermark or a highlight is
not part of the fill, but a paint fit that includes it will bend
the whole result toward them. The damage is easy to misread: it makes an
ordinary gradient look like something exotic that the model cannot express.

Fit with `trim` (0.1-0.25), which discards the worst-fitting pixels and
refits. On one mark this moved a gradient from rms 6.5 to 1.5 and turned an
apparently unfittable paint into a plain two-stop ramp. Then raise the
trimmed region as its own shape, rather than pretending it belongs to the
gradient.

Related shape traps:

- **A ring cannot carry a linear gradient.** A ramp fitted across one runs
  from one side, through the hole in the middle, to the other. Use the
  median colour.
- **Colour-based masks need a spatial cut.** A bright unsaturated waveform
  inside a screen matched the same test as a brushed metal plate, joined its
  component, and dragged the plate's outline down through the middle of the
  icon. Bound the region by where it can actually be.
- **A soft colour boundary inside a shape cannot be traced.** Both sides are
  foreground, so there is no background to scan against and the contour
  falls back to raw ±0.5px pixels; which renders as a ragged, dripping
  edge. Fit the shape the designer drew instead: an ellipse, a circle, a
  line.

## 11. Free-form outlines: contour, then chain

For artwork that is not built from lines and arcs, use
`measure.subpixel_contour` to get an ordered subpixel boundary and
`curves.fit_bezier_chain` to fit the fewest line and cubic segments within a
tolerance, breaking only at detected corners.

Choose the tolerance against the contour's own noise, not against the
metric target. On a small or soft source the contour itself carries ~0.2px
of noise, and a tolerance below that buys nothing but segments; a hundred
of them where the design has fifteen. If the fit wants far more segments
than the artwork plausibly has, the tolerance is chasing noise.

**Prefer structure where structure exists.** A logo made of straight runs
joined by round corners reconstructs better and far smaller from fitted
lines plus fitted joins than from a generic chain over the whole contour.
That is what `segment_outline` does; longest straight run first, then
longest arc, Béziers only where the outline is genuinely free-form. Reach
for `fit_bezier_chain` directly only when the shape has no straight
structure at all.

## 12. Judge the instrument before rebuilding the model

Twice in this tool's history a reconstruction scored badly while being
correct. Both times the measuring apparatus was at fault:

- resvg rasterises edges to the nearest **quarter pixel**, which flips whole
  pixel rows when a true edge sits near the mask's 50% threshold (fixed by
  supersampling in `check`);
- comparing a fitted curve against a fixed-resolution sampling puts a floor
  under the reported error equal to half the sample spacing;
- a fully opaque SVG has no partial-alpha edge band. The halo check treats
  that empty sample as not applicable and passing, rather than taking the mean
  of an empty array.

Before rebuilding: if the measured residuals are a fifth of a pixel and the
score says otherwise, suspect the instrument. One previous failure came from
`foreground_mask` normalising a light region by an unrelated darker colour
across an internal seam. It cut a false channel through the silhouette and
moved a primitive fit from mean 0.10px to 0.28px. The mask now rechecks
rejected pixels against neighbours with the same RGB direction from the
background. The regression case is a hard light/dark seam with no actual
background between the colours.

## Noise floors; know when to stop

- **Renderer quantisation**: resvg antialiases with four coverage levels per
  axis, so every rasterised edge snaps to the nearest quarter pixel; ask for
  an edge at y=60.394 and you get 60.50. That is invisible normally but
  dominates subpixel comparison, and it flips an entire pixel row whenever a
  true edge sits near the mask's 50% threshold. `check` therefore renders at
  4x and box-downsamples by default (`--supersample`), which drops the
  quantisation to ±0.031px. On the P mark this moved IoU 0.9938 -> 0.9980 and
  edge mean 0.344 -> 0.114 **with no change to the model at all**. If a
  reconstruction scores badly while every measured residual is small,
  suspect the instrument before rebuilding the model.
- **Knife edges**: an edge whose true position falls within ~0.05px of a
  pixel centre is a coin toss; reference and render land on opposite sides
  and a whole row reports as missing plus another as excess. Nothing to fix.
- **Resampling ringing**: lossy or resampled sources over/undershoot at
  internal colour seams in ~2px bands. ΔE max of 10-17 confined to 1-2px
  rows along a seam is that floor, not a model error. Judge by p95.
- **Edge quantisation**: `edge_dist_p95` reads 1.0 unless more than 95% of
  boundary pixels land exactly. A p95 of 1.0 with a mean of 0.1 is converged.
- **Sharp spikes**: tips a few pixels wide always disagree by one pixel. Where
  two primitives cross at a shallow angle the union has a genuine spike, and
  rasterising rounds it: expect the contour to read **1.2-1.5px inside** the
  model for the handful of points at the tip. Leave it sharp; it is sharp in
  the artwork. Absorb it with `fit_union(..., trim=0.05)` so it does not bend
  every parameter, and confirm via `fit.worst_points()` that the reported
  original contour coordinates really are at tips.
- **Watermarks and artefacts**: reconstruct clean and let ΔE carry the
  difference. Say so in the model notes so the next reader isn't confused.
