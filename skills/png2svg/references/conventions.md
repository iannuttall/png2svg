# Measurement conventions

Every rule here was paid for with a failed reconstruction. Violating one
costs hours and the symptom is rarely obvious — a model that looks right and
scores badly.

## 1. The half-pixel shift

Pixel index `j` covers SVG coordinate range `[j, j+1]`, so a crossing found
at interpolated index `m` is at SVG `m + 0.5`. Every subpixel measurement
taken in index space needs `+0.5` in **both** axes.

`measure.edge_point` applies this for you (`INDEX_TO_SVG`). Hand-rolled
scans — colour-seam interpolation, centroid estimates, anything you compute
from raw array indices — must add it themselves.

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
away — different edges of the same shape disagree.

`measure.edge_cross` handles the normalisation. It also requires the ray to
**start in background and run into the shape**; a ray starting inside the
foreground returns `None`. When scans mysteriously fail, this is why —
reposition the start into a background gap.

The same asymmetry bites in the comparison step, so `compare.foreground_mask`
uses an adaptive per-pixel 50%-coverage contour for reference and render
alike. Edge metrics additionally run on closed + hole-filled masks, because a
strong internal colour seam otherwise reads as a spurious boundary.

## 3. Corners are squircle cubics, not arcs

Design tools smooth corners with cubic Béziers whose tangent lengths and
handle lengths are independent, not with circular arcs. The signature: the
flat run ends earlier than a circular fit predicts, yet mid-corner curvature
behaves like a *larger* radius.

Fit the cubic. `fit_corner_full(vertex, u_in, u_out, samples)` solves tangent
lengths and handle lengths together; `fit_corner_cubic` fits handles when you
already know the tangent points.

Keep flank samples inside the corner zone. Samples out on the straight sides
blow the fit up (a tangent length of 71px with 1.2px error is the tell), and
scans that are too short to reach the boundary silently return nothing.

## 4. Constrain tangency

Where an arc meets a flat edge, force the arc's pole onto that edge rather
than letting a free fit land near it. If the top and bottom tangents of a cap
disagree by under a pixel, split the cap into two quarter-arcs with
independent vertical radii so each pole lands exactly on its flat edge.

This matters more than it sounds. A round cap between two parallel sides has
radius **exactly** the half-width — fit it freely and you get something like
9.61 where the half-width is 9.56. Emit an SVG arc of radius 9.61 between two
points 19.12 apart and the renderer cannot put the centre on the chord, so it
draws a *shallower* arc: the cap lands a full pixel low. The constrained fit
is also the better fit (rms 0.056 against 0.190 in the P mark), which is the
tell that the constraint was the designer's, not yours.

Whenever a constrained fit beats the free one, believe the constraint.

## 5. Overlap hidden edges boldly

Where one shape tucks under another, extend the hidden edge 1–2px past the
covering edge. Abutting edges leave background slivers and mask pinholes:
two antialiased edges that meet composite to `1-(1-a)(1-b)`, which is less
than full alpha, so the seam shows.

The same effect is why conic fans stroke each wedge with its own gradient —
see [model.md](model.md).

## 6. Read gradient stops off the image

Two approaches depending on the gradient:

**Sampled stops** — median of a 5x5 patch at 9–13 positions along the axis,
margins ~3px from any seam. Robust to anything, costs bytes.

**Fitted stops** — when the gradient is piecewise-linear in sRGB (most design
tools), recover the real construction instead:

1. Find the axis by scanning angles and minimising cross-axis colour
   variance (bin pixels by projection, sum within-bin variance).
2. Fit line segments per channel along that axis; the knee is where one
   channel's slope goes flat and another's starts.
3. Read stop colours off the segment ends.

Saturated channels read ~0.4 low in lossy sources — take the **mode**, not
the mean, or you will chase a phantom `254` that is really `255`. Once you
have candidate stops, refit the stop *positions* against all interior pixels
with the colours fixed; offsets that land on 0.5 or 0/1 are the designer's,
not a coincidence.

## 7. Counters: winding or fill-rule

A second subpath cuts a hole either by running the **opposite way round**
from the outer path (the nonzero rule then cancels it), or by setting
`"fill_rule": "evenodd"` on the shape, which cuts regardless of direction.

Winding is the default and keeps the path portable. Reach for `evenodd` when
the consumer expects it, or when the two subpaths come from separate traces
whose directions you would otherwise have to reason about.

Either way the hole must be a real hole. A background-coloured shape laid on
top looks identical until the artwork is recoloured or composited onto
anything else, at which point the "hole" turns opaque — which is exactly how
it fails in a macOS tinted icon variant.

## 8. Straight-edged logos: fit lines, intersect for vertices

Never trust traced vertices — RDP output is a ±0.5px proposal and its corner
points are the least reliable part of it. Instead:

1. Sample each edge's boundary at 20–30 points, skipping the ends (corners
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
edge by edge — the constraint is linear in (centre, apothem) when the edge
normals are known.

## 9. A constraint is a hypothesis — verify it

Snapping edges to a shared direction, radii to a shared value, an arc to
tangency: each of these is a claim about how the artwork was drawn. Right
ones improve the fit *and* remove a free parameter. Wrong ones move geometry
by tens of pixels — a "fillet" invented between two nearly parallel edges
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

A shadow where a shape crosses itself, a glow, a watermark, a highlight —
these are not part of the fill, but a paint fit that includes them will bend
the whole result toward them. The damage is easy to misread: it makes an
ordinary gradient look like something exotic that the model cannot express.

Fit with `trim` (0.1–0.25), which discards the worst-fitting pixels and
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
  falls back to raw ±0.5px pixels — which renders as a ragged, dripping
  edge. Fit the shape the designer drew instead: an ellipse, a circle, a
  line.

## 11. Free-form outlines: contour, then chain

For artwork that is not built from lines and arcs, use
`measure.subpixel_contour` to get an ordered subpixel boundary and
`curves.fit_bezier_chain` to fit the fewest line and cubic segments within a
tolerance, breaking only at detected corners.

Choose the tolerance against the contour's own noise, not against the
metric target. On a small or soft source the contour itself carries ~0.2px
of noise, and a tolerance below that buys nothing but segments — a hundred
of them where the design has fifteen. If the fit wants far more segments
than the artwork plausibly has, the tolerance is chasing noise.

**Prefer structure where structure exists.** A logo made of straight runs
joined by round corners reconstructs better and far smaller from fitted
lines plus fitted joins than from a generic chain over the whole contour.
That is what `segment_outline` does — longest straight run first, then
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
  under the reported error equal to half the sample spacing.

Before rebuilding: if the measured residuals are a fifth of a pixel and the
score says otherwise, suspect the instrument. `foreground_mask` has a known
weak spot too — it normalises coverage by the local maximum, so a light
region adjacent to a much darker one can fall below the threshold and
produce a spurious internal boundary.

## Noise floors — know when to stop

- **Renderer quantisation**: resvg antialiases with four coverage levels per
  axis, so every rasterised edge snaps to the nearest quarter pixel — ask for
  an edge at y=60.394 and you get 60.50. That is invisible normally but
  dominates subpixel comparison, and it flips an entire pixel row whenever a
  true edge sits near the mask's 50% threshold. `check` therefore renders at
  4x and box-downsamples by default (`--supersample`), which drops the
  quantisation to ±0.031px. On the P mark this moved IoU 0.9938 → 0.9980 and
  edge mean 0.344 → 0.114 **with no change to the model at all**. If a
  reconstruction scores badly while every measured residual is small,
  suspect the instrument before rebuilding the model.
- **Knife edges**: an edge whose true position falls within ~0.05px of a
  pixel centre is a coin toss — reference and render land on opposite sides
  and a whole row reports as missing plus another as excess. Nothing to fix.
- **Resampling ringing**: lossy or resampled sources over/undershoot at
  internal colour seams in ~2px bands. ΔE max of 10–17 confined to 1–2px
  rows along a seam is that floor, not a model error. Judge by p95.
- **Edge quantisation**: `edge_dist_p95` reads 1.0 unless more than 95% of
  boundary pixels land exactly. A p95 of 1.0 with a mean of 0.1 is converged.
- **Sharp spikes**: tips a few pixels wide always disagree by one pixel.
- **Watermarks and artefacts**: reconstruct clean and let ΔE carry the
  difference. Say so in the model notes so the next reader isn't confused.
