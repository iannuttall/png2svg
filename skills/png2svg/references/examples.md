# Worked reconstructions

Read the one closest to your image before starting. Every script referenced
lives in `examples/` in the source repository.

| source | result | what it teaches |
|---|---|---|
| plug 'P' mark | IoU 0.9983, edge 0.092px, 33 nodes | tangency, fillets, hand measurement |
| 'N' monogram | IoU 0.9976, edge 0.101px, 26 nodes | flat polygons, discarding a watermark |
| gradient loop | IoU 0.9873, 108 nodes | counters, a cap drawn over an end, trimming a shadow out of a gradient |
| ribbon + cylinder | IoU 0.9740, 85 nodes | occlusion in three pieces, fitting an ellipse instead of tracing one |
| textured app icon | IoU 0.9264, low-freq ΔE 5.3 | "good enough": structure kept, grain dropped |

The last two are worth reading even if your image looks nothing like them:
they are where the decisions are, and the decisions are the part no library
can make for you.

## Curves, fillets and tangency: a plug 'P' mark (498x524)

**Final**: IoU 0.9983, edge mean 0.092px (max 1.0), ΔE2000 mean 0.71 / p95
1.60, 33 nodes in one path, 998-byte export.

A flat black P with a plug notched out of it. The first thing to establish
was that the plug **opens at the bottom** — it is not a counter, so the whole
mark is one contour. Tracing the filled mask and subtracting the mask itself
finds holes; here there were none, which settled it in one command.

What the measurements revealed:

- **The bowl is tangent to both flats.** Profiling the boundary along the top
  showed y = 60.39 dead flat from x=90 to x=270 (varying by 0.03px), then
  departing smoothly. Same at the bottom. That fixes the bowl's vertical
  extent exactly and removes two parameters.
- **It is not a conic.** The right side is about a pixel fuller above the
  midline than below. A free ellipse left a systematic residual pattern, so
  the bowl is modelled as a type designer would draw it: two cubics with
  horizontal tangents at the flats and a vertical tangent at the right
  extreme, handle lengths free. Four anchor points, two segments.
- **The prong caps had to be tangency-constrained** — see convention 4. The
  free fit gave r=9.612 against a half-width of 9.559, and the resulting SVG
  arc rendered a full pixel low. Forcing r to the half-width both fixed it
  and fitted better (rms 0.056 against 0.190).
- **The plug tapers are S-cubics**: the body sides and the neck sides are all
  vertical, so each transition is one cubic with vertical tangents at both
  ends — two free handle lengths and two endpoint positions.
- **Fillet radii came from one scan each.** For a fillet of radius r between
  edges meeting at half-angle θ, the vertex-to-arc distance along the
  bisector is r/sin(θ) − r. Measure that distance and r follows. The mark
  uses 3.6–4.9 on the stem, 7.1 on the plug body, 1.6–2.9 where the prongs
  meet it.

The plug is a light region *inside* the ink, which inverts the scan
direction: its rays start in the plug interior and run outward. Getting that
backwards returns zero samples every time (convention 2).

Note also that the first score was IoU 0.9938 with edge mean 0.344 — bad
enough to suggest a broken model, when the model was already correct. It was
renderer quantisation. See the noise floors in
[conventions.md](conventions.md) before rebuilding anything.

See `examples/measure_p.py` and `examples/build_p_model.py` in the source
repository.

## Flat polygons with an artefact: an N-monogram (1504x1128 WebP)

**Final**: IoU 0.9976, edge mean 0.101px, ΔE mean 0.60 / p95 5.17, 26 nodes,
775-byte export. Converged in one iteration.

Four flat polygons, two colours, six rounded corners. Sides fell into two
slope families (+0.45 / −0.424) — matching each measured side to its family
was most of the work.

The source carried a diagonal stock watermark. It was reconstructed **clean**
and the watermark deliberately discarded, which is why p95 is elevated. That
choice is recorded in the model notes. The watermark also fooled the
automatic `analyse` pass — it fragmented straight edges into "curves" and
pushed paint probes to `linear`/`complex` — so colours were sampled as
medians of eroded interiors instead. Treat `analyse` output as a proposal.

## Occlusion: a ribbon behind a cylinder

**Final**: IoU 0.9740, ΔE mean 1.27 / p95 2.31, 85 nodes across 5 shapes.
Script: `examples/build_r_model.py`, ~130 lines, almost all of it decisions.

A ribbon spirals behind a cylinder, so it arrives as **three separate
visible pieces**. No occlusion machinery was needed — each piece is its own
region — but each is grown underneath the cylinder before tracing, and the
cylinder is painted last, so no hairline of background can show along a
join. Grow into the occluder itself, never a dilated copy of it: a ribbon
allowed past the cylinder's own edge hangs out over the background, which
costs far more than the seam it was avoiding.

The cylinder's top face taught the other lesson. Traced, it came out ragged
and dripping, because the blue/dark boundary is a soft colour transition
*inside* the silhouette — no background to scan against, so the contour fell
back to raw pixels. Fitted as the ellipse it obviously is, max residual
0.65px. When a boundary is internal, fit the shape rather than trace it.

## A gradient loop with a cap and a shadow

**Final**: IoU 0.9873, ΔE mean 1.02, 108 nodes. Script:
`examples/build_q_model.py`.

Three things made this one instructive. Its width looked constant (medial
axis median 60.0) so it looked like a stroked path — but measured *per run*
the widths were 58.9, 63.9 and 56.8, steady within each run and different
between them. A single stroke would have been wrong by 3.5px. **Verify a
width per run before believing it is a stroke.**

Its paint fitted a gradient at rms 6.5, which read as "not a linear
gradient". It was: a shadow where the stroke crosses itself was dragging the
fit. With `trim=0.12` the same paint came back as a plain two-stop ramp at
rms 1.5.

Its outline was traced from the loop **alone**, not from the silhouette.
Tracing the union would have put the cap's outline into the loop's path,
where the two meet at sharp concave junctions that any segmenter cuts the
corner on. Tracing the loop alone and running it under the cap took the
outer path from 59 primitives to 25.

## "Good enough": a textured app icon

**Final**: IoU 0.9264, `texture_std` 4.3, low-frequency ΔE 5.3 / 22.5, 351
nodes. Script: `examples/build_x_model.py`.

A 3D-rendered icon: brushed metal, grain, drop shadows, a glow, engraved
text and a scanline waveform. Its interior varies by ~15 levels on its own,
so per-pixel ΔE cannot be satisfied and chasing it would be chasing noise.

Reconstructed: backdrop wash, dark shell, metal plate, screen, slot.
Dropped deliberately, and recorded in the model notes: grain, brushed
streaks, engraved text, the waveform. Outlines fitted at `tol=1.2px` — near
the texture scale, because fitting tighter than the grain fits the grain.

Known remaining defects, since these are more useful than a clean-looking
summary: the waveform still cuts a notch into the screen, and `validate`
fails its halo checks here because the design covers the whole canvas while
those checks assume artwork sits on transparency.

## Straight edges on a design grid

For a logo built entirely from straight lines, the method in convention 8 is
the whole job: sample every edge, fit each as a line, take vertices from the
intersections, then read the fitted angles for structure.

Angles that collapse into a few families are the design grid confessing
itself. When a frame is provably regular — say an outer and inner hexagon
sharing a centre — fit it as **one** constrained system rather than edge by
edge: with the edge normals known, `(Cx, Cy, apothem_out, apothem_in)` is a
linear least-squares problem over all of that frame's samples at once, and it
will fit tighter than any edge fitted alone.

Then verify the coincidences before enforcing them — a vertex landing exactly
on another shape's edge, or on the frame's centre, is a real constraint worth
keeping. Leave anything that does not survive the check free: forcing two
near-parallel diagonals onto the grid can cost more than it saves.

## Gradients

Where a gradient is piecewise-linear in sRGB, recover its construction rather
than sampling it (convention 6). Find the axis by minimising cross-axis
colour variance, fit line segments per channel, and read the stop colours off
the segment ends. Then refit the stop *positions* with the colours held
fixed. Offsets that land exactly on 0.5, or an axis that passes precisely
through the artwork's centre, are the designer's decisions rather than
coincidences — and a palette whose channels repeat across stops is a good
sign the fit found the real construction rather than a local minimum.

## Overlapping primitives: a slanted "S" of three bars (1500x1125 WebP)

The case where tracing is the wrong tool. Read it before reaching for
`segment_outline` on anything that might be built from repeated shapes.

The silhouette is a Z with two notches, ten straight runs and ten corners —
`analyse` reports exactly that, and traced it costs around forty nodes. But
the corner radii it returns are the giveaway: four at ~17.0-17.4, two that fit
as 122.7, and two at ~2.5. A designer does not use five radii on one outline.
The 122.7 pair were shallow corners measured with too wide a window
(convention 3), and the ~2.5 pair were not corners at all — they were the
points where two shapes cross.

Three tells said "overlapping primitives", and any one would have been enough:

- **A seam collinear with a silhouette edge.** The lower-left outer edge,
  extended up past the notch, continued as the pink/blue colour seam to within
  0.05px over 300px. One line serving as an outer boundary in one place and an
  interior seam in another means two shapes sharing an edge.
- **A spacing that repeats.** 142.979 between the two left slant lines,
  143.002 between the two right ones.
- **A middle colour that is exactly an overlap.** Pink band 144px wide, cyan
  band 144px wide, and the blue between them widening from 147 to 314 exactly
  as the two Z-bands' intersection would.

What it actually is: three rounded parallelograms sharing one slant, with 180°
symmetry about `(750.541, 559.487)`. Two congruent Z-bands (pink = upper-left
rect + lower rect; cyan = the 180° rotation of it) overlap, and the overlap is
a separate flat fill.

    U   = [cx-a, cx-a+2b] x [cy-g, cy+k]     pink upper
    P2  = [cx-b, cx+b   ] x [cy-g, cy+g]     = pink lower + cyan upper
    Lo' = [cx+a-2b, cx+a] x [cy-k, cy+g]     cyan lower

Eight numbers — slant `128.39244`, centre, `a`, `b`, `g`, `k`, radius
`17.2655` — fixed all of it via `primitives.fit_union` against 2087 contour
points: **mean 0.080px, p95 0.221px**. The band offset (`a - b` = 143.049) and
each short rect's width (`2b`) fell out of the symmetry; both had been measured
independently first, and their agreement to 0.02px is what confirmed the
decomposition before anything was derived from it.

Three things this mark taught that generalise:

1. **The overlap's corners are free.** Blue is `pink ∩ cyan`, and it decomposes
   into three more rounded parallelograms whose every corner is a genuine
   corner of one of the four band rects. So all twelve of its fillets are the
   artwork's own `r`; none had to be invented for an intersection.
2. **Each parallelogram carries its own ramp.** There is a hard colour jump at
   the seam tip — the upper rect ends `#fd30f8` where the lower restarts
   `#fc59e9`. Four gradients, not two, and the two rects of a band share one
   ramp in local y (`paint.fit_shared_ramp`, rms 1.7 pooled).
3. **The blue overlap is flat**, at `#4548e9` with a spread of 0.8. Worth
   checking before assuming a blend mode: a multiply of two gradients would not
   be flat, so this was drawn as its own shape.

Final: IoU 0.9988, edge mean 0.081px, ΔE mean 0.91, p95 2.52, at 63 nodes and
2.8KB. The ΔE max of 58.9 is a "logo for sale" watermark badge that was
deliberately not reconstructed, and the rest of the residual is the source's
own sharpening ringing — it undershoots to `(200,89,181)` one pixel inside
edges, darker than either side, which no clean vector can or should match.
