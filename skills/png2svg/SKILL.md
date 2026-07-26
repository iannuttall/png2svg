---
name: png2svg
description: Reconstruct a geometric PNG (logo, icon, monogram) as a clean native SVG with exact geometry and gradients, using an analyse -> measure -> model -> check -> residuals iteration loop scored by perceptual metrics. Use when asked to vectorise, trace or convert a PNG/WebP logo to SVG, recreate a logo's geometry, rebuild a logo as editable vector, or produce colour variants of a reconstructed logo. Not for photos, textures, or organic artwork.
license: MIT
compatibility: Requires Python 3.12+ and uv. Scripts fetch their own dependencies into a cached environment on first run; nothing else needs installing.
metadata:
  author: iannuttall
  version: "0.1.0"
---

# png2svg: geometry-first PNG → SVG reconstruction

You are the editor in this loop. Deterministic commands measure, render, and
score; you make the design judgments — shape decomposition, layering, paint
model — and iterate until the metrics converge.

This is reconstruction, not tracing. The output is the handful of shapes a
designer would have drawn, not a contour fitted to pixels.

## 0. Triage — is this image suitable?

**Look at the image first.** Proceed when it reads as designer-built
geometry: flat or gradient fills, crisp edges, shapes decomposable into
lines, arcs and Béziers.

Rule of thumb: **if it could have been built in Figma from shapes and
gradients, it is recoverable.** There are three outcomes, and picking the
right one up front decides which targets you are aiming at:

| the source | aim for | judge by |
|---|---|---|
| flat or gradient fills, crisp edges | an exact reconstruction | IoU, edge distance, ΔE |
| a render with grain, brushed metal, shadows, glow | **structure and shading; drop the texture** | IoU and `deltaE_lowfreq` |
| photos, painterly art, anything under ~100px | decline | — |

The middle case is the one to get right rather than refuse. A textured
source disagrees with **itself** by more than any vector can match, so
per-pixel ΔE will report failure for a reconstruction that reads correctly
at a glance. `check` reports `texture_std` — around 1-2 for clean vector
art, 10+ for a textured render. When it is high, reconstruct the structure
and the large-scale shading, drop grain and fine detail deliberately, say so
in the model notes, and judge by the low-frequency figures.

For the third case say so plainly and suggest `vtracer` or `potrace`, which
do the different job of fitting contours to pixels.

## 1. Setup

Everything runs through one entry point: `scripts/png2svg_cli.py` in this
skill's directory. On first run uv builds a cached environment from the
script's own dependency block — there is nothing to install.

Below, `$SKILL` stands for this skill's directory (the folder containing this
file). **Substitute the literal path**: each command may run in a fresh
shell, so an assignment made in one will not survive to the next.

```bash
uv run --no-project "$SKILL/scripts/png2svg_cli.py" init INPUT.png --project work/<name>
uv run --no-project "$SKILL/scripts/png2svg_cli.py" analyse work/<name>
```

`init` copies the source and estimates the background from a 2px border
median. `analyse` writes `analysis/features.json` and `analysis/overlay.png`:
components with bboxes, boundary segments (line / arc / corner / curve with
fitted params and errors), and a paint probe per component —

- `flat` — solid colour
- `linear` — gradient, dominant direction given
- `angular` — conic sweep, centre given
- `complex` — layered or occluded paint; your judgment needed

Each component also carries a `structure` block, and `analyse` prints its
`hint` line if it found anything: repeated spacings between parallel edges,
exact 180°/mirror symmetry with its centre, and corners far tighter than the
shape's usual radius. **Read that line first** — it is the fastest available
answer to the route question in §2, and every number in it is one you would
otherwise derive by hand. A repeated spacing or a tight corner means
overlapping primitives; a symmetry centre means parameters you can delete
rather than fit.

It reports only what it measured. Silence means nothing was found, not that
the mark is simple.

**Treat every coordinate in features.json as a ±0.5px proposal**, and read
`overlay.png` next to the source before deciding anything. `analyse` is
fooled by watermarks and compression artefacts.

## 2. Decompose — and pick your route

Decide the structure before measuring anything: how many overlapping
primitives, what occludes what, where the paint seams are.

A hard colour boundary **inside** one silhouette usually means two
overlapping shapes — the seam is the top shape's edge, not a gradient stop,
and it often passes through an arc centre. Getting this right is most of the
work; everything downstream is arithmetic.

This decision also picks which of **two routes** you measure with. They are
siblings, not a default and a fallback — reaching for the wrong one costs you
either a pile of nodes or an afternoon:

| what you are looking at | route | §  |
|---|---|---|
| one silhouette whose boundary **is** the design — a letterform, a swoosh, an organic blob | **trace it** | 3a |
| overlapping filled primitives — bars, rects, discs, capsules — where the union's boundary is a *by-product* of where they landed | **fit the primitives** | 3b |

`analyse`'s `structure` hint checks the first two of these for you. Three
tells, any one of which is enough:

- **A corner that is an intersection, not a fillet.** Walk the silhouette and
  ask of each corner: did a designer round this, or did it appear because two
  shapes crossed? Sharp spikes and tiny radii next to generous ones mean
  crossings.
- **Coincidences at a distance.** Two edges collinear but far apart; a spacing
  that repeats; 180°/mirror symmetry across the whole mark. Separate shapes
  do not accidentally line up — that is one primitive, duplicated.
- **A colour region that is exactly an overlap.** Three bands where the middle
  one is the intersection of the outer two.

Tracing in the second case is not merely bigger, it throws the structure
away: the constraints that made the artwork regular are exactly what a
contour fit cannot see.

## 3a. Measure by tracing

Copy `scripts/measure_template.py` next to your work and edit the marked
sections — it carries the whole pipeline and writes `project.json` directly.
(`scripts/build_template.py` is the split-file variant, worth it only when
hand measurement dominates and you want `analysis/measurements.json` as a
separate artefact.)

For each region you decided on above:

```python
C = subpixel_contour(field, region)            # ordered subpixel boundary
prims = segment_outline(C, tol=0.4)            # lines / arcs / cubics
prims, notes = snap_outline(prims, contour=C)  # constraints, each verified
segments = to_segments(prims)                  # model path segments
```

That is a complete outline in four calls, and on a clean logo it lands
within a fifth of a pixel. Read `notes` — it says which constraints were
kept and which were rejected for making the fit worse, which tells you how
the shape is built.

`tol` is the one dial. It trades segment count against deviation; sweep it
(0.25 to 0.6 suits most sources) and take the knee. If the fit wants far
more segments than the artwork plausibly has, the tolerance is chasing
noise. On a textured source, set it near the texture scale — fitting tighter
than the grain fits the grain.

## 3b. Measure by fitting primitives

Declare the decomposition as a function of a parameter vector and solve for
the parameters against every contour point at once:

```python
def build(p):                                  # -> [(vertices, radius), ...]
    ang, cx, cy, a, b, g, k, r = p
    ...                                        # your construction, in code
    return [(rect_1, r), (rect_2, r), (rect_3, r)]

fit = primitives.fit_union(contour, build, p0)  # signed distance -> zero
print(fit.summary())                            # mean/rms/p95/max, in pixels
d = geom.rounded_polygon(vertices, radius)      # emit the same shape as a path
```

The residual is a real distance in pixels, so it reads straight against the
edge targets in section 5. Build the model path with `geom.rounded_polygon` from the
*same* vertices you fitted, so the fit and the model cannot drift apart.

**Let the symmetry remove parameters.** This is where the route pays. On one
mark, three rounded parallelograms with 180° symmetry came to **eight**
numbers — and the offset between primitives and each one's width were
*consequences* of the symmetry rather than things to fit. Eight numbers, mean
residual 0.08px, against ~40 nodes for the traced equivalent. Every parameter
you can derive instead of fit is one that was absorbing noise.

Use `fit_union(..., trim=0.05)` when the union has sharp spikes: rasterising a
narrow tip rounds it, so those points sit ~1px inside the model through no
fault of the parameters. Check `fit.worst()` before believing a trim — trimmed
points anywhere other than a tip mean the decomposition is wrong, not the data.

## 3c. Then measure by hand what no fitter can know

A tangency, a shared centre, a radius that is exactly a half-width: these are
design decisions, and each one you confirm removes a free parameter. This is
where the reconstruction goes from good to exact.

| | |
|---|---|
| `segment_outline` + `snap_outline` + `to_segments` | trace an outline (3a) |
| `primitives.fit_union` | fit overlapping rounded primitives as one system (3b) |
| `geom.rounded_polygon`, `geom.smooth_polygon` | emit a filleted / squircled polygon as path segments |
| `primitives.raster`, `primitives.ink_bounds` | masks for paint fitting; exact crop box |
| `edge_samples` + `fit_line` + `intersect` | straight edges → exact vertices |
| `fit_circle`, `fit_corner_full` | arcs and squircle corners |
| `curves.fit_bezier_chain` | free-form runs with no straight structure |
| `paint.fit_linear_gradient`, `paint.flat_colour` | recover the paint |
| `paint.fit_shared_ramp` | one ramp shared by duplicated shapes |

**Read [references/conventions.md](references/conventions.md) before writing
any hand measurement.** The two that cost the most hours:

- Scan rays must **start in background** and run into the shape. A ray
  starting inside the foreground returns `None`. A light counter *inside*
  dark ink is background too — its rays run outward from the interior.
- **Never trust a traced vertex.** Sample each edge, fit a line, and take
  vertices from the intersections.

Constrain before you fit: a round cap between parallel sides has radius
*exactly* the half-width, and a bowl meeting a flat edge is tangent to it.
When a constrained fit beats the free one, the constraint was the designer's.

## 3d. Paint

```python
fit_linear_gradient(rgb, region, trim=0.12)   # axis, stop positions, colours
flat_colour(rgb, region)                       # median, ignores overlays
```

Pass `trim` whenever anything is painted **on top of** the fill — a shadow
where a shape crosses itself, a glow, a watermark. Without it the overlay
drags the whole fit and the paint looks like something exotic; with it the
same paint reads as the plain two-stop ramp it is. Raise the trimmed pixels
as their own shape rather than pretending they belong to the gradient.

A **ring cannot carry a linear gradient** — a ramp fitted across one runs
from one side, through the hole in the middle, to the other. Rings want
their median colour, or a shape-following paint.

## 4. Model

Keep the model a pure function of the measurements — no hand-typed
coordinates — so that changing one measurement moves the whole model
consistently, and a rerun reproduces it exactly.

Schema, paint types, conic-gradient compilation and the winding rule for
counters: [references/model.md](references/model.md).

## 5. Iterate — the core loop

```bash
uv run --no-project "$SKILL/scripts/png2svg_cli.py" check work/<name> --label r1
uv run --no-project "$SKILL/scripts/png2svg_cli.py" residuals work/<name> --label r1
```

`check` writes `comparisons/r1/`: `reference.png`, `render.png`,
`overlay.png`, `difference.png` (4x gain), `deltaE.png` (CIEDE2000 heatmap),
`edge-difference.png` (red = reference boundary, green = render, white =
coincident) and `metrics.json`.

Read the numbers, then **look at the images**. `residuals` clusters colour
errors and edge misses into bboxes — fix the model where the clusters are,
bump the label, repeat.

Targets: IoU ≥ 0.995, edge mean ≲ 0.2px, edge max ≤ ~1.4px, ΔE2000 mean ≤
3.0, p95 ≤ 8.0. A well-converged simple logo reaches IoU ~0.997, edge mean
under 0.15, ΔE mean under 1.

On a textured source those colour targets are unreachable by anything —
switch to `deltaE_lowfreq_mean/p95` and accept an IoU nearer 0.93. Check
`texture_std` before concluding a reconstruction failed.

Know the noise floors (conventions.md) — resampling ringing, renderer
quarter-pixel quantisation and single-pixel edge quantisation are not yours
to fix. **Judge by p95, never by max.** Before rebuilding a model that
scores badly, confirm the fit is actually bad: measured residuals of a fifth
of a pixel alongside a poor score means the instrument, not the model.

**Keep the model small.** Prefer one measured arc over many fitted nodes. A
higher pixel score never justifies path soup: if a change adds nodes or
layers for less than 0.1 IoU, revert it.

## 6. Finish

```bash
uv run --no-project "$SKILL/scripts/png2svg_cli.py" validate work/<name>
uv run --no-project "$SKILL/scripts/png2svg_cli.py" export work/<name> -o out.svg
```

`validate` must pass everything. `alpha_mid_fraction` must **decrease**
across 1x / 4x / 16x — that is the operational definition of geometrically
sharp. Export refuses raster, script and external references.

If the artwork sits on a large artboard, also offer a version with the
viewBox cropped to the ink bounds. Same path data, different viewBox, and a
far more usable asset.

## Colour variants

```bash
CLI="$SKILL/scripts/png2svg_cli.py"
uv run --no-project "$CLI" recolor work/<name> -o work/<name>-alt --rotate 140
uv run --no-project "$CLI" recolor work/<name> -o work/<name>-alt --map "#294952=#4a2952,#94d49a=#d4a394"
uv run --no-project "$CLI" build work/<name>-alt
uv run --no-project "$CLI" export work/<name>-alt -o alt.svg
```

`--map` applies each colour's nearest anchor's Lab delta, so sampled gradient
stops move coherently with their anchor. Geometry is untouched and seams stay
seamless because the transform is uniform.

## Reference material

- [references/conventions.md](references/conventions.md) — the measurement
  rules and the noise floors. Read before measuring.
- [references/model.md](references/model.md) — `project.json` schema, paint
  types, conic wedge compilation, validation checks.
- [references/examples.md](references/examples.md) — two complete
  reconstructions with the reasoning that got there.
