# png2svg

Rebuild a geometric PNG — a logo, icon or monogram — as clean, editable,
native SVG. Small paths and real gradients, no embedded raster, no path soup.

This is **reconstruction, not tracing**. Tools like vtracer and potrace fit
contours to pixels. png2svg recovers the shapes a designer would have drawn:
measure the source at subpixel precision, write a parametric model, render
it, score it against the reference, and iterate until the numbers converge.
The output of a good run is a handful of nodes, not a thousand.

On clean vector-style artwork: **IoU 0.998, edge error under 0.1px, 26–33
path nodes, sub-kilobyte exports.** On textured 3D renders it reconstructs
the structure and shading, drops the grain, and is scored accordingly.

| source | result | |
|---|---|---|
| plug "P" mark | IoU 0.9983, edge 0.092px, 33 nodes, 998 B | tangency and fillets |
| "N" monogram | IoU 0.9976, edge 0.101px, 26 nodes, 775 B | watermark discarded |
| gradient loop | IoU 0.9873, 108 nodes | counter, cap, shadow trimmed out of the paint |
| ribbon + cylinder | IoU 0.9740, 85 nodes | occlusion in three pieces |
| textured app icon | IoU 0.9264, low-freq ΔE 5.3 | structure kept, texture dropped |

Scripts for each are in `examples/`; the reasoning behind them is in
`skills/png2svg/references/examples.md`.

## Install as a skill

```bash
npx skills add iannuttall/png2svg
```

That's the whole install. The skill bundles its own engine, and the scripts
declare their dependencies inline — the first run builds a cached environment
via `uv` and nothing is added to your project.

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

Then just ask: *"vectorise this logo"*, pointing at a PNG or WebP.

## How it works

An agent drives the loop; the library does the measuring and scoring.
Neither half works alone — numbers cannot decide how a logo is built, and
judgment cannot hit 0.05px by eye.

**There is no single algorithm that reconstructs every logo.** Decomposition
— how many shapes, what covers what, which coincidences are real design
constraints — needs eyes on the image. What *is* deterministic is every
measurement the agent asks for, and every reconstruction is reproducible
byte-for-byte once written.

1. **Triage** — exact, "good enough", or decline?
2. **Decompose** — shapes, occlusion, seams *(agent)*
3. **Outline** — `subpixel_contour` → `segment_outline` → `snap_outline` →
   `to_segments`, four calls per shape
4. **Paint** — `fit_linear_gradient` / `flat_colour`
5. **Constrain** — the tangencies and shared radii no fitter can infer *(agent)*
6. **Check** — render, compare, read residual clusters, iterate
7. **Validate and export**

Constraints are proposed and then **verified against the measured
boundary**: `snap_outline` keeps the ones that fit better and reverts the
rest, because a wrong constraint moves geometry by tens of pixels. On one
mark it proposed 19 and rejected 8.

The editable source of truth is `project.json`; SVG is generated from it
byte-deterministically.

## Metrics

- `silhouette_iou` — foreground-mask intersection over union
- `edge_dist_mean/p95/max` — symmetric distance between mask boundaries (px)
- `deltaE_mean/p95/max` — CIEDE2000 over interior pixels ≥2.5px from edges,
  via linear-light RGB → Lab
- `mae_linear_rgb` — mean absolute error in linear RGB
- `texture_std` — how much the reference varies locally in its own interior:
  ~1-2 for clean vector art, 10+ for a textured render
- `deltaE_lowfreq_mean/p95` — CIEDE2000 after blurring both images, so the
  score reflects structure and shading rather than grain

When `texture_std` is high the source disagrees with **itself** by more than
any vector can match, and per-pixel ΔE will report failure for a
reconstruction that reads correctly. Judge those by the low-frequency
figures.

Masks use an adaptive ~50%-coverage contour (background distance normalised
by local full-strength foreground), so dark and light shapes get unbiased
boundaries. Edge metrics run on closed and hole-filled masks, because a
strong internal colour seam otherwise reads as a spurious boundary.

`check` renders at 4x and box-downsamples by default. resvg antialiases with
four coverage levels per axis, so a rasterised edge snaps to the nearest
quarter pixel — enough to flip a whole pixel row when a true edge sits near
the mask threshold, and enough to make a correct model look broken. On the P
mark, supersampling alone moved IoU 0.9938 → 0.9980 and edge mean 0.344 →
0.114 with no change to the model. Pass `--supersample 1` to see raw
renderer output.

## Repository layout

```
skills/png2svg/          the distributable skill — this is the product
  SKILL.md               the workflow
  references/            conventions, model schema, worked examples
  scripts/
    png2svg/             the engine (also the package this repo builds)
      measure.py         subpixel boundaries, line/circle/corner fits
      outline.py         contour -> lines/arcs/cubics, verified constraints
      curves.py          Bezier chain fitting
      paint.py           gradient and flat-colour recovery
      svggen.py          deterministic SVG generation
      compare.py         metrics, incl. texture_std and low-frequency deltaE
    png2svg_cli.py       zero-install entry point
    measure_template.py  copy-and-edit starting point
    build_template.py    split-file variant
examples/                real per-image measure/build scripts
tests/                   45 tests, including rendering regressions
```

The package lives inside the skill so the skill is self-contained; this
repo's `pyproject.toml` points at it there, so there is exactly one copy.

## Local development

```bash
uv sync
uv run pytest
uv run png2svg --help
```

Tests cover schema validation, deterministic generation, raster/script
absence, solid/linear/conic rendering against ground truth (including the
wedge-seam and centre-pinhole regressions), geometry helpers, and metric
behaviour on synthetic shifted and recoloured fixtures.

## What it can and cannot do

**Suitable**: flat or gradient fills, crisp edges, shapes decomposable into
lines, arcs and Béziers. If it could have been built in Figma from shapes and
gradients, it is recoverable — including layered overlaps, conic sweeps and
squircle-smoothed corners.

**Good enough**: textured 3D renders, brushed metal, grain, drop shadows and
glow. Structure and large-scale shading reconstruct; texture is dropped
deliberately and the result is judged on IoU and low-frequency ΔE. Refusing
these outright would be the wrong call — they reconstruct usefully, just not
exactly.

**Not suitable**: photographs, painterly artwork, anything under ~100px.
Use vtracer or potrace there.

## Known limitations

- Conic gradients compile to wedge fans (48 wedges per cap), which costs
  bytes. A future exporter could emit CSS `conic-gradient` for HTML.
- There is no fully automatic one-shot `reconstruct` command, and there will
  not be: decomposition needs judgment. The library makes every measurement
  deterministic; the agent writes the per-image script.
- `validate`'s halo checks assume artwork sits on transparency, so they
  false-fail on full-canvas designs.
- No interactive editor UI — the agent loop plays that role.

## License

MIT
