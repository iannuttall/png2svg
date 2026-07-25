# The model: `project.json`

The editable source of truth. SVG is generated from it deterministically —
byte-identical for an unchanged model, with fixed attribute order, fixed
float formatting and ids derived from shape id + layer index.

```json
{
  "version": 1,
  "source": {
    "path": "source/logo.png",
    "width": 498,
    "height": 524,
    "sha256": "d798...",
    "background": [251, 247, 245, 255]
  },
  "model": {
    "viewBox": [0, 0, 498, 524],
    "shapes": [ ... ]
  },
  "notes": ["free-form; record what you decided and why"]
}
```

`init` writes everything except `model.shapes` and your notes. Load and save
it with `png2svg.model.load_project` / `save_project`, which flatten this
into a `Project` with `.source_path`, `.width`, `.height`, `.background`,
`.view_box`, `.shapes` and `.notes` — build scripts should go through those
rather than touching the JSON directly.

## Shapes

```json
{
  "id": "monogram",
  "type": "path",
  "d": [["M", 10, 10], ["L", 90, 10], ["A", 40, 40, 0, 0, 1, 90, 90], ["Z"]],
  "fills": [ ... ]
}
```

Segments are JSON arrays, absolute coordinates only:

| Segment | Arity | Meaning |
|---|---|---|
| `["M", x, y]` | 2 | move to (starts a subpath) |
| `["L", x, y]` | 2 | line to |
| `["A", rx, ry, rot, laf, sf, x, y]` | 7 | elliptical arc |
| `["C", x1, y1, x2, y2, x, y]` | 6 | cubic Bézier |
| `["Q", x1, y1, x, y]` | 4 | quadratic Bézier |
| `["Z"]` | 0 | close subpath |

Multiple subpaths live in one `d`. A counter (hole) must be wound **opposite**
to the outer subpath — there is no `fill-rule` support, so the nonzero rule
is what cuts the hole.

## Fills

`fills` is a stack painted bottom-up, each layer clipped to the shape. Most
shapes have exactly one.

```json
{"type": "solid",  "color": "#ff9529", "opacity": 1.0}

{"type": "linear", "x1": .., "y1": .., "x2": .., "y2": ..,
 "stops": [{"offset": 0.0, "color": "#ff9529", "opacity": 1.0}, ...]}

{"type": "radial", "cx": .., "cy": .., "r": ..,
 "fx": .., "fy": ..,                       // optional focal point
 "stops": [...]}

{"type": "conic",  "cx": .., "cy": .., "radius": ..,
 "angle_start": -88, "angle_end": -272,    // degrees
 "stops": [...], "wedges": 48, "opacity": 1.0}
```

## Stroke

A shape may carry a `stroke` alongside (or instead of) its fills:

```json
{
  "id": "path", "type": "path", "d": [...],
  "fills": [],
  "stroke": {
    "paint": {"type": "linear", "x1": .., "y1": .., "x2": .., "y2": ..,
              "stops": [...]},
    "width": 60.0, "linecap": "round", "linejoin": "round", "opacity": 1.0
  }
}
```

`fills` may be empty only on a stroked shape. The stroke is painted after
the fills and deliberately **outside** any clip — clipping a stroke to its
own path would keep only the inner half of it. Its paint may be solid,
linear or radial; conic is rejected, since that compiles to filled wedges
which cannot follow a path.

Reach for this when a logo is genuinely constant width: a stroked path keeps
its width and colour as parameters and costs a handful of nodes, where the
outlined equivalent costs forty and can no longer be restyled. **Verify the
width first** — measure it at several points along every run. Widths that
differ run to run mean it is not a stroke, and forcing one will be wrong by
however much they differ.

Any fill may carry `"rect": [x, y, w, h]` to restrict that layer to a region
— useful when one shape carries different paint along different runs.

Coordinates are user-space; gradients are emitted with
`gradientUnits="userSpaceOnUse"`, so measure them in image pixels directly.

## Conic gradients

SVG has no conic primitive, so a `conic` paint compiles to a fan of wedge
polygons, each with its own two-stop linear gradient laid chord-wise between
its edge midpoints at `r_mid = 0.66 * radius`.

Things that matter:

- **Angles**: `0°` is `+x`; positive goes clockwise on screen (y grows
  downward). Extend the sweep 2–4° past a boundary that abuts other paint,
  or a sliver of background shows through.
- **Wedge count**: 48 for a ~180° sweep is the tested figure. Fewer banding,
  more bytes.
- **Every wedge is stroked with its own gradient** at `stroke-width="1.6"`.
  This is not decoration: two abutting antialiased edges composite to
  `1-(1-a)(1-b)` < 1, which shows as a lattice of seam lines at every wedge
  boundary (alpha dips to 186–230 without the stroke). Angular padding cannot
  fix it at all radii; stroking can.
- **A centre cover is required**: wedge vertices are degenerate at the centre
  and antialiasing leaves a pinhole. A pie spanning the fan's own angular
  range covers it without bleeding into neighbouring paint. A full-circle
  sweep (≥359.9°) needs a `<circle>` instead — an arc back to its own start
  point renders as nothing.
- **Seamless flow**: where a conic meets a linear along a shared edge, sample
  both at the seam and set the shared endpoint colours equal.

The regression tests assert minimum alpha ≥250 along wedge boundaries and at
the fan centre. If you touch the fan geometry, run them.

## Validation and export

`validate` checks XML parses, no raster/script/foreignObject/external refs, a
viewBox is present, regeneration is byte-identical, node counts, halo-freedom
on white and black, and that `alpha_mid_fraction` **decreases** across 1x /
4x / 16x renders — that is the operational definition of "geometrically
sharp". It exits non-zero on failure.

`export` writes a standalone SVG and refuses raster or script content.
