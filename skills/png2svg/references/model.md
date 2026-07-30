# The model: `project.json`

The editable source of truth. SVG is generated from it deterministically:
byte-identical for an unchanged model, with fixed attribute order, fixed
float formatting and short internal ids from a reserved namespace.

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
`.view_box`, `.shapes` and `.notes`; build scripts should go through those
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
to the outer subpath, so the nonzero rule cuts it; or set
`"fill_rule": "evenodd"` on the shape, which cuts a hole regardless of the
subpaths' direction. `fill_rule` accepts `"nonzero"` (default, and the
attribute is then omitted) or `"evenodd"`.

Shape ids must be XML-safe: start with a letter, then use only letters,
digits, `.`, `_` and `-`. The semantic and animation export profiles use
these ids as stable DOM targets. Generated definition ids start with `_`, so
the two namespaces cannot collide.

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
the fills and deliberately **outside** any clip; clipping a stroke to its
own path would keep only the inner half of it. Its paint may be solid,
linear or radial; conic is rejected, since that compiles to filled wedges
which cannot follow a path.

Reach for this when a logo is genuinely constant width: a stroked path keeps
its width and colour as parameters and costs a handful of nodes, where the
outlined equivalent costs forty and can no longer be restyled. **Verify the
width first**; measure it at several points along every run. Widths that
differ run to run mean it is not a stroke, and forcing one will be wrong by
however much they differ.

Any fill may carry `"rect": [x, y, w, h]` to restrict that layer to a region
 -  useful when one shape carries different paint along different runs.

Coordinates are user-space; gradients are emitted with
`gradientUnits="userSpaceOnUse"`, so measure them in image pixels directly.

## Conic gradients

SVG has no conic primitive, so a `conic` paint compiles to a fan of wedge
polygons. The ramp stops are defined once, then each wedge references that
ramp through a chord-wise linear gradient at `r_mid = 0.66 * radius`. Extending
the chord's gradient coordinates maps the wedge endpoints to their correct
global ramp positions without repeating stops in every definition.

Things that matter:

- **Angles**: `0°` is `+x`; positive goes clockwise on screen (y grows
  downward). Extend the sweep 2-4° past a boundary that abuts other paint,
  or a sliver of background shows through.
- **Wedge count**: 48 for a ~180° sweep is the tested figure. Fewer banding,
  more bytes.
- **Every wedge is stroked with its own gradient** at `stroke-width="1.6"`.
  This is not decoration: two abutting antialiased edges composite to
  `1-(1-a)(1-b)` < 1, which shows as a lattice of seam lines at every wedge
  boundary (alpha dips to 186-230 without the stroke). Angular padding cannot
  fix it at all radii; stroking can.
- **A centre cover is required**: wedge vertices are degenerate at the centre
  and antialiasing leaves a pinhole. A pie spanning the fan's own angular
  range covers it without bleeding into neighbouring paint. A full-circle
  sweep (≥359.9°) needs a `<circle>` instead; an arc back to its own start
  point renders as nothing.
- **Matching seams**: where a conic meets a linear along a shared edge, sample
  both at the seam and set the shared endpoint colours equal.

The regression tests assert minimum alpha ≥250 along wedge boundaries and at
the fan centre. If you touch the fan geometry, run them.

## Validation and export

`validate` checks XML parses, no raster/script/foreignObject/external refs, a
viewBox is present, regeneration is byte-identical in every profile, node
counts, SVG byte/element counts, halo-freedom on white and black, and that
`alpha_mid_fraction` **decreases** across 1x / 4x / 16x renders. It exits
non-zero on failure.

`export` writes a standalone SVG and refuses raster, script or external
references. It has three deterministic profiles:

| profile | output |
|---|---|
| `compact` (default) | authoring ids and whitespace removed; H/V path commands where shorter |
| `semantic` | each logical shape id is kept on its path or wrapper |
| `animation` | every logical shape is wrapped in `<g id="shape-id">` |

`--tight --padding N` solves fill bounds exactly from path lines, Bézier
extrema and elliptical arc extrema. Stroke bounds use safe cap, join and
miter expansion. It overrides only the emitted viewBox and does not mutate
`project.json`.
