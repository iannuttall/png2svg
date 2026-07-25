"""Project schema: the editable source of truth lives in project.json.

The model is deliberately plain JSON (lists/dicts) validated lightly here,
so both humans and AI agents can edit it without touching SVG markup.

Shapes
------
{"id": str, "type": "path", "d": [segment, ...], "fills": [paint, ...]}

Segments are JSON forms of SVG path commands (absolute only):
  ["M", x, y]
  ["L", x, y]
  ["A", rx, ry, x_rot, large_arc, sweep, x, y]
  ["C", x1, y1, x2, y2, x, y]
  ["Q", x1, y1, x, y]
  ["Z"]

Paints (a shape's `fills` is a stack, painted bottom-up, clipped to the shape)
------
{"type": "solid",  "color": "#rrggbb", "opacity": 1.0}
{"type": "linear", "x1": .., "y1": .., "x2": .., "y2": ..,
 "stops": [{"offset": 0.0, "color": "#rrggbb", "opacity": 1.0}, ...]}
{"type": "radial", "cx": .., "cy": .., "r": ..,
 "fx": .., "fy": ..,            # optional focal point
 "stops": [...]}
{"type": "conic",  "cx": .., "cy": .., "radius": ..,
 "angle_start": deg, "angle_end": deg,   # sweep range, 0deg = +x axis, CCW in
                                          # image coords means visually clockwise
 "stops": [{"offset": 0..1, "color": ...}],  # offset maps into the angle range
 "wedges": 16, "opacity": 1.0}
  -> compiled to clipped wedge polygons with per-wedge linear gradients,
     because native SVG has no conic gradient.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEGMENT_ARITY = {"M": 2, "L": 2, "A": 7, "C": 6, "Q": 4, "Z": 0}
PAINT_TYPES = {"solid", "linear", "radial", "conic"}


class ModelError(ValueError):
    pass


def validate_shape(shape: dict[str, Any]) -> None:
    if shape.get("type") != "path":
        raise ModelError(f"shape {shape.get('id')!r}: only type 'path' is supported")
    d = shape.get("d")
    if not isinstance(d, list) or not d:
        raise ModelError(f"shape {shape.get('id')!r}: 'd' must be a non-empty list")
    if d[0][0] != "M":
        raise ModelError(f"shape {shape.get('id')!r}: path must start with M")
    for seg in d:
        cmd = seg[0]
        if cmd not in SEGMENT_ARITY:
            raise ModelError(f"shape {shape.get('id')!r}: unknown command {cmd!r}")
        if len(seg) - 1 != SEGMENT_ARITY[cmd]:
            raise ModelError(
                f"shape {shape.get('id')!r}: {cmd} expects "
                f"{SEGMENT_ARITY[cmd]} numbers, got {len(seg) - 1}"
            )
    rule = shape.get("fill_rule", "nonzero")
    if rule not in ("nonzero", "evenodd"):
        raise ModelError(
            f"shape {shape.get('id')!r}: fill_rule must be 'nonzero' or 'evenodd'"
        )
    fills = shape.get("fills")
    stroke = shape.get("stroke")
    if not isinstance(fills, list):
        raise ModelError(f"shape {shape.get('id')!r}: 'fills' must be a list")
    if not fills and stroke is None:
        raise ModelError(
            f"shape {shape.get('id')!r}: 'fills' may only be empty on a stroked shape"
        )
    for paint in fills:
        validate_paint(shape.get("id"), paint)
    if stroke is not None:
        validate_stroke(shape.get("id"), stroke)


LINECAPS = {"butt", "round", "square"}
LINEJOINS = {"miter", "round", "bevel"}


def validate_stroke(owner: Any, stroke: dict[str, Any]) -> None:
    """A stroked outline: paint + width, optional cap/join style.

    The stroke is painted after the fills and is deliberately NOT clipped to
    the shape — clipping a stroke to its own path would keep only the inner
    half of it.
    """
    if not isinstance(stroke, dict):
        raise ModelError(f"shape {owner!r}: 'stroke' must be an object")
    for key in ("paint", "width"):
        if key not in stroke:
            raise ModelError(f"shape {owner!r}: stroke missing {key!r}")
    if float(stroke["width"]) <= 0:
        raise ModelError(f"shape {owner!r}: stroke width must be positive")
    paint = stroke["paint"]
    if paint.get("type") == "conic":
        raise ModelError(
            f"shape {owner!r}: conic stroke is not supported (it compiles to "
            f"filled wedges, which cannot be stroked along a path)"
        )
    validate_paint(owner, paint)
    if "linecap" in stroke and stroke["linecap"] not in LINECAPS:
        raise ModelError(f"shape {owner!r}: linecap must be one of {sorted(LINECAPS)}")
    if "linejoin" in stroke and stroke["linejoin"] not in LINEJOINS:
        raise ModelError(f"shape {owner!r}: linejoin must be one of {sorted(LINEJOINS)}")


def validate_paint(owner: Any, paint: dict[str, Any]) -> None:
    ptype = paint.get("type")
    if ptype not in PAINT_TYPES:
        raise ModelError(f"shape {owner!r}: unknown paint type {ptype!r}")
    if ptype == "solid":
        _require(owner, paint, "color")
    elif ptype == "linear":
        _require(owner, paint, "x1", "y1", "x2", "y2", "stops")
    elif ptype == "radial":
        _require(owner, paint, "cx", "cy", "r", "stops")
    elif ptype == "conic":
        _require(owner, paint, "cx", "cy", "radius", "angle_start", "angle_end", "stops")
    for stop in paint.get("stops", []):
        if not (0.0 <= float(stop["offset"]) <= 1.0):
            raise ModelError(f"shape {owner!r}: stop offset out of [0,1]")


def _require(owner: Any, paint: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in paint:
            raise ModelError(f"shape {owner!r}: paint {paint['type']!r} missing {key!r}")


@dataclass
class Project:
    source_path: str
    width: int
    height: int
    sha256: str
    background: list[int]  # RGBA of the page behind the artwork
    view_box: list[float] = field(default_factory=list)
    shapes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        ids = [s.get("id") for s in self.shapes]
        if len(ids) != len(set(ids)):
            raise ModelError("duplicate shape ids")
        for shape in self.shapes:
            validate_shape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "source": {
                "path": self.source_path,
                "width": self.width,
                "height": self.height,
                "sha256": self.sha256,
                "background": self.background,
            },
            "model": {
                "viewBox": self.view_box or [0, 0, self.width, self.height],
                "shapes": self.shapes,
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        src = data["source"]
        model = data["model"]
        return cls(
            source_path=src["path"],
            width=src["width"],
            height=src["height"],
            sha256=src["sha256"],
            background=list(src["background"]),
            view_box=list(model.get("viewBox", [])),
            shapes=model.get("shapes", []),
            notes=data.get("notes", []),
        )


def load_project(project_dir: Path) -> Project:
    data = json.loads((project_dir / "project.json").read_text())
    project = Project.from_dict(data)
    project.validate()
    return project


def save_project(project_dir: Path, project: Project) -> None:
    project.validate()
    path = project_dir / "project.json"
    path.write_text(json.dumps(project.to_dict(), indent=2) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
