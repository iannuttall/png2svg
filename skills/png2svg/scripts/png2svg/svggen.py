"""Deterministic SVG generation from the project model.

Output is stable byte-for-byte for an unchanged model: fixed attribute
order, fixed float formatting and short internal ids from a reserved
namespace. Semantic shape ids stay stable DOM targets. Conic paints are
compiled to clipped wedge fans of linear gradients since SVG has no native
conic gradient.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .geom import path_bounds
from .model import Project

PROFILES = ("semantic", "compact", "animation")


def fmt(x: float) -> str:
    """Format a number with 3 decimals, trimming trailing zeros."""
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_data(segments: list[list[Any]], compact: bool = False) -> str:
    parts = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for seg in segments:
        cmd = seg[0]
        if cmd == "A":
            rx, ry, rot, laf, sf, x, y = seg[1:]
            parts.append(
                f"A{fmt(rx)} {fmt(ry)} {fmt(rot)} {int(laf)} {int(sf)} {fmt(x)} {fmt(y)}"
            )
            current = (float(x), float(y))
        elif cmd == "Z":
            parts.append("Z")
            current = start
        else:
            values = [float(v) for v in seg[1:]]
            if compact and cmd == "L" and current is not None:
                if abs(values[1] - current[1]) < 1e-12:
                    parts.append("H" + fmt(values[0]))
                elif abs(values[0] - current[0]) < 1e-12:
                    parts.append("V" + fmt(values[1]))
                else:
                    parts.append("L" + " ".join(fmt(v) for v in values))
            else:
                parts.append(cmd + " ".join(fmt(v) for v in values))
            if cmd == "M":
                current = (values[0], values[1])
                start = current
            elif cmd == "L":
                current = (values[0], values[1])
            elif cmd == "Q":
                current = (values[2], values[3])
            elif cmd == "C":
                current = (values[4], values[5])
    return " ".join(parts)


def _stops_svg(stops: list[dict[str, Any]], indent: str) -> str:
    out = []
    for stop in stops:
        opacity = stop.get("opacity", 1.0)
        extra = f' stop-opacity="{fmt(opacity)}"' if opacity != 1.0 else ""
        out.append(
            f'{indent}<stop offset="{fmt(stop["offset"])}" '
            f'stop-color="{stop["color"]}"{extra}/>'
        )
    return "\n".join(out)


def _interp_color(stops: list[dict[str, Any]], t: float) -> tuple[str, float]:
    """Linearly interpolate stop colour/opacity (sRGB) at offset t."""
    stops = sorted(stops, key=lambda s: float(s["offset"]))
    if t <= float(stops[0]["offset"]):
        return stops[0]["color"], float(stops[0].get("opacity", 1.0))
    if t >= float(stops[-1]["offset"]):
        return stops[-1]["color"], float(stops[-1].get("opacity", 1.0))
    for a, b in zip(stops, stops[1:]):
        oa, ob = float(a["offset"]), float(b["offset"])
        if oa <= t <= ob:
            f = 0.0 if ob == oa else (t - oa) / (ob - oa)
            ca = _hex_rgb(a["color"])
            cb = _hex_rgb(b["color"])
            rgb = tuple(round(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
            op = float(a.get("opacity", 1.0))
            op += (float(b.get("opacity", 1.0)) - op) * f
            return "#{:02x}{:02x}{:02x}".format(*rgb), op
    raise AssertionError("unreachable")


def _hex_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _wedge_path(cx: float, cy: float, radius: float, a0: float, a1: float) -> str:
    """Pie-slice path from angle a0 to a1 (radians), slightly overscanned."""
    r = radius * 1.02  # overscan so wedge seams sit outside the clip
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    large = 1 if abs(a1 - a0) > math.pi else 0
    sweep = 1 if a1 > a0 else 0
    return (
        f"M{fmt(cx)} {fmt(cy)} L{fmt(x0)} {fmt(y0)} "
        f"A{fmt(r)} {fmt(r)} 0 {large} {sweep} {fmt(x1)} {fmt(y1)} Z"
    )


def _conic_svg(pid: str, paint: dict[str, Any], defs: list[str], body: list[str]) -> None:
    """Compile a conic paint into wedge polygons with linear gradients."""
    cx, cy = float(paint["cx"]), float(paint["cy"])
    radius = float(paint["radius"])
    a_start = math.radians(float(paint["angle_start"]))
    a_end = math.radians(float(paint["angle_end"]))
    n = int(paint.get("wedges", 16))
    opacity = float(paint.get("opacity", 1.0))
    op_attr = f' opacity="{fmt(opacity)}"' if opacity != 1.0 else ""
    base_id = f"{pid}r"
    defs.append(
        f'<linearGradient id="{base_id}">\n'
        + _stops_svg(paint["stops"], "  ")
        + "\n</linearGradient>"
    )
    body.append(f"<g{op_attr}>")
    # small pie under the fan: the wedge vertices are degenerate at the
    # centre and antialiasing can leave a pinhole there; the pie spans only
    # the fan's own angular range so it cannot bleed into neighbour paint
    # (a full-circle sweep needs a circle — an arc to its own start point
    # renders as nothing)
    c_mid, _ = _interp_color(paint["stops"], 0.5)
    if abs(a_end - a_start) >= math.radians(359.9):
        body.append(f'  <circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="4" fill="{c_mid}"/>')
    else:
        body.append(
            f'  <path d="{_wedge_path(cx, cy, 4.0 / 1.02, a_start, a_end)}" '
            f'fill="{c_mid}"/>'
        )
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        a0 = a_start + (a_end - a_start) * t0
        a1 = a_start + (a_end - a_start) * t1
        # Gradient runs chord-wise across the wedge between its edge midpoints,
        # so colour varies with angle, approximating the conic sweep.
        r_mid = radius * 0.66
        gx0, gy0 = cx + r_mid * math.cos(a0), cy + r_mid * math.sin(a0)
        gx1, gy1 = cx + r_mid * math.cos(a1), cy + r_mid * math.sin(a1)
        # Reuse one full ramp for every wedge. Extending the wedge chord so
        # its endpoints land at global ramp positions t0/t1 gives the exact
        # local slice without repeating two colour stops in every definition.
        span = t1 - t0
        dx, dy = (gx1 - gx0) / span, (gy1 - gy0) / span
        full_x0, full_y0 = gx0 - dx * t0, gy0 - dy * t0
        full_x1, full_y1 = full_x0 + dx, full_y0 + dy
        gid = f"{pid}w{i:x}"
        defs.append(
            f'<linearGradient id="{gid}" href="#{base_id}" '
            f'gradientUnits="userSpaceOnUse" x1="{fmt(full_x0)}" '
            f'y1="{fmt(full_y0)}" x2="{fmt(full_x1)}" y2="{fmt(full_y1)}"/>'
        )
        # Stroke each wedge with its own gradient: two abutting antialiased
        # edges composite to less than full alpha (1-(1-a)(1-b)), so unstroked
        # fans show faint seam lines at every wedge boundary.
        body.append(
            f'  <path d="{_wedge_path(cx, cy, radius, a0, a1)}" '
            f'fill="url(#{gid})" stroke="url(#{gid})" stroke-width="1.6"/>'
        )
    body.append("</g>")


def _gradient_def(pid: str, paint: dict[str, Any]) -> str:
    """The <linearGradient>/<radialGradient> definition for a paint."""
    if paint["type"] == "linear":
        return (
            f'<linearGradient id="{pid}" gradientUnits="userSpaceOnUse" '
            f'x1="{fmt(paint["x1"])}" y1="{fmt(paint["y1"])}" '
            f'x2="{fmt(paint["x2"])}" y2="{fmt(paint["y2"])}">\n'
            + _stops_svg(paint["stops"], "  ")
            + "\n</linearGradient>"
        )
    focal = ""
    if "fx" in paint:
        focal = f' fx="{fmt(paint["fx"])}" fy="{fmt(paint["fy"])}"'
    return (
        f'<radialGradient id="{pid}" gradientUnits="userSpaceOnUse" '
        f'cx="{fmt(paint["cx"])}" cy="{fmt(paint["cy"])}" '
        f'r="{fmt(paint["r"])}"{focal}>\n'
        + _stops_svg(paint["stops"], "  ")
        + "\n</radialGradient>"
    )


def _stroke_svg(
    gradient_id: str, d: str, stroke: dict[str, Any], defs: list[str]
) -> str:
    """A stroked outline, painted after the fills and outside any clip."""
    paint = stroke["paint"]
    if paint["type"] == "solid":
        value = paint["color"]
    else:
        defs.append(_gradient_def(gradient_id, paint))
        value = f"url(#{gradient_id})"
    attrs = [f'stroke="{value}"', f'stroke-width="{fmt(stroke["width"])}"']
    if "linecap" in stroke:
        attrs.append(f'stroke-linecap="{stroke["linecap"]}"')
    if "linejoin" in stroke:
        attrs.append(f'stroke-linejoin="{stroke["linejoin"]}"')
    opacity = float(stroke.get("opacity", 1.0))
    if opacity != 1.0:
        attrs.append(f'stroke-opacity="{fmt(opacity)}"')
    return f'<path d="{d}" fill="none" {" ".join(attrs)}/>'


def _layer_target(
    d: str,
    fill: str,
    op_attr: str,
    li: int,
    paint: dict,
    vb: list,
    fill_rule: str,
) -> str:
    """The element a fill layer paints: the shape path for the base layer,
    a bounded rect for overlays (clipped to the shape by the enclosing group).
    A paint's optional "rect": [x, y, w, h] restricts the layer's region."""
    if "rect" in paint:
        x, y, w, h = paint["rect"]
    elif li == 0:
        return f'  <path d="{d}" fill="{fill}"{fill_rule}{op_attr}/>'
    else:
        x, y, w, h = vb
    return (
        f'  <rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'fill="{fill}"{op_attr}/>'
    )


def tight_view_box(project: Project, padding: float = 0.0) -> list[float]:
    """Bounds of all model paths as `[x, y, width, height]`."""
    if not project.shapes:
        raise ValueError("cannot crop a project with no shapes")
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for shape in project.shapes:
        stroke = shape.get("stroke")
        stroke_width = 0.0
        if stroke is not None:
            width = float(stroke["width"])
            # SVG's default miterlimit can extend a sharp join to four
            # half-widths. Round and bevel joins need only half a width.
            stroke_width = width * 4.0 if stroke.get("linejoin", "miter") == "miter" else width
        bx0, by0, bx1, by1 = path_bounds(shape["d"], stroke_width)
        x0, y0 = min(x0, bx0), min(y0, by0)
        x1, y1 = max(x1, bx1), max(y1, by1)
    pad = max(0.0, float(padding))
    return [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad]


def svg_stats(svg: str) -> dict[str, int]:
    """Small deterministic complexity report for an emitted SVG."""
    return {
        "bytes": len(svg.encode()),
        "elements": len(re.findall(r"<(?!/|!)[A-Za-z]", svg)),
        "paths": svg.count("<path"),
        "groups": svg.count("<g"),
        "defs": svg.count("<defs"),
        "gradients": svg.count("<linearGradient") + svg.count("<radialGradient"),
    }


def generate_svg(
    project: Project,
    *,
    profile: str = "semantic",
    view_box: list[float] | None = None,
) -> str:
    """Generate stable SVG for editing, compact delivery, or animation.

    `semantic` puts each logical shape id on its path or wrapper. `animation`
    always gives every logical shape a stable group. `compact` removes those
    authoring ids and whitespace while retaining ids required by SVG defs.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}")
    project.validate()
    vb = view_box or project.view_box or [0, 0, project.width, project.height]
    defs: list[str] = []
    body: list[str] = []
    gradient_serial = 0

    for shape_index, shape in enumerate(project.shapes):
        sid = shape["id"]
        d = path_data(shape["d"], compact=profile == "compact")
        # evenodd lets a compound path carry a real hole without relying on
        # subpath winding — the only way to cut a shape that stays a hole
        # when the artwork is recoloured or composited on anything else
        rule = shape.get("fill_rule", "nonzero")
        fr = f' fill-rule="{rule}"' if rule != "nonzero" else ""
        clip_needed = len(shape["fills"]) > 1 or any(
            f["type"] == "conic" for f in shape["fills"]
        )
        outer_group = profile == "animation" or (
            profile == "semantic" and (clip_needed or shape.get("stroke") is not None)
        )
        if outer_group:
            body.append(f'<g id="{sid}">')
        if clip_needed:
            clip_id = f"_c{shape_index:x}"
            clip_rule = f' clip-rule="{rule}"' if rule != "nonzero" else ""
            defs.append(
                f'<clipPath id="{clip_id}">\n'
                f'  <path d="{d}"{clip_rule}/>\n</clipPath>'
            )
            body.append(f'<g clip-path="url(#{clip_id})">')
        for li, paint in enumerate(shape["fills"]):
            ptype = paint["type"]
            if ptype == "solid":
                pid = ""
            else:
                pid = f"_g{gradient_serial:x}"
                gradient_serial += 1
            opacity = float(paint.get("opacity", 1.0))
            op_attr = f' fill-opacity="{fmt(opacity)}"' if opacity != 1.0 else ""
            if ptype == "solid":
                fill = paint["color"]
                if clip_needed:
                    body.append(_layer_target(d, fill, op_attr, li, paint, vb, fr))
                else:
                    id_attr = (
                        f' id="{sid}"'
                        if profile == "semantic" and not outer_group
                        else ""
                    )
                    body.append(f'<path{id_attr} d="{d}" fill="{fill}"{fr}{op_attr}/>')
            elif ptype in ("linear", "radial"):
                defs.append(_gradient_def(pid, paint))
                fill = f"url(#{pid})"
                if clip_needed:
                    body.append(_layer_target(d, fill, op_attr, li, paint, vb, fr))
                else:
                    id_attr = (
                        f' id="{sid}"'
                        if profile == "semantic" and not outer_group
                        else ""
                    )
                    body.append(f'<path{id_attr} d="{d}" fill="{fill}"{fr}{op_attr}/>')
            elif ptype == "conic":
                _conic_svg(pid, paint, defs, body)
        if clip_needed:
            body.append("</g>")
        if shape.get("stroke"):
            stroke_gradient_id = ""
            if shape["stroke"]["paint"]["type"] != "solid":
                stroke_gradient_id = f"_g{gradient_serial:x}"
                gradient_serial += 1
            body.append(
                _stroke_svg(stroke_gradient_id, d, shape["stroke"], defs)
            )
        if outer_group:
            body.append("</g>")

    root = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{fmt(vb[0])} {fmt(vb[1])} {fmt(vb[2])} {fmt(vb[3])}"'
    )
    if profile != "compact":
        root += f' width="{fmt(vb[2])}" height="{fmt(vb[3])}"'
    lines = [root + ">"]
    if defs:
        lines.append("<defs>")
        lines.extend(defs)
        lines.append("</defs>")
    lines.extend(body)
    lines.append("</svg>")
    svg = "\n".join(lines) + "\n"
    if profile == "compact":
        svg = re.sub(r">\s+<", "><", svg).strip() + "\n"
        svg = re.sub(
            r"#([0-9a-fA-F])\1([0-9a-fA-F])\2([0-9a-fA-F])\3(?=\")",
            lambda match: "#" + match.group(1) + match.group(2) + match.group(3),
            svg,
        )
        svg = re.sub(r'([= "\'])0\.', r"\1.", svg)
        svg = re.sub(r'([= "\'])-0\.', r"\1-.", svg)
    return svg
