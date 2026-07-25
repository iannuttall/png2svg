"""Deterministic SVG generation from the project model.

Output is stable byte-for-byte for an unchanged model: fixed attribute
order, fixed float formatting, ids derived from shape ids and layer index.
Conic paints are compiled to clipped wedge fans of linear gradients since
SVG has no native conic gradient.
"""

from __future__ import annotations

import math
from typing import Any

from .model import Project


def fmt(x: float) -> str:
    """Format a number with 3 decimals, trimming trailing zeros."""
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_data(segments: list[list[Any]]) -> str:
    parts = []
    for seg in segments:
        cmd = seg[0]
        if cmd == "A":
            rx, ry, rot, laf, sf, x, y = seg[1:]
            parts.append(
                f"A{fmt(rx)} {fmt(ry)} {fmt(rot)} {int(laf)} {int(sf)} {fmt(x)} {fmt(y)}"
            )
        elif cmd == "Z":
            parts.append("Z")
        else:
            parts.append(cmd + " ".join(fmt(v) for v in seg[1:]))
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
        c0, o0 = _interp_color(paint["stops"], t0)
        c1, o1 = _interp_color(paint["stops"], t1)
        gid = f"{pid}-w{i}"
        defs.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'x1="{fmt(gx0)}" y1="{fmt(gy0)}" x2="{fmt(gx1)}" y2="{fmt(gy1)}">\n'
            f'  <stop offset="0" stop-color="{c0}"'
            + (f' stop-opacity="{fmt(o0)}"' if o0 != 1.0 else "")
            + '/>\n'
            f'  <stop offset="1" stop-color="{c1}"'
            + (f' stop-opacity="{fmt(o1)}"' if o1 != 1.0 else "")
            + "/>\n</linearGradient>"
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


def _stroke_svg(sid: str, d: str, stroke: dict[str, Any], defs: list[str]) -> str:
    """A stroked outline, painted after the fills and outside any clip."""
    paint = stroke["paint"]
    if paint["type"] == "solid":
        value = paint["color"]
    else:
        pid = f"{sid}-s"
        defs.append(_gradient_def(pid, paint))
        value = f"url(#{pid})"
    attrs = [f'stroke="{value}"', f'stroke-width="{fmt(stroke["width"])}"']
    if "linecap" in stroke:
        attrs.append(f'stroke-linecap="{stroke["linecap"]}"')
    if "linejoin" in stroke:
        attrs.append(f'stroke-linejoin="{stroke["linejoin"]}"')
    opacity = float(stroke.get("opacity", 1.0))
    if opacity != 1.0:
        attrs.append(f'stroke-opacity="{fmt(opacity)}"')
    return f'<path d="{d}" fill="none" {" ".join(attrs)}/>'


def _layer_target(d: str, fill: str, op_attr: str, li: int, paint: dict, vb: list) -> str:
    """The element a fill layer paints: the shape path for the base layer,
    a bounded rect for overlays (clipped to the shape by the enclosing group).
    A paint's optional "rect": [x, y, w, h] restricts the layer's region."""
    if "rect" in paint:
        x, y, w, h = paint["rect"]
    elif li == 0:
        return f'  <path d="{d}" fill="{fill}"{op_attr}/>'
    else:
        x, y, w, h = vb
    return (
        f'  <rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'fill="{fill}"{op_attr}/>'
    )


def generate_svg(project: Project) -> str:
    vb = project.view_box or [0, 0, project.width, project.height]
    defs: list[str] = []
    body: list[str] = []

    for shape in project.shapes:
        sid = shape["id"]
        d = path_data(shape["d"])
        clip_needed = len(shape["fills"]) > 1 or any(
            f["type"] == "conic" for f in shape["fills"]
        )
        if clip_needed:
            defs.append(f'<clipPath id="clip-{sid}">\n  <path d="{d}"/>\n</clipPath>')
            body.append(f'<g clip-path="url(#clip-{sid})">')
        for li, paint in enumerate(shape["fills"]):
            pid = f"{sid}-p{li}"
            ptype = paint["type"]
            opacity = float(paint.get("opacity", 1.0))
            op_attr = f' fill-opacity="{fmt(opacity)}"' if opacity != 1.0 else ""
            if ptype == "solid":
                fill = paint["color"]
                if clip_needed:
                    body.append(_layer_target(d, fill, op_attr, li, paint, vb))
                else:
                    body.append(f'<path d="{d}" fill="{fill}"{op_attr}/>')
            elif ptype in ("linear", "radial"):
                defs.append(_gradient_def(pid, paint))
                fill = f"url(#{pid})"
                if clip_needed:
                    body.append(_layer_target(d, fill, op_attr, li, paint, vb))
                else:
                    body.append(f'<path d="{d}" fill="{fill}"{op_attr}/>')
            elif ptype == "conic":
                _conic_svg(pid, paint, defs, body)
        if clip_needed:
            body.append("</g>")
        if shape.get("stroke"):
            body.append(_stroke_svg(sid, d, shape["stroke"], defs))

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{fmt(vb[0])} {fmt(vb[1])} {fmt(vb[2])} {fmt(vb[3])}" '
        f'width="{fmt(vb[2])}" height="{fmt(vb[3])}">'
    ]
    if defs:
        lines.append("<defs>")
        lines.extend(defs)
        lines.append("</defs>")
    lines.extend(body)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
