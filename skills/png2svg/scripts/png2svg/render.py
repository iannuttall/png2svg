"""Rasterise SVG via resvg (resvg-py binding) at exact pixel dimensions."""

from __future__ import annotations

import io
from importlib.metadata import version

import resvg_py
from PIL import Image

RENDERER = f"resvg-py {version('resvg-py')}"

# We only ever open bytes produced by resvg itself; large 16x validation
# renders legitimately exceed PIL's decompression-bomb heuristic.
Image.MAX_IMAGE_PIXELS = None


def render_svg(
    svg_text: str, width: int, height: int, supersample: int = 1
) -> Image.Image:
    """Render SVG to an RGBA PIL image at the given pixel size.

    resvg antialiases with 4 coverage levels per axis, so a rasterised edge
    lands on the nearest quarter pixel: asking for an edge at y=60.394 puts
    it at 60.50. That ±0.125px quantisation is invisible in most uses but
    dominates subpixel comparison, and it flips an entire row whenever a
    true edge sits near a mask threshold.

    `supersample` renders at N times the size and box-downsamples, giving
    4N coverage levels. N=4 drops the quantisation to ±0.031px, far below
    the measurement floor, so comparison scores the model rather than the
    renderer. Leave it at 1 when the actual rasteriser output is the
    subject (validate's sharpness and halo checks).
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    w, h = width * supersample, height * supersample
    data = resvg_py.svg_to_bytes(svg_string=svg_text, width=w, height=h)
    img = Image.open(io.BytesIO(bytes(data)))
    img = img.convert("RGBA")
    if img.size != (w, h):
        raise RuntimeError(f"renderer produced {img.size}, expected {(w, h)}")
    if supersample > 1:
        img = img.resize((width, height), Image.BOX)
    return img
