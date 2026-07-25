"""Localise reconstruction residuals: where is the render still wrong, and how.

Turns global metrics into actionable clusters: interior colour-error blobs
(deltaE > threshold) and boundary segments missing by >= 1px, each with a
bounding box so the model can be corrected surgically.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from . import compare as cmp


def _clusters(hot: np.ndarray, values: np.ndarray | None, top: int):
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    sizes = ndimage.sum(hot, lab, range(1, n + 1))
    out = []
    for i in np.argsort(sizes)[::-1][:top]:
        ys, xs = np.where(lab == i + 1)
        c = {
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "px": int(sizes[i]),
        }
        if values is not None:
            c["max"] = round(float(values[ys, xs].max()), 2)
        out.append(c)
    return out


def find_residuals(
    ref_img: Image.Image,
    render_img: Image.Image,
    background: tuple[int, int, int],
    de_threshold: float = 5.0,
    top: int = 12,
) -> dict:
    ref = cmp.composite_over(ref_img, background).astype(float)
    ren = cmp.composite_over(render_img, background).astype(float)
    de = cmp.ciede2000(
        cmp.linear_to_lab(cmp.srgb_to_linear(ref)),
        cmp.linear_to_lab(cmp.srgb_to_linear(ren)),
    )
    close = lambda m: ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((3, 3))))
    ref_mask = close(cmp.foreground_mask(ref_img, background))
    ren_mask = close(cmp.foreground_mask(render_img, background))

    interior = ndimage.distance_transform_edt(ref_mask) >= 2.5
    hot = (de > de_threshold) & interior

    ref_edge = cmp.mask_boundary(ref_mask)
    ren_edge = cmp.mask_boundary(ren_mask)
    dt_ren = ndimage.distance_transform_edt(~ren_edge)
    dt_ref = ndimage.distance_transform_edt(~ref_edge)
    miss_ref = ref_edge & (dt_ren >= 1.0)   # reference boundary the render missed
    miss_ren = ren_edge & (dt_ref >= 1.0)   # render boundary that shouldn't exist

    return {
        "deltaE_threshold": de_threshold,
        "hot_px": int(hot.sum()),
        "colour_clusters": _clusters(hot, de, top),
        "edge_exact_fraction": round(float((dt_ren[ref_edge] == 0).mean()), 4)
        if ref_edge.any() else 1.0,
        "edge_missing_reference": _clusters(miss_ref, dt_ren, top),
        "edge_excess_render": _clusters(miss_ren, dt_ref, top),
    }
