"""Render-vs-reference comparison: metrics and diagnostic images.

All colour error is computed in perceptual space (CIEDE2000 on Lab derived
from linear-light RGB); silhouette and edge metrics come from foreground
masks. Interior metrics exclude antialiased edge pixels via the distance
transform, per the build spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------- colour math


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = rgb_lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(
        C1p * C2p == 0,
        hsum,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hbp - 30))
        + 0.24 * np.cos(np.radians(2 * hbp))
        + 0.32 * np.cos(np.radians(3 * hbp + 6))
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    )
    d_theta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc
    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


# ------------------------------------------------------------------- masking


def composite_over(img: Image.Image, background: tuple[int, int, int]) -> np.ndarray:
    """Flatten RGBA onto an opaque background; returns HxWx3 uint8."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
    alpha = arr[..., 3:4] / 255.0
    bg = np.array(background, dtype=np.float64)
    rgb = arr[..., :3] * alpha + bg * (1 - alpha)
    return rgb.round().astype(np.uint8)


def foreground_mask(
    img: Image.Image, background: tuple[int, int, int], threshold: float | None = None
) -> np.ndarray:
    """Foreground at ~50% coverage from a flattened image.

    Works on colour distance from the background so reference PNG and SVG
    render are treated identically (an alpha>=128 rule on one side and a
    colour rule on the other would bias mask edges by up to a pixel).
    The default threshold is half the 5th-percentile full-strength
    foreground distance, approximating the 50%-coverage boundary.
    """
    rgb = composite_over(img, background).astype(np.float64)
    dist = np.linalg.norm(rgb - np.array(background, dtype=np.float64), axis=-1)
    if threshold is not None:
        return dist > threshold
    # Adaptive 50%-coverage contour: normalise each pixel's background
    # distance by the full-strength foreground level in its neighbourhood,
    # so dark and light shapes get the same effective boundary.
    full = ndimage.maximum_filter(dist, size=9)
    coverage = dist / np.maximum(full, 1e-6)
    return (coverage >= 0.5) & (dist > 25.0)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask)


# ------------------------------------------------------------------- compare


def compare(
    ref_img: Image.Image,
    render_img: Image.Image,
    background: tuple[int, int, int],
    out_dir: Path | None = None,
    interior_margin: float = 2.5,
) -> dict:
    ref = composite_over(ref_img, background)
    ren = composite_over(render_img, background)
    if ref.shape != ren.shape:
        raise ValueError(f"size mismatch: {ref.shape} vs {ren.shape}")

    ref_mask = foreground_mask(ref_img, background)
    ren_mask = foreground_mask(render_img, background)

    inter = np.logical_and(ref_mask, ren_mask).sum()
    union = np.logical_or(ref_mask, ren_mask).sum()
    iou = float(inter / union) if union else 1.0

    # Edge metrics use closed, hole-filled masks: strong internal colour
    # seams can dip below the coverage threshold and read as spurious
    # boundaries (closing bridges 1px dip channels that reach the outside).
    # Real holes in a design would be hidden here — IoU still sees them.
    ref_mask = ndimage.binary_fill_holes(ndimage.binary_closing(ref_mask, np.ones((3, 3))))
    ren_mask = ndimage.binary_fill_holes(ndimage.binary_closing(ren_mask, np.ones((3, 3))))

    # symmetric edge distance between mask boundaries
    ref_edge = mask_boundary(ref_mask)
    ren_edge = mask_boundary(ren_mask)
    edge_metrics: dict[str, float] = {}
    if ref_edge.any() and ren_edge.any():
        dt_ref = ndimage.distance_transform_edt(~ref_edge)
        dt_ren = ndimage.distance_transform_edt(~ren_edge)
        d1 = dt_ren[ref_edge]  # ref boundary -> nearest render boundary
        d2 = dt_ref[ren_edge]
        both = np.concatenate([d1, d2])
        edge_metrics = {
            "edge_dist_mean": float(both.mean()),
            "edge_dist_p95": float(np.percentile(both, 95)),
            "edge_dist_max": float(both.max()),
        }

    # confident interior: inside both masks, away from the reference edge
    dt_inside = ndimage.distance_transform_edt(ref_mask)
    interior = np.logical_and(ref_mask & ren_mask, dt_inside >= interior_margin)

    ref_lab = linear_to_lab(srgb_to_linear(ref.astype(np.float64)))
    ren_lab = linear_to_lab(srgb_to_linear(ren.astype(np.float64)))
    de = ciede2000(ref_lab, ren_lab)

    colour_metrics: dict[str, float] = {}
    if interior.any():
        de_in = de[interior]
        mae_lin = np.abs(
            srgb_to_linear(ref.astype(np.float64)) - srgb_to_linear(ren.astype(np.float64))
        )[interior].mean()
        colour_metrics = {
            "deltaE_mean": float(de_in.mean()),
            "deltaE_p95": float(np.percentile(de_in, 95)),
            "deltaE_max": float(de_in.max()),
            "mae_linear_rgb": float(mae_lin),
            "interior_pixels": int(interior.sum()),
        }

    # Texture floor and low-frequency colour.
    #
    # Per-pixel deltaE asks "is every pixel right", which is the correct
    # question only when the source is flat or smoothly shaded. On artwork
    # carrying grain, brushed metal or noise, the source disagrees with
    # ITSELF by more than any vector could, and the metric reports failure
    # for a reconstruction that reads correctly at a glance.
    #
    # `texture_std` measures how much the reference varies locally in its own
    # interior: around 1-2 for clean vector art, 10+ for a textured render.
    # When it is high, judge by the low-frequency figures, which blur both
    # images first and so score structure and shading rather than grain.
    texture_metrics: dict[str, float] = {}
    if interior.any():
        ref_grey = ref.astype(np.float64).mean(axis=2)
        local_var = (ndimage.uniform_filter(ref_grey ** 2, 5)
                     - ndimage.uniform_filter(ref_grey, 5) ** 2)
        texture_metrics["texture_std"] = float(
            np.sqrt(np.clip(local_var[interior], 0, None)).mean())
        sigma = max(2.0, min(ref.shape[0], ref.shape[1]) / 200.0)
        ref_lo = ndimage.gaussian_filter(ref.astype(np.float64), (sigma, sigma, 0))
        ren_lo = ndimage.gaussian_filter(ren.astype(np.float64), (sigma, sigma, 0))
        de_lo = ciede2000(linear_to_lab(srgb_to_linear(ref_lo)),
                          linear_to_lab(srgb_to_linear(ren_lo)))[interior]
        texture_metrics.update({
            "deltaE_lowfreq_mean": float(de_lo.mean()),
            "deltaE_lowfreq_p95": float(np.percentile(de_lo, 95)),
            "lowfreq_sigma": float(sigma),
        })

    metrics = {"silhouette_iou": iou, **edge_metrics, **colour_metrics,
               **texture_metrics}

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ref).save(out_dir / "reference.png")
        Image.fromarray(ren).save(out_dir / "render.png")
        overlay = ((ref.astype(np.uint16) + ren.astype(np.uint16)) // 2).astype(np.uint8)
        Image.fromarray(overlay).save(out_dir / "overlay.png")
        diff = np.abs(ref.astype(np.int16) - ren.astype(np.int16))
        gain = np.clip(diff * 4, 0, 255).astype(np.uint8)
        Image.fromarray(gain).save(out_dir / "difference.png")
        # deltaE heatmap: grayscale, 4x gain, masked to union of foregrounds
        de_vis = np.clip(de * 12, 0, 255).astype(np.uint8)
        de_vis[~np.logical_or(ref_mask, ren_mask)] = 0
        Image.fromarray(de_vis).save(out_dir / "deltaE.png")
        edge_vis = np.zeros((*ref_mask.shape, 3), dtype=np.uint8)
        edge_vis[ref_edge] = [255, 64, 64]   # red: reference boundary
        edge_vis[ren_edge] = [64, 255, 64]   # green: render boundary
        edge_vis[ref_edge & ren_edge] = [255, 255, 255]  # white: coincident
        Image.fromarray(edge_vis).save(out_dir / "edge-difference.png")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    return metrics
