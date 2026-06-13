"""Tier 0 pixel-statistic metrics for video frames.

All functions operate on numpy arrays only — no model, no GPU, no optional
dependencies. They run on a downscaled frame so the cost is dominated by the
upstream decode rather than the math here.

References:
    - Pech-Pacheco et al. (2000) — variance of the Laplacian as a focus measure.
    - Hasler & Süsstrunk (2003) — "Measuring Colourfulness in Natural Images".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Rec. 601 luma weights — matches the yuv420p path used by the video encoder.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _to_float_image(img: NDArray) -> NDArray:
    """Coerce a decoded image to float in the 0–255 range.

    Accepts uint8 HWC/HW arrays or float arrays in either 0–1 or 0–255.
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float64)
    a = a.astype(np.float64)
    # Float images from some readers are normalized to [0, 1]; rescale to 0–255.
    finite = a[np.isfinite(a)]
    if finite.size and finite.max() <= 1.0 + 1e-6 and finite.min() >= -1e-6:
        a = a * 255.0
    return np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)


def _downscale(arr: NDArray, target: int) -> NDArray:
    """Nearest-neighbour downscale so the longest spatial side is ``target``.

    Works on (H, W) or (H, W, C) arrays. Returns the input unchanged if it is
    already at or below the target size.
    """
    h, w = arr.shape[0], arr.shape[1]
    longest = max(h, w)
    if longest <= target:
        return arr
    scale = target / longest
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    ys = np.minimum((np.arange(nh) * (h / nh)).astype(np.intp), h - 1)
    xs = np.minimum((np.arange(nw) * (w / nw)).astype(np.intp), w - 1)
    return arr[ys][:, xs]


def prepare(img: NDArray, target: int) -> tuple[NDArray, NDArray | None]:
    """Downscale once and return ``(gray, rgb_small)``.

    ``gray`` is a 2-D float array in 0–255. ``rgb_small`` is the downscaled
    colour image (float 0–255) when the source has 3+ channels, else None — so
    colourfulness is simply skipped for grayscale/depth streams.
    """
    a = _to_float_image(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        rgb_small = _downscale(a[..., :3], target)
        gray = rgb_small @ _LUMA
        return gray, rgb_small
    if a.ndim == 3:
        gray = _downscale(a[..., 0], target)
        return gray, None
    return _downscale(a, target), None


def laplacian_variance(gray: NDArray) -> float:
    """Variance of the 4-neighbour Laplacian — higher = sharper.

    Low values indicate motion blur, defocus, or compression mush.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


def clip_fractions(gray: NDArray, clip_low: float, clip_high: float) -> tuple[float, float]:
    """Fraction of pixels at the black floor and white ceiling."""
    low = float(np.mean(gray <= clip_low))
    high = float(np.mean(gray >= clip_high))
    return low, high


def colorfulness(rgb: NDArray) -> float:
    """Hasler–Süsstrunk colourfulness on a float 0–255 RGB image."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def frame_mae(gray_a: NDArray, gray_b: NDArray) -> float:
    """Mean absolute pixel difference between two same-shape grayscale frames."""
    return float(np.mean(np.abs(gray_a - gray_b)))


def longest_true_run(flags: list[bool]) -> int:
    """Length of the longest consecutive run of True in ``flags``."""
    best = 0
    cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
