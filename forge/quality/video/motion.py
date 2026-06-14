"""Tier 1 motion metrics — classical optical flow (opencv).

Dense Farnebäck flow between consecutive downsampled grayscale frames, plus a
global-affine fit to split camera motion from scene motion, and a luma-histogram
discontinuity for shot/cut detection. opencv is an optional dependency (the
``[video]`` extra); importing this module's helpers without it raises a helpful
error rather than a bare ImportError.

References:
    - Farnebäck (2003) — two-frame motion estimation by polynomial expansion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forge.core.exceptions import MissingDependencyError

_cv2 = None


def _get_cv2():
    """Lazy import opencv."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            raise MissingDependencyError(
                dependency="opencv-python",
                feature="Tier 1 video motion metrics (--video-level motion)",
                install_hint="pip install forge-robotics[video]",
            )
    return _cv2


def to_uint8_gray(gray: NDArray) -> NDArray:
    """Clamp a float 0–255 grayscale frame to a contiguous uint8 array."""
    return np.ascontiguousarray(np.clip(gray, 0, 255).astype(np.uint8))


def dense_flow(prev_u8: NDArray, cur_u8: NDArray) -> NDArray:
    """Farnebäck dense optical flow → (H, W, 2) displacement field."""
    cv2 = _get_cv2()
    return cv2.calcOpticalFlowFarneback(
        prev_u8, cur_u8, None,
        0.5,   # pyr_scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0,     # flags
    )


def flow_magnitude(flow: NDArray) -> NDArray:
    """Per-pixel flow magnitude."""
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def affine_design(h: int, w: int) -> NDArray:
    """Design matrix [x, y, 1] (normalized coords) for a global affine fit."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs /= max(w - 1, 1)
    ys /= max(h - 1, 1)
    return np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=1)


def camera_scene_split(flow: NDArray, design: NDArray) -> tuple[float, float]:
    """Split flow into global (camera) and residual (scene) motion.

    Fits a global affine model ``flow ≈ design @ A`` by least squares; the
    residual is independent (object/scene) motion. Returns
    ``(mean_residual_magnitude, mean_total_magnitude)``.
    """
    f = flow.reshape(-1, 2).astype(np.float64)
    # numpy/BLAS can emit spurious divide/overflow RuntimeWarnings from the SIMD
    # matmul path; they don't affect the result, so silence them locally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coef, *_ = np.linalg.lstsq(design, f, rcond=None)
        residual = f - design @ coef
    resid_mag = np.sqrt((residual ** 2).sum(axis=1))
    total_mag = np.sqrt((f ** 2).sum(axis=1))
    return float(resid_mag.mean()), float(total_mag.mean())


def histogram_correlation(prev_u8: NDArray, cur_u8: NDArray) -> float:
    """Correlation of 32-bin luma histograms — drops sharply across a cut."""
    cv2 = _get_cv2()
    h1 = cv2.calcHist([prev_u8], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([cur_u8], [0], None, [32], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
