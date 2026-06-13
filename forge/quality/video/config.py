"""Configuration for video (pixel) quality metrics.

Tier 0 only for now — pure pixel statistics computed on a downscaled
grayscale frame. All tunable thresholds live here, mirroring
``forge/quality/config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoQualityConfig:
    """Configuration for Tier 0 video quality analysis.

    Groups:
        Decode / sampling: how frames are pulled and downscaled before metrics.
        Sharpness: variance-of-Laplacian blur detection.
        Exposure: over/under-exposure from per-frame luma + clipping.
        Frozen frames: consecutive-frame MAE for encoder-stall / static video.
    """

    # ── Decode / frame sampling ──
    downscale: int = 128                 # longest side (px) of the analysis frame
    sample_stride: int = 1               # analyze every Nth frame that carries images
    max_frames: int = 0                  # cap analyzed frames per camera (0 = no cap)
    cameras: list[str] | None = None     # restrict to these camera names (None = all)

    # ── Sharpness (variance of Laplacian, on 0–255 grayscale) ──
    sharpness_flag: float = 100.0        # per-frame var-of-Laplacian below this = blurry
    blurry_fraction_flag: float = 0.30   # >30% blurry frames => 'blurry' flag

    # ── Exposure (per-frame luma + pixel clipping, 0–255) ──
    clip_low: float = 16.0               # luma <= this counts as crushed-black
    clip_high: float = 239.0             # luma >= this counts as blown-white
    clip_pixel_fraction: float = 0.20    # frame is clipped if >20% of pixels are clipped
    dark_luma_flag: float = 40.0         # mean luma below this = underexposed frame
    exposure_flag: float = 0.25          # >25% over/under-exposed frames => exposure flag

    # ── Frozen frame / encoder stall (consecutive-frame MAE, 0–255) ──
    frozen_mae: float = 1.0              # MAE below this between consecutive frames = frozen
    frozen_run_flag: int = 10            # longest frozen run above this => 'frozen_frames'
    frozen_fraction_flag: float = 0.50   # or >50% frozen transitions => 'frozen_frames'
