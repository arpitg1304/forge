"""Data models for video (pixel) quality results.

``VideoQuality`` is attached to ``EpisodeQuality.video`` as a single nested,
optional block (same pattern as ``static`` / ``timestamps``). Headline scalars
are flattened into the per-episode report row via :meth:`VideoQuality.to_flat_dict`
so the JSON report stays one flat record per episode and ``forge filter`` can read
video fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraVideoQuality:
    """Tier 0 pixel metrics for a single camera stream."""

    camera: str
    num_frames: int = 0
    width: int | None = None
    height: int | None = None

    # Sharpness (variance of Laplacian)
    mean_sharpness: float | None = None
    min_sharpness: float | None = None
    blurry_fraction: float | None = None

    # Exposure
    mean_luma: float | None = None
    overexposed_fraction: float | None = None
    underexposed_fraction: float | None = None

    # Frozen frames / encoder stall
    frozen_fraction: float | None = None
    longest_frozen_run: int = 0

    # Scene diversity input
    mean_colorfulness: float | None = None

    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "mean_sharpness": _round(self.mean_sharpness),
            "min_sharpness": _round(self.min_sharpness),
            "blurry_fraction": _round(self.blurry_fraction),
            "mean_luma": _round(self.mean_luma),
            "overexposed_fraction": _round(self.overexposed_fraction),
            "underexposed_fraction": _round(self.underexposed_fraction),
            "frozen_fraction": _round(self.frozen_fraction),
            "longest_frozen_run": self.longest_frozen_run,
            "mean_colorfulness": _round(self.mean_colorfulness),
            "flags": self.flags,
        }


@dataclass
class VideoQuality:
    """Episode-level video quality, with per-camera detail and rollups."""

    per_camera: dict[str, CameraVideoQuality] = field(default_factory=dict)

    # Episode rollups (worst-case across cameras, for one-line filtering)
    min_sharpness: float | None = None
    mean_sharpness: float | None = None
    blurry_fraction: float | None = None
    overexposed_fraction: float | None = None
    underexposed_fraction: float | None = None
    frozen_fraction: float | None = None
    max_frozen_run: int = 0
    mean_colorfulness: float | None = None

    flags: list[str] = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        """Headline scalars for the flat per-episode report row."""
        return {
            "video_num_cameras": len(self.per_camera),
            "video_min_sharpness": _round(self.min_sharpness),
            "video_mean_sharpness": _round(self.mean_sharpness),
            "video_blurry_fraction": _round(self.blurry_fraction),
            "video_overexposed_fraction": _round(self.overexposed_fraction),
            "video_underexposed_fraction": _round(self.underexposed_fraction),
            "video_frozen_fraction": _round(self.frozen_fraction),
            "video_max_frozen_run": self.max_frozen_run,
            "video_colorfulness": _round(self.mean_colorfulness),
        }

    def to_dict(self) -> dict:
        d = self.to_flat_dict()
        d["per_camera"] = {n: c.to_dict() for n, c in self.per_camera.items()}
        return d

    @classmethod
    def from_flat_dict(cls, data: dict) -> VideoQuality | None:
        """Reconstruct rollup scalars from a flat report row (filter-side).

        Per-camera detail is not round-tripped — only the rollups that filtering
        needs. Returns None if the row carries no video fields.
        """
        if "video_min_sharpness" not in data and "video_num_cameras" not in data:
            return None
        return cls(
            min_sharpness=data.get("video_min_sharpness"),
            mean_sharpness=data.get("video_mean_sharpness"),
            blurry_fraction=data.get("video_blurry_fraction"),
            overexposed_fraction=data.get("video_overexposed_fraction"),
            underexposed_fraction=data.get("video_underexposed_fraction"),
            frozen_fraction=data.get("video_frozen_fraction"),
            max_frozen_run=data.get("video_max_frozen_run", 0) or 0,
            mean_colorfulness=data.get("video_colorfulness"),
        )


def _round(v: float | None, ndigits: int = 3) -> float | None:
    return round(v, ndigits) if v is not None else None
