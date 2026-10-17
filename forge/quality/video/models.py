"""Data models for video (pixel) quality results.

``VideoQuality`` is attached to ``EpisodeQuality.video`` as a single nested,
optional block (same pattern as ``static`` / ``timestamps``). Headline scalars
are flattened into the per-episode report row via :meth:`VideoQuality.to_flat_dict`
so the JSON report stays one flat record per episode and ``forge filter`` can read
video fields directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class VideoEvent:
    """Camera-scoped observation evidence; frame end is exclusive.

    Times are None when neither timestamps nor FPS support a timebase.
    Sparse pixel events describe analyzed samples, not unseen intervening frames.
    """

    kind: str
    camera: str
    start_frame: int
    end_frame: int
    start_time_s: float | None
    end_time_s: float | None
    evidence: dict = field(default_factory=dict)


@dataclass
class CameraVideoQuality:
    """Tier 0 pixel metrics for a single camera stream."""

    camera: str
    num_frames: int = 0
    width: int | None = None
    height: int | None = None

    expected_frames: int = 0
    present_frames: int = 0
    missing_frames: int = 0
    decode_attempts: int = 0
    decode_failures: int = 0
    invalid_frames: int = 0
    sampled_out_frames: int = 0
    capped_frames: int = 0
    missing_fraction: float | None = None
    decode_coverage: float | None = None
    timestamp_coverage: float | None = None
    effective_fps: float | None = None
    timestamp_jitter_ratio: float | None = None
    estimated_drop_fraction: float | None = None
    timestamp_gaps: int = 0
    estimated_dropped_frames: int | None = None
    max_frame_gap_s: float | None = None
    timestamp_invalid_count: int = 0
    resolution_changes: int = 0
    max_frozen_duration_s: float | None = None
    mean_motion_px_s: float | None = None
    metric_status: dict[str, str] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    events: list[VideoEvent] = field(default_factory=list)

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

    # Tier 1: motion (None unless level="motion")
    mean_motion: float | None = None  # mean flow magnitude, px/frame
    motion_smoothness: float | None = None  # pixel-space LDLJ (more negative = jerkier)
    scene_motion_fraction: float | None = None  # residual / total flow (object vs camera)
    cut_count: int | None = None

    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CameraVideoQuality:
        cam = cls(
            **{f.name: data[f.name] for f in fields(cls) if f.name in data and f.name != "events"}
        )
        cam.events = [VideoEvent(**e) for e in data.get("events", [])]
        return cam


@dataclass
class VideoQuality:
    """Episode-level video quality, with per-camera detail and rollups."""

    per_camera: dict[str, CameraVideoQuality] = field(default_factory=dict)
    scorer_version: str = "video-v2-integrity"
    provenance: dict = field(default_factory=dict)

    # Episode rollups (worst-case across cameras, for one-line filtering)
    min_sharpness: float | None = None
    mean_sharpness: float | None = None
    blurry_fraction: float | None = None
    overexposed_fraction: float | None = None
    underexposed_fraction: float | None = None
    frozen_fraction: float | None = None
    max_frozen_run: int = 0
    mean_colorfulness: float | None = None

    # Tier 1: motion rollups
    mean_motion: float | None = None
    motion_smoothness: float | None = None
    scene_motion_fraction: float | None = None
    cut_count: int | None = None

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
            "video_mean_motion": _round(self.mean_motion),
            "video_motion_smoothness": _round(self.motion_smoothness),
            "video_scene_motion_fraction": _round(self.scene_motion_fraction),
            "video_cut_count": self.cut_count,
        }

    def to_dict(self) -> dict:
        d = self.to_flat_dict()
        d["per_camera"] = {n: c.to_dict() for n, c in self.per_camera.items()}
        d["scorer_version"] = self.scorer_version
        d["provenance"] = self.provenance
        d["flags"] = self.flags
        return d

    @classmethod
    def from_dict(cls, data: dict) -> VideoQuality:
        video = cls.from_flat_dict(data) or cls()
        video.per_camera = {
            name: CameraVideoQuality.from_dict(cam)
            for name, cam in data.get("per_camera", {}).items()
        }
        video.scorer_version = data.get("scorer_version", "video-v1")
        video.provenance = data.get("provenance", {})
        video.flags = data.get("flags", [])
        return video

    @classmethod
    def from_flat_dict(cls, data: dict) -> VideoQuality | None:
        """Reconstruct rollup scalars from a flat report row (filter-side).

        Per-camera detail is not round-tripped — only the rollups that filtering
        needs. Returns None if the row carries no video fields.
        """
        if "video_min_sharpness" not in data and "video_num_cameras" not in data:
            return None
        return cls(
            scorer_version="video-v1",
            min_sharpness=data.get("video_min_sharpness"),
            mean_sharpness=data.get("video_mean_sharpness"),
            blurry_fraction=data.get("video_blurry_fraction"),
            overexposed_fraction=data.get("video_overexposed_fraction"),
            underexposed_fraction=data.get("video_underexposed_fraction"),
            frozen_fraction=data.get("video_frozen_fraction"),
            max_frozen_run=data.get("video_max_frozen_run", 0) or 0,
            mean_colorfulness=data.get("video_colorfulness"),
            mean_motion=data.get("video_mean_motion"),
            motion_smoothness=data.get("video_motion_smoothness"),
            scene_motion_fraction=data.get("video_scene_motion_fraction"),
            cut_count=data.get("video_cut_count"),
        )


def _round(v: float | None, ndigits: int = 3) -> float | None:
    return round(v, ndigits) if v is not None else None
