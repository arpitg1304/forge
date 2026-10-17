"""Data models for quality analysis results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.quality.video.models import VideoQuality


@dataclass
class TimestampResult:
    """Result of timestamp regularity analysis."""

    dt_mean: float
    dt_std: float
    jitter_ratio: float
    num_gaps: int
    gap_locations: list[int]
    effective_fps: float


@dataclass
class StaticResult:
    """Result of static episode detection."""

    static_fraction: float
    is_mostly_static: bool
    longest_static_run: int
    static_segments: list[tuple[int, int, float]]


@dataclass
class QualityEvent:
    """Review candidate on [start_frame, end_frame), with absolute data times."""

    kind: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    channels: list[int] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass
class EpisodeQuality:
    """Quality metrics for a single episode."""

    episode_id: str
    num_frames: int = 0
    scorer_version: str = "q-v2-proprio"
    metric_status: dict[str, str] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    events: list[QualityEvent] = field(default_factory=list)
    diagnostics: dict[str, float | int | None] = field(default_factory=dict)
    grippers: dict[str, dict] = field(default_factory=dict)
    observed_range_edge_fraction: float | None = None

    # Metric 1: Dead actions
    dead_fraction: float | None = None
    dead_ranges: list[tuple[int, int]] | None = None

    # Metric 2: Smoothness (LDLJ)
    ldlj: float | None = None

    # Metric 3: Gripper chatter
    gripper_chatter_count: int | None = None
    gripper_chatter_rate: float | None = None
    is_chattery: bool | None = None

    # Metric 4: Path length
    joint_path_length: float | None = None
    cartesian_path_length: float | None = None

    # Metric 5: Timestamp regularity
    timestamps: TimestampResult | None = None

    # Metric 6: Action saturation
    saturation_per_dim: list[float] | None = None
    overall_saturation: float | None = None
    saturated_dims: list[int] | None = None

    # Metric 7: Static detection
    static: StaticResult | None = None

    # Metric 8: Entropy
    entropy_per_dim: list[float] | None = None
    mean_entropy: float | None = None

    # Metric 9a: SPARC smoothness (frequency-domain alternative to LDLJ)
    sparc: float | None = None

    # Metric 9b: PSD band energy — high_fraction is the headline chatter scalar
    psd_high_fraction: float | None = None
    psd_bands: dict | None = None

    # Metric 9c: State-conditioned action variance (proprio cousin of STAC)
    state_action_consistency: float | None = None

    # Metric 10: Composite score
    overall_score: float | None = None
    subscores: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    # Video modality (Tier 0 pixel metrics) — populated only with --video.
    video: VideoQuality | None = None

    @property
    def is_flagged(self) -> bool:
        return len(self.flags) > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("video")
        if self.static is not None:
            d["static_fraction"] = self.static.static_fraction
        if self.timestamps is not None:
            d["jitter_ratio"] = self.timestamps.jitter_ratio
        if self.video is not None:
            d.update(self.video.to_flat_dict())
            d["video"] = self.video.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> EpisodeQuality:
        nested = {"static", "timestamps", "video", "events"}
        kwargs = {
            f.name: data[f.name] for f in fields(cls) if f.name in data and f.name not in nested
        }
        kwargs.setdefault("scorer_version", "q-v1")
        eq = cls(**kwargs)
        if data.get("static") is not None:
            values = dict(data["static"])
            values["static_segments"] = [tuple(x) for x in values["static_segments"]]
            eq.static = StaticResult(**values)
        elif data.get("static_fraction") is not None:
            value = data["static_fraction"]
            eq.static = StaticResult(value, value > 0.5, 0, [])
        if data.get("timestamps") is not None:
            eq.timestamps = TimestampResult(**data["timestamps"])
        eq.events = [QualityEvent(**event) for event in data.get("events", [])]
        if eq.dead_ranges is not None:
            eq.dead_ranges = [tuple(x) for x in eq.dead_ranges]
        from forge.quality.video.models import VideoQuality

        eq.video = (
            VideoQuality.from_dict(data["video"])
            if data.get("video") is not None
            else VideoQuality.from_flat_dict(data)
        )
        return eq


@dataclass
class QualityReport:
    """Quality report for an entire dataset."""

    dataset_path: str
    num_episodes: int = 0
    overall_score: float | None = None
    computed_at: str = ""

    subscores: dict[str, float] = field(default_factory=dict)
    per_episode: list[EpisodeQuality] = field(default_factory=list)
    flagged_episodes: dict[str, list[str]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "num_episodes": self.num_episodes,
            "overall_score": round(self.overall_score, 2)
            if self.overall_score is not None
            else None,
            "computed_at": self.computed_at,
            "subscores": {k: round(v, 3) for k, v in self.subscores.items()},
            "per_episode": [ep.to_dict() for ep in self.per_episode],
            "flags": self.flags,
            "flagged_episodes": self.flagged_episodes,
            "recommendations": self.recommendations,
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> QualityReport:
        """Load a QualityReport from a JSON file exported by to_json()."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> QualityReport:
        """Reconstruct a QualityReport from a dictionary."""
        report = cls(
            dataset_path=data["dataset_path"],
            num_episodes=data.get("num_episodes", 0),
            overall_score=data.get("overall_score", 0.0),
            computed_at=data.get("computed_at", ""),
        )
        report.subscores = data.get("subscores", {})
        report.flags = data.get("flags", [])
        report.flagged_episodes = data.get("flagged_episodes", {})
        report.recommendations = data.get("recommendations", [])

        for ep_data in data.get("per_episode", []):
            report.per_episode.append(EpisodeQuality.from_dict(ep_data))

        return report
