"""Data models for quality analysis results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
class EpisodeQuality:
    """Quality metrics for a single episode."""

    episode_id: str
    num_frames: int = 0

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
    video: "VideoQuality | None" = None

    @property
    def is_flagged(self) -> bool:
        return len(self.flags) > 0

    def to_dict(self) -> dict:
        d: dict = {
            "episode_id": self.episode_id,
            "num_frames": self.num_frames,
            "overall_score": self.overall_score,
            "ldlj": self.ldlj,
            "sparc": self.sparc,
            "dead_fraction": self.dead_fraction,
            "gripper_chatter_rate": self.gripper_chatter_rate,
            "joint_path_length": self.joint_path_length,
            "overall_saturation": self.overall_saturation,
            "mean_entropy": self.mean_entropy,
            "psd_high_fraction": self.psd_high_fraction,
            "state_action_consistency": self.state_action_consistency,
            "flags": self.flags,
        }
        if self.static is not None:
            d["static_fraction"] = self.static.static_fraction
        if self.timestamps is not None:
            d["jitter_ratio"] = self.timestamps.jitter_ratio
        if self.video is not None:
            d.update(self.video.to_flat_dict())
        return d


@dataclass
class QualityReport:
    """Quality report for an entire dataset."""

    dataset_path: str
    num_episodes: int = 0
    overall_score: float = 0.0
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
            "overall_score": round(self.overall_score, 2),
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
            eq = EpisodeQuality(
                episode_id=ep_data["episode_id"],
                num_frames=ep_data.get("num_frames", 0),
                overall_score=ep_data.get("overall_score"),
                dead_fraction=ep_data.get("dead_fraction"),
                ldlj=ep_data.get("ldlj"),
                sparc=ep_data.get("sparc"),
                gripper_chatter_rate=ep_data.get("gripper_chatter_rate"),
                joint_path_length=ep_data.get("joint_path_length"),
                overall_saturation=ep_data.get("overall_saturation"),
                mean_entropy=ep_data.get("mean_entropy"),
                psd_high_fraction=ep_data.get("psd_high_fraction"),
                state_action_consistency=ep_data.get("state_action_consistency"),
                flags=ep_data.get("flags", []),
            )
            from forge.quality.video.models import VideoQuality

            eq.video = VideoQuality.from_flat_dict(ep_data)
            report.per_episode.append(eq)

        return report
