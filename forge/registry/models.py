"""Data models for the dataset registry.

Defines the schema for registry entries: datasets, sources, and scale info.
All models use frozen dataclasses with __post_init__ validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_TAGS = frozenset({
    "manipulation",
    "bi_manual",
    "mobile_manipulation",
    "humanoid",
    "language_conditioned",
    "contact_rich",
    "simulation",
    "real_world",
    "multi_task",
    "single_task",
    "large_scale",
    # Access / usage restrictions (surfaced as a filter + badge in the HTML view)
    "registration_required",
    "non_commercial",
})

VALID_SOURCE_TYPES = frozenset({"gcs", "hf_hub", "http", "rsync"})

VALID_FORMATS = frozenset({
    "rlds", "lerobot", "lerobot-v2", "lerobot-v3",
    "hdf5", "zarr", "mcap", "rosbag", "other",
})


@dataclass(frozen=True)
class ScaleInfo:
    """Approximate scale of a dataset."""

    episodes: int | None = None
    hours: float | None = None
    approximate: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScaleInfo:
        return cls(
            episodes=data.get("episodes"),
            hours=data.get("hours"),
            approximate=data.get("approximate", True),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.episodes is not None:
            d["episodes"] = self.episodes
        if self.hours is not None:
            d["hours"] = self.hours
        d["approximate"] = self.approximate
        return d


@dataclass(frozen=True)
class SourceEntry:
    """A download source for a dataset."""

    type: str
    uri: str
    split: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source type '{self.type}'. "
                f"Must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}"
            )
        if not self.uri:
            raise ValueError("Source URI cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceEntry:
        return cls(
            type=data["type"],
            uri=data["uri"],
            split=data.get("split"),
            notes=data.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "uri": self.uri}
        if self.split is not None:
            d["split"] = self.split
        if self.notes is not None:
            d["notes"] = self.notes
        return d


@dataclass(frozen=True)
class DatasetEntry:
    """A dataset entry in the registry."""

    id: str
    name: str
    description: str
    format: str
    embodiment: list[str]
    sources: list[SourceEntry]
    paper_url: str | None = None
    license: str | None = None
    task_types: list[str] = field(default_factory=list)
    scale: ScaleInfo | None = None
    demo_suitable: bool = False
    demo_episodes: int | None = None
    demo_source_index: int | None = None
    tags: list[str] = field(default_factory=list)
    forge_verified: bool = False
    added_at: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Dataset id cannot be empty")
        if not self.name:
            raise ValueError(f"Dataset '{self.id}': name cannot be empty")
        if not self.sources:
            raise ValueError(f"Dataset '{self.id}': must have at least one source")
        invalid_tags = set(self.tags) - VALID_TAGS
        if invalid_tags:
            raise ValueError(
                f"Dataset '{self.id}': invalid tags {invalid_tags}. "
                f"Valid tags: {', '.join(sorted(VALID_TAGS))}"
            )
        if self.demo_source_index is not None:
            if self.demo_source_index < 0 or self.demo_source_index >= len(self.sources):
                raise ValueError(
                    f"Dataset '{self.id}': demo_source_index {self.demo_source_index} "
                    f"out of range (0-{len(self.sources) - 1})"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetEntry:
        sources = [SourceEntry.from_dict(s) for s in data["sources"]]
        scale = ScaleInfo.from_dict(data["scale"]) if data.get("scale") else None
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            format=data["format"],
            embodiment=data.get("embodiment", []),
            sources=sources,
            paper_url=data.get("paper_url"),
            license=data.get("license"),
            task_types=data.get("task_types", []),
            scale=scale,
            demo_suitable=data.get("demo_suitable", False),
            demo_episodes=data.get("demo_episodes"),
            demo_source_index=data.get("demo_source_index"),
            tags=data.get("tags", []),
            forge_verified=data.get("forge_verified", False),
            added_at=data.get("added_at"),
            notes=data.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format": self.format,
            "embodiment": self.embodiment,
            "sources": [s.to_dict() for s in self.sources],
        }
        if self.paper_url is not None:
            d["paper_url"] = self.paper_url
        if self.license is not None:
            d["license"] = self.license
        if self.task_types:
            d["task_types"] = self.task_types
        if self.scale is not None:
            d["scale"] = self.scale.to_dict()
        d["demo_suitable"] = self.demo_suitable
        if self.demo_episodes is not None:
            d["demo_episodes"] = self.demo_episodes
        if self.demo_source_index is not None:
            d["demo_source_index"] = self.demo_source_index
        if self.tags:
            d["tags"] = self.tags
        d["forge_verified"] = self.forge_verified
        if self.added_at is not None:
            d["added_at"] = self.added_at
        if self.notes is not None:
            d["notes"] = self.notes
        return d
