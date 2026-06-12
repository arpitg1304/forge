"""Data models for tokenizer comparison results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TokenizerStats:
    """Reconstruction / compression metrics for one strategy on a dataset."""

    strategy: str
    vocab_size: int
    granularity: str
    mse: float
    mae: float
    max_abs: float
    per_dim_rmse: list[float]
    tokens_per_step: float
    compression_ratio: float
    vocab_utilization: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "vocab_size": self.vocab_size,
            "granularity": self.granularity,
            "mse": self.mse,
            "mae": self.mae,
            "max_abs": self.max_abs,
            "per_dim_rmse": self.per_dim_rmse,
            "tokens_per_step": self.tokens_per_step,
            "compression_ratio": self.compression_ratio,
            "vocab_utilization": self.vocab_utilization,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenizerStats:
        return cls(**data)


@dataclass
class ComparisonReport:
    """Benchmark of every strategy on a dataset's action corpus."""

    dataset_path: str
    num_frames: int = 0
    action_dim: int = 0
    sample_size: int = 0
    computed_at: str = ""
    stats_by_strategy: dict[str, TokenizerStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    def best_by(self, metric: str = "mae") -> str | None:
        """Name of the strategy minimizing ``metric`` (lower is better)."""
        if not self.stats_by_strategy:
            return None
        return min(
            self.stats_by_strategy,
            key=lambda s: getattr(self.stats_by_strategy[s], metric),
        )

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "num_frames": self.num_frames,
            "action_dim": self.action_dim,
            "sample_size": self.sample_size,
            "computed_at": self.computed_at,
            "stats_by_strategy": {
                k: v.to_dict() for k, v in self.stats_by_strategy.items()
            },
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> ComparisonReport:
        report = cls(
            dataset_path=data["dataset_path"],
            num_frames=data.get("num_frames", 0),
            action_dim=data.get("action_dim", 0),
            sample_size=data.get("sample_size", 0),
            computed_at=data.get("computed_at", ""),
        )
        report.stats_by_strategy = {
            k: TokenizerStats.from_dict(v)
            for k, v in data.get("stats_by_strategy", {}).items()
        }
        return report

    @classmethod
    def from_json(cls, path: str | Path) -> ComparisonReport:
        with open(path) as f:
            return cls.from_dict(json.load(f))
