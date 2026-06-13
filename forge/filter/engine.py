"""Filter engine for quality-based episode filtering.

Reads a dataset, evaluates each episode against quality criteria,
and writes only passing episodes to the output (same format).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from forge.core.exceptions import ForgeError, UnsupportedFormatError
from forge.formats.registry import FormatRegistry
from forge.quality.models import EpisodeQuality, QualityReport

logger = logging.getLogger(__name__)


# Flags produced by the Tier 0 video analyzer — used to decide whether
# --exclude-flags requires video analysis during live filtering.
_VIDEO_FLAGS = frozenset({"blurry", "over_exposed", "under_exposed", "frozen_frames"})


@dataclass
class FilterConfig:
    """Configuration for a filter operation."""

    # Quality-based filters
    min_quality: float | None = None
    exclude_flags: list[str] | None = None

    # Video (Tier 0) numeric filters — keep episodes within these bounds.
    min_sharpness: float | None = None          # reject if video min_sharpness < this
    max_frozen_fraction: float | None = None    # reject if frozen_fraction > this
    max_overexposed_fraction: float | None = None
    max_underexposed_fraction: float | None = None

    # Explicit episode selection
    include_episodes: list[str] | None = None
    exclude_episodes: list[str] | None = None

    # Pre-computed report
    from_report: Path | None = None

    # Quality config for live analysis
    gripper_dim: int = -1
    fps: float = 30.0
    action_bounds: tuple[float, float] | None = None

    # Video analysis controls (live mode only)
    video_downscale: int = 128
    video_stride: int = 1


@dataclass
class FilterResult:
    """Result of a filter operation."""

    success: bool
    format: str
    total_episodes: int = 0
    episodes_kept: int = 0
    episodes_excluded: int = 0
    total_frames_kept: int = 0
    output_path: Path | None = None
    dry_run: bool = False

    kept_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class FilterEngine:
    """Orchestrates quality-based filtering of dataset episodes."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def filter(
        self,
        source: str | Path,
        output: Path | None = None,
        source_format: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> FilterResult:
        """Filter a dataset based on quality criteria.

        Args:
            source: Path to source dataset.
            output: Output path (None for dry-run).
            source_format: Format hint (auto-detect if None).
            progress_callback: Optional callback(stage, current, total).

        Returns:
            FilterResult with filtering outcome.
        """
        source = Path(source)
        dry_run = output is None
        result = FilterResult(success=False, format="unknown", dry_run=dry_run)

        # 1. Detect format
        if source_format is None:
            try:
                source_format = FormatRegistry.detect_format(source)
            except Exception as e:
                result.errors.append(f"Format detection failed: {e}")
                return result

        result.format = source_format
        reader = FormatRegistry.get_reader(source_format)

        # 2. Get writer (only if writing)
        writer = None
        if not dry_run:
            if not FormatRegistry.has_writer(source_format):
                raise ForgeError(
                    f"No writer for format '{source_format}'. "
                    f"Convert to a writable format first with `forge convert`."
                )
            writer = FormatRegistry.get_writer(source_format)

        # 3. Inspect source
        try:
            dataset_info = reader.inspect(source)
        except Exception as e:
            result.errors.append(f"Inspection failed: {e}")
            return result

        result.total_episodes = dataset_info.num_episodes or 0

        # 4. Load quality data
        quality_map: dict[str, EpisodeQuality] = {}
        analyzer = None

        if self.config.from_report:
            report = QualityReport.from_json(self.config.from_report)
            quality_map = {eq.episode_id: eq for eq in report.per_episode}
        elif self._needs_quality_analysis():
            from forge.quality import QualityAnalyzer, QualityConfig

            qconfig = QualityConfig(
                gripper_dim=self.config.gripper_dim,
                fps=self.config.fps,
                action_bounds=self.config.action_bounds,
            )
            video_config = None
            if self._needs_video_analysis():
                from forge.quality.video import VideoQualityConfig

                video_config = VideoQualityConfig(
                    downscale=self.config.video_downscale,
                    sample_stride=self.config.video_stride,
                )
            analyzer = QualityAnalyzer(config=qconfig, video_config=video_config)

        # 5. Configure writer
        if writer is not None and hasattr(writer, "config"):
            wconfig = writer.config
            if hasattr(wconfig, "fps") and dataset_info.inferred_fps:
                wconfig.fps = dataset_info.inferred_fps
            if hasattr(wconfig, "robot_type") and dataset_info.inferred_robot_type:
                wconfig.robot_type = dataset_info.inferred_robot_type

        # 6. Iterate episodes
        kept_index = 0
        for i, episode in enumerate(reader.read_episodes(source)):
            if progress_callback:
                progress_callback("episode", i, result.total_episodes)

            # Get quality data
            eq: EpisodeQuality | None = None
            if self.config.from_report:
                eq = quality_map.get(episode.episode_id)
            elif analyzer is not None:
                eq = analyzer.analyze_episode(episode)

            # Evaluate
            keep, reasons = self._evaluate_episode(episode.episode_id, eq)

            if keep:
                result.kept_ids.append(episode.episode_id)
                if writer is not None:
                    try:
                        episode.load_frames()
                        writer.write_episode(episode, output, episode_index=kept_index)
                        result.total_frames_kept += len(episode._frames_cache or [])
                    except Exception as e:
                        result.errors.append(f"Write failed for {episode.episode_id}: {e}")
                        result.episodes_excluded += 1
                        continue
                kept_index += 1
                result.episodes_kept += 1
            else:
                result.excluded_ids.append(episode.episode_id)
                result.exclusion_reasons[episode.episode_id] = reasons
                result.episodes_excluded += 1

        # 7. Finalize
        if writer is not None and kept_index > 0:
            dataset_info.num_episodes = kept_index
            dataset_info.total_frames = result.total_frames_kept
            try:
                writer.finalize(output, dataset_info)
            except Exception as e:
                result.errors.append(f"Finalization failed: {e}")
                return result
            result.output_path = output

        result.success = len(result.errors) == 0
        return result

    def _needs_quality_analysis(self) -> bool:
        """Check if any quality-based filter criteria are set."""
        return (
            self.config.min_quality is not None
            or self.config.exclude_flags is not None
            or self._needs_video_analysis()
        )

    def _needs_video_analysis(self) -> bool:
        """Check if any criterion requires Tier 0 video metrics.

        True when a numeric video threshold is set, or when --exclude-flags
        names a video flag (which the analyzer only produces with video on).
        """
        c = self.config
        if any(
            v is not None
            for v in (
                c.min_sharpness,
                c.max_frozen_fraction,
                c.max_overexposed_fraction,
                c.max_underexposed_fraction,
            )
        ):
            return True
        if c.exclude_flags and any(f in _VIDEO_FLAGS for f in c.exclude_flags):
            return True
        return False

    def _evaluate_episode(
        self, episode_id: str, eq: EpisodeQuality | None
    ) -> tuple[bool, list[str]]:
        """Evaluate whether an episode passes all filter criteria.

        Returns:
            (keep, reasons) where reasons lists why it was excluded.
        """
        reasons: list[str] = []

        # Include list
        if self.config.include_episodes is not None:
            if episode_id not in self.config.include_episodes:
                reasons.append("not in include list")
                return False, reasons

        # Exclude list
        if self.config.exclude_episodes is not None:
            if episode_id in self.config.exclude_episodes:
                reasons.append("in exclude list")
                return False, reasons

        # Quality-based filters require quality data
        if eq is None and self._needs_quality_analysis():
            logger.warning(
                "No quality data for %s — skipping quality filters", episode_id
            )
            return True, []

        # Min quality score
        if self.config.min_quality is not None and eq is not None:
            score = eq.overall_score
            if score is not None and score < self.config.min_quality:
                reasons.append(f"score {score:.1f} < min {self.config.min_quality}")

        # Exclude flags
        if self.config.exclude_flags is not None and eq is not None:
            for flag in self.config.exclude_flags:
                if flag in eq.flags:
                    reasons.append(f"flag: {flag}")

        # Video (Tier 0) numeric filters
        self._evaluate_video(episode_id, eq, reasons)

        if reasons:
            return False, reasons

        return True, []

    def _evaluate_video(
        self, episode_id: str, eq: EpisodeQuality | None, reasons: list[str]
    ) -> None:
        """Append exclusion reasons for any failed Tier 0 video threshold."""
        c = self.config
        video_criteria = (
            c.min_sharpness,
            c.max_frozen_fraction,
            c.max_overexposed_fraction,
            c.max_underexposed_fraction,
        )
        if all(v is None for v in video_criteria):
            return

        vq = eq.video if eq is not None else None
        if vq is None:
            logger.warning(
                "No video metrics for %s — skipping video filters "
                "(run quality with --video, or a report that has video fields)",
                episode_id,
            )
            return

        if (
            c.min_sharpness is not None
            and vq.min_sharpness is not None
            and vq.min_sharpness < c.min_sharpness
        ):
            reasons.append(f"sharpness {vq.min_sharpness:.0f} < min {c.min_sharpness:g}")
        if (
            c.max_frozen_fraction is not None
            and vq.frozen_fraction is not None
            and vq.frozen_fraction > c.max_frozen_fraction
        ):
            reasons.append(
                f"frozen {vq.frozen_fraction:.2f} > max {c.max_frozen_fraction:g}"
            )
        if (
            c.max_overexposed_fraction is not None
            and vq.overexposed_fraction is not None
            and vq.overexposed_fraction > c.max_overexposed_fraction
        ):
            reasons.append(
                f"overexposed {vq.overexposed_fraction:.2f} > max {c.max_overexposed_fraction:g}"
            )
        if (
            c.max_underexposed_fraction is not None
            and vq.underexposed_fraction is not None
            and vq.underexposed_fraction > c.max_underexposed_fraction
        ):
            reasons.append(
                f"underexposed {vq.underexposed_fraction:.2f} > max {c.max_underexposed_fraction:g}"
            )
