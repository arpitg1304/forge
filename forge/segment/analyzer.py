"""Segment analyzer — runs PELT changepoint detection on episode signals."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from forge.segment.config import SegmentConfig
from forge.segment.models import EpisodeSegmentation, Segment, SegmentationReport

logger = logging.getLogger(__name__)

# Signal name → Frame attribute mapping
_SIGNAL_MAP: dict[str, str] = {
    "observation.state": "state",
    "state": "state",
    "qpos": "joint_positions",
    "joint_positions": "joint_positions",
    "joint_velocities": "joint_velocities",
    "action": "action",
}


def _ensure_ruptures():
    """Lazy-import guard for ruptures."""
    try:
        import ruptures  # noqa: F401

        return ruptures
    except ImportError:
        from forge.core.exceptions import MissingDependencyError

        raise MissingDependencyError(
            dependency="ruptures",
            feature="segment",
            install_hint="pip install forge-robotics[segment]",
        )


def _resolve_penalty(penalty_str: str, n_samples: int, n_features: int) -> float:
    """Resolve a penalty string to a numeric value.

    Args:
        penalty_str: "bic", "aic", or a numeric string.
        n_samples: Number of time steps.
        n_features: Signal dimensionality.

    Returns:
        Numeric penalty value.
    """
    lower = penalty_str.strip().lower()
    if lower == "bic":
        return np.log(n_samples) * n_features
    if lower == "aic":
        return 2.0 * n_features
    try:
        return float(penalty_str)
    except ValueError:
        raise ValueError(
            f"Invalid penalty '{penalty_str}'. Use 'bic', 'aic', or a numeric value."
        )


class SegmentAnalyzer:
    """Runs PELT changepoint detection on episode proprioception signals.

    Usage::

        analyzer = SegmentAnalyzer(penalty="bic", cost_model="rbf")
        report = analyzer.segment_dataset("./bridge_v2")
    """

    def __init__(self, config: SegmentConfig | None = None, **kwargs) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SegmentConfig(**kwargs)

    def segment_episode_arrays(
        self,
        episode_id: str,
        signal: np.ndarray,
        signal_name: str = "",
        fps: float | None = None,
    ) -> EpisodeSegmentation:
        """Run PELT on a raw numpy signal array.

        Args:
            episode_id: Episode identifier.
            signal: Shape (T, D) signal array.
            signal_name: Name of the signal used.
            fps: Frames per second (for duration computation).

        Returns:
            EpisodeSegmentation with detected changepoints and segments.
        """
        n_samples, n_features = signal.shape

        result = EpisodeSegmentation(
            episode_id=episode_id,
            num_frames=n_samples,
            signal_name=signal_name,
            signal_dim=n_features,
            fps=fps,
        )

        if n_samples < self.config.min_frames:
            # Too short — return single segment covering the whole episode
            seg = Segment(
                start=0,
                end=n_samples,
                duration_frames=n_samples,
                duration_seconds=n_samples / fps if fps else None,
            )
            result.segments = [seg]
            result.num_segments = 1
            return result

        # Clean NaN/Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional z-score normalization per dimension
        if self.config.normalize:
            mean = signal.mean(axis=0)
            std = signal.std(axis=0)
            signal = (signal - mean) / (std + 1e-8)

        # Run PELT
        ruptures = _ensure_ruptures()
        algo = ruptures.Pelt(
            model=self.config.cost_model,
            min_size=self.config.min_segment_length,
            jump=1,
        )
        algo.fit(signal)

        pen = _resolve_penalty(self.config.penalty, n_samples, n_features)
        breakpoints = algo.predict(pen=pen)
        # ruptures returns breakpoints including the final index (n_samples)
        # e.g. [45, 120, 200] where 200 == n_samples

        # Build segments from breakpoints
        changepoints = [bp for bp in breakpoints if bp < n_samples]
        boundaries = [0] + changepoints + [n_samples]
        # Deduplicate in case n_samples was already in changepoints
        boundaries = sorted(set(boundaries))

        segments: list[Segment] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            dur_frames = end - start
            dur_sec = dur_frames / fps if fps else None
            segments.append(Segment(
                start=start,
                end=end,
                duration_frames=dur_frames,
                duration_seconds=round(dur_sec, 3) if dur_sec is not None else None,
            ))

        result.changepoints = changepoints
        result.segments = segments
        result.num_segments = len(segments)

        # Label segments from signal alone (no gripper) when using array API
        if self.config.label_phases:
            self._label_segments(result, signal)

        return result

    def _extract_signal(
        self, episode, extract_gripper: bool = False
    ) -> tuple[np.ndarray, str, np.ndarray | None]:
        """Extract the configured signal from an Episode as a (T, D) array.

        Args:
            episode: A forge.core.models.Episode object.
            extract_gripper: Also extract gripper state in the same pass.

        Returns:
            Tuple of (signal_array, actual_signal_name, gripper_array_or_None).

        Raises:
            ValueError: If the signal cannot be found in any frame.
        """
        attr = _SIGNAL_MAP.get(self.config.signal)
        if attr is None:
            raise ValueError(
                f"Unknown signal '{self.config.signal}'. "
                f"Supported: {list(_SIGNAL_MAP.keys())}"
            )

        fallback_attr = None
        if self.config.signal in ("observation.state", "state"):
            fallback_attr = "joint_positions"

        signal_list: list[np.ndarray] = []
        gripper_list: list[float] = []
        gripper_available = extract_gripper
        actual_name = self.config.signal

        for frame in episode.frames():
            val = getattr(frame, attr, None)

            # Fallback for observation.state → joint_positions
            if val is None and fallback_attr is not None:
                val = getattr(frame, fallback_attr, None)
                if val is not None:
                    actual_name = fallback_attr

            if val is not None:
                signal_list.append(np.asarray(val, dtype=np.float64))

            # Extract gripper in same pass
            if gripper_available:
                g = frame.gripper_state
                if g is None:
                    g = frame.action_gripper
                if g is None:
                    gripper_available = False
                else:
                    gripper_list.append(float(g))

        if not signal_list:
            raise ValueError(
                f"Signal '{self.config.signal}' not found in episode '{episode.episode_id}'. "
                f"Frames have no '{attr}' data."
            )

        gripper = None
        if gripper_available and gripper_list:
            gripper = np.array(gripper_list, dtype=np.float64)

        return np.stack(signal_list), actual_name, gripper

    def _label_segments(
        self,
        result: EpisodeSegmentation,
        signal: np.ndarray,
        frames: list | None = None,
    ) -> None:
        """Apply semantic phase labels to segments if configured."""
        if not self.config.label_phases or not result.segments:
            return

        from forge.segment.labeler import PhaseLabeler, extract_gripper_signal

        gripper = None
        if frames is not None:
            gripper = extract_gripper_signal(frames)
            if gripper is None:
                logger.debug(
                    "No gripper data for %s — labeling without gripper",
                    result.episode_id,
                )

        labeler = PhaseLabeler()
        labeler.label_segments(result.segments, signal, gripper)

    def segment_episode(self, episode) -> EpisodeSegmentation:
        """Segment a Forge Episode object.

        Extracts the configured signal and delegates to segment_episode_arrays.

        Args:
            episode: A forge.core.models.Episode object.

        Returns:
            EpisodeSegmentation with detected changepoints.
        """
        try:
            signal, actual_name, gripper = self._extract_signal(
                episode, extract_gripper=self.config.label_phases
            )
        except ValueError as e:
            logger.warning("Skipping episode %s: %s", episode.episode_id, e)
            return EpisodeSegmentation(
                episode_id=episode.episode_id,
                signal_name=self.config.signal,
            )

        result = self.segment_episode_arrays(
            episode_id=episode.episode_id,
            signal=signal,
            signal_name=actual_name,
            fps=episode.fps,
        )

        # Re-label with gripper data if available (overrides signal-only labels)
        if self.config.label_phases and gripper is not None:
            from forge.segment.labeler import PhaseLabeler
            labeler = PhaseLabeler()
            labeler.label_segments(result.segments, signal, gripper)

        return result

    def segment_dataset(
        self,
        path: str | Path,
        format: str | None = None,
        sample: int = 0,
        progress_callback=None,
    ) -> SegmentationReport:
        """Segment an entire dataset.

        Args:
            path: Local path or HF URL.
            format: Force format (auto-detect if None).
            sample: Segment only N episodes (0 = all).
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            SegmentationReport with per-episode and aggregate results.
        """
        from forge.formats.registry import FormatRegistry
        from forge.io import is_remote_uri, localize

        if is_remote_uri(path):
            # Cloud URI (s3://, gs://) — download to a temp dir.
            resolved = localize(str(path))
        else:
            resolved = Path(path)
            if not resolved.exists():
                from forge.hub.download import download_dataset

                resolved = download_dataset(str(path))

        if format is None:
            format = FormatRegistry.detect_format(resolved)

        reader = FormatRegistry.get_reader(format)
        report = SegmentationReport(
            dataset_path=str(path),
            config=dataclasses.asdict(self.config),
        )

        for i, episode in enumerate(reader.read_episodes(resolved)):
            if sample > 0 and i >= sample:
                break

            es = self.segment_episode(episode)
            report.per_episode.append(es)

            if progress_callback:
                progress_callback(i + 1, sample or 0)

        report.num_episodes = len(report.per_episode)
        report.compute_summary()
        return report
