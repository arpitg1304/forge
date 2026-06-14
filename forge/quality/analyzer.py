"""Quality analyzer — orchestrates metrics across episodes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from forge.quality.config import QualityConfig
from forge.quality.metrics import (
    action_entropy,
    action_saturation,
    composite_score,
    dead_action_detection,
    gripper_chatter,
    log_dimensionless_jerk,
    path_length,
    psd_band_energy,
    spectral_arc_length,
    state_conditioned_action_variance,
    static_detection,
    timestamp_regularity,
)
from forge.quality.models import EpisodeQuality, QualityReport


class QualityAnalyzer:
    """Orchestrates quality metrics across a dataset.

    Usage::

        analyzer = QualityAnalyzer(gripper_dim=-1, fps=30.0)
        report = analyzer.analyze_dataset("./bridge_v2")
        print(report.overall_score)
    """

    def __init__(
        self,
        config: QualityConfig | None = None,
        video_config=None,
        **kwargs,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = QualityConfig(**kwargs)

        # Optional Tier 0 video analysis. Left None for the proprio-only default
        # path so importing/using quality never touches the video module.
        self._video_analyzer = None
        if video_config is not None:
            from forge.quality.video import VideoQualityAnalyzer

            self._video_analyzer = VideoQualityAnalyzer(config=video_config)

    def analyze_episode_arrays(
        self,
        episode_id: str,
        actions: np.ndarray | None = None,
        states: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        fps: float | None = None,
    ) -> EpisodeQuality:
        """Analyze a single episode from raw numpy arrays.

        Args:
            episode_id: Episode identifier.
            actions: Shape (T, D) action array, or None.
            states: Shape (T, D) state array, or None.
            timestamps: Shape (T,) timestamp array, or None.
            fps: FPS override for this episode.

        Returns:
            EpisodeQuality with all computable metrics.
        """
        eq = EpisodeQuality(episode_id=episode_id)
        effective_fps = fps or self.config.fps

        # Determine frame count
        for arr in (actions, states, timestamps):
            if arr is not None:
                eq.num_frames = len(arr)
                break

        if eq.num_frames < self.config.min_frames:
            return eq

        # Compute dt for derivative metrics
        if timestamps is not None and len(timestamps) > 1:
            dt = float(np.mean(np.diff(timestamps)))
        else:
            dt = 1.0 / effective_fps

        duration = (eq.num_frames - 1) * dt

        # ── Action-based metrics ──
        if actions is not None and actions.size > 0:
            # Handle NaN/Inf
            if np.any(~np.isfinite(actions)):
                eq.flags.append("nan_in_actions")
                actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. Dead actions
            eq.dead_fraction, eq.dead_ranges = dead_action_detection(
                actions, self.config
            )

            # 3. Gripper chatter
            if actions.shape[1] > 1:
                eq.gripper_chatter_count, eq.gripper_chatter_rate, eq.is_chattery = (
                    gripper_chatter(actions, duration, self.config)
                )

            # 6. Action saturation
            sat_per_dim, eq.overall_saturation, eq.saturated_dims = action_saturation(
                actions, self.config
            )
            eq.saturation_per_dim = sat_per_dim.tolist()

            # 7. Static detection
            eq.static = static_detection(actions, effective_fps, self.config)

            # 8. Action entropy
            ent_per_dim, eq.mean_entropy = action_entropy(actions)
            eq.entropy_per_dim = ent_per_dim.tolist()

            # 9b. PSD band energy (generalized chatter across all action dims)
            bands = psd_band_energy(
                actions,
                effective_fps,
                low_hz=self.config.psd_low_hz,
                high_hz=self.config.psd_high_hz,
            )
            if bands is not None:
                eq.psd_bands = bands
                eq.psd_high_fraction = bands["high_fraction"]
                if bands["high_fraction"] > self.config.psd_high_flag:
                    eq.flags.append("psd_high_band_chatter")

        # ── State-based metrics ──
        if states is not None and states.size > 0:
            if np.any(~np.isfinite(states)):
                eq.flags.append("nan_in_states")
                states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

            # 2. LDLJ smoothness (joint space)
            eq.ldlj = log_dimensionless_jerk(states, dt)

            # 9a. SPARC smoothness (frequency-domain complement to LDLJ)
            eq.sparc = spectral_arc_length(states, dt)

            # 4. Path length (joint space)
            eq.joint_path_length = path_length(states)

        # If no states but actions available, compute LDLJ + SPARC from actions
        if actions is not None and actions.size > 0:
            if eq.ldlj is None:
                eq.ldlj = log_dimensionless_jerk(actions, dt)
            if eq.sparc is None:
                eq.sparc = spectral_arc_length(actions, dt)

        # 9c. State-conditioned action variance (proprio STAC analogue) —
        # needs both states and actions of matching length.
        if (
            states is not None
            and actions is not None
            and states.size > 0
            and actions.size > 0
            and len(states) == len(actions)
        ):
            _per_frame, mean_var = state_conditioned_action_variance(
                states, actions, k=self.config.state_action_k
            )
            eq.state_action_consistency = mean_var

        # ── Timestamp metrics ──
        if timestamps is not None and len(timestamps) > 1:
            eq.timestamps = timestamp_regularity(timestamps, effective_fps)

        # ── Composite score ──
        eq.overall_score, eq.subscores, score_flags = composite_score(
            dead_fraction=eq.dead_fraction,
            ldlj_score=eq.ldlj,
            is_chattery=eq.is_chattery,
            chatter_rate=eq.gripper_chatter_rate,
            static_fraction=eq.static.static_fraction if eq.static else None,
            jitter_ratio=eq.timestamps.jitter_ratio if eq.timestamps else None,
            overall_saturation=eq.overall_saturation,
            mean_entropy=eq.mean_entropy,
            config=self.config,
        )
        eq.flags.extend(score_flags)

        return eq

    def analyze_episode(self, episode) -> EpisodeQuality:
        """Analyze a Forge Episode object.

        Extracts numpy arrays from frames and delegates to analyze_episode_arrays.

        Args:
            episode: A forge.core.models.Episode object.

        Returns:
            EpisodeQuality with all computable metrics.
        """
        actions_list: list[np.ndarray] = []
        states_list: list[np.ndarray] = []
        timestamps_list: list[float] = []

        # Single pass: collect proprio arrays and, when enabled, feed video
        # frames into per-camera accumulators so the episode decodes only once.
        video_accums = (
            self._video_analyzer.new_accumulators()
            if self._video_analyzer is not None
            else None
        )
        image_frame_idx = -1

        for frame in episode.frames():
            if frame.action is not None:
                actions_list.append(np.asarray(frame.action, dtype=np.float64))
            if frame.state is not None:
                states_list.append(np.asarray(frame.state, dtype=np.float64))
            if frame.timestamp is not None:
                timestamps_list.append(frame.timestamp)
            if video_accums is not None and frame.images:
                image_frame_idx += 1
                self._video_analyzer.consume_frame(frame, video_accums, image_frame_idx)

        actions = np.stack(actions_list) if actions_list else None
        states = np.stack(states_list) if states_list else None
        timestamps = np.array(timestamps_list) if timestamps_list else None

        eq = self.analyze_episode_arrays(
            episode_id=episode.episode_id,
            actions=actions,
            states=states,
            timestamps=timestamps,
            fps=episode.fps,
        )

        if video_accums is not None:
            eq.video = self._video_analyzer.finalize(video_accums)
            if eq.video is not None:
                eq.flags.extend(eq.video.flags)

        return eq

    def analyze_dataset(
        self,
        path: str | Path,
        format: str | None = None,
        sample: int = 0,
        progress_callback=None,
    ) -> QualityReport:
        """Analyze an entire dataset.

        Args:
            path: Local path or HF URL.
            format: Force format (auto-detect if None).
            sample: Analyze only N episodes (0 = all).
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            QualityReport with per-episode and aggregate results.
        """
        from forge.formats.registry import FormatRegistry

        resolved = Path(path)
        if not resolved.exists():
            # Try HF resolution
            from forge.hub.download import download_dataset

            resolved = download_dataset(str(path))

        if format is None:
            format = FormatRegistry.detect_format(resolved)

        reader = FormatRegistry.get_reader(format)
        report = QualityReport(dataset_path=str(path))

        flagged: dict[str, list[str]] = defaultdict(list)

        for i, episode in enumerate(reader.read_episodes(resolved)):
            if sample > 0 and i >= sample:
                break

            eq = self.analyze_episode(episode)
            report.per_episode.append(eq)

            for flag in eq.flags:
                flagged[flag].append(eq.episode_id)

            if progress_callback:
                progress_callback(i + 1, sample or 0)

        report.num_episodes = len(report.per_episode)
        report.flagged_episodes = dict(flagged)

        # Aggregate scores
        if report.per_episode:
            scores = [
                eq.overall_score
                for eq in report.per_episode
                if eq.overall_score is not None
            ]
            report.overall_score = float(np.mean(scores)) if scores else 0.0

            # Aggregate subscores
            all_subscore_keys = set()
            for eq in report.per_episode:
                all_subscore_keys.update(eq.subscores.keys())

            for key in all_subscore_keys:
                vals = [
                    eq.subscores[key]
                    for eq in report.per_episode
                    if key in eq.subscores
                ]
                if vals:
                    report.subscores[key] = float(np.mean(vals))

        # Generate flags and recommendations
        report.flags, report.recommendations = _generate_recommendations(
            report, self.config
        )

        return report


def _generate_recommendations(
    report: QualityReport, config: QualityConfig
) -> tuple[list[str], list[str]]:
    """Generate human-readable flags and actionable recommendations."""
    flags: list[str] = []
    recs: list[str] = []

    flagged = report.flagged_episodes

    if "jerky" in flagged:
        n = len(flagged["jerky"])
        flags.append(f"{n} episodes with jerky actions (LDLJ < -25)")
        ids = ",".join(flagged["jerky"][:5])
        recs.append(f"Filter jerky episodes: forge filter --exclude-episodes {ids}...")

    if "gripper_chatter" in flagged:
        n = len(flagged["gripper_chatter"])
        flags.append(
            f"{n} episodes with gripper chatter (rate > {config.chatter_threshold}/sec)"
        )
        recs.append(
            f"Review gripper threshold or filter: forge filter --max-gripper-chatter {config.chatter_threshold}"
        )

    if "mostly_static" in flagged:
        n = len(flagged["mostly_static"])
        flags.append(f"{n} episodes mostly static (>50% idle)")
        recs.append("Run: forge filter --min-active-fraction 0.5")

    if "dead_actions" in flagged:
        n = len(flagged["dead_actions"])
        flags.append(f"{n} episodes with dead actions (>10% zero-action frames)")

    if "timestamp_jitter" in flagged:
        n = len(flagged["timestamp_jitter"])
        flags.append(f"{n} episodes with timestamp jitter")

    if "low_entropy" in flagged:
        n = len(flagged["low_entropy"])
        flags.append(f"{n} episodes with low action diversity")

    if "saturated" in flagged:
        n = len(flagged["saturated"])
        flags.append(f"{n} episodes with action saturation (>30% at bounds)")

    if "psd_high_band_chatter" in flagged:
        n = len(flagged["psd_high_band_chatter"])
        flags.append(
            f"{n} episodes with high-frequency action chatter "
            f"(>{int(config.psd_high_flag * 100)}% energy above {config.psd_high_hz:g} Hz)"
        )

    if report.overall_score < 6.0 and report.num_episodes > 0:
        recs.append(
            f"Dataset quality is low ({report.overall_score:.1f}/10). Consider: forge filter --min-quality 6.0"
        )

    if recs:
        recs.append("Export flagged IDs: forge quality <path> --export report.json")

    return flags, recs
