"""Quality analyzer — orchestrates metrics across episodes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from forge.quality.config import QualityConfig
from forge.quality.metrics import (
    action_entropy,
    action_saturation,
    composite_score,
    dead_action_detection,
    gripper_chatter,
    log_dimensionless_jerk,
    psd_band_energy,
    spectral_arc_length,
    state_conditioned_action_variance,
    timestamp_regularity,
)
from forge.quality.models import EpisodeQuality, QualityReport


class QualityAnalyzer:
    """Orchestrates quality metrics across a dataset.

    Usage::

        analyzer = QualityAnalyzer(fps=30.0)
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
        from forge.quality.diagnostics import diagnose, indices

        eq = EpisodeQuality(episode_id=episode_id)
        effective_fps = self.config.fps if fps is None else fps
        if not np.isfinite(effective_fps) or effective_fps <= 0:
            raise ValueError("fps must be finite and positive")
        eq.provenance = {
            "config": asdict(self.config),
            "fps": effective_fps,
            "timebase": "stored" if timestamps is not None else "inferred_from_fps",
        }
        arrays = {}
        for name, value in (("actions", actions), ("states", states), ("timestamps", timestamps)):
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float64)
            expected_ndim = 1 if name == "timestamps" else 2
            if value.ndim != expected_ndim or (value.ndim == 2 and value.shape[1] == 0):
                raise ValueError(
                    f"{name} must have {'(T,)' if expected_ndim == 1 else '(T, D)'} shape"
                )
            arrays[name] = value
        lengths = {len(x) for x in arrays.values()}
        if len(lengths) > 1:
            raise ValueError("actions, states, and timestamps must have matching frame counts")
        eq.num_frames = next(iter(lengths), 0)
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                eq.flags.append(f"nan_in_{name}")
                eq.metric_status[name] = (
                    "invalid: nonfinite or missing samples; no zero substitution"
                )
        if eq.flags:
            eq.metric_status["score"] = "unavailable: invalid input"
            return eq
        if eq.num_frames < self.config.min_frames:
            eq.metric_status["score"] = "unavailable: insufficient frames"
            return eq
        actions, states = arrays.get("actions"), arrays.get("states")
        timestamps = arrays.get("timestamps")
        t = np.arange(eq.num_frames) / effective_fps if timestamps is None else timestamps
        if np.any(np.diff(t) <= 0):
            eq.flags.append("invalid_timestamps")
            eq.metric_status["score"] = "unavailable: timestamps must strictly increase"
            return eq
        dt = float(np.mean(np.diff(t)))
        duration = float(t[-1] - t[0])
        eq.diagnostics, eq.events, statuses, eq.static = diagnose(
            actions, states, t, self.config, effective_fps
        )
        eq.metric_status.update(statuses)
        regular = True
        if timestamps is not None:
            eq.timestamps = timestamp_regularity(t, effective_fps)
            regular = (
                eq.timestamps.num_gaps == 0
                and eq.timestamps.jitter_ratio <= self.config.jitter_warn_ratio
            )
            if eq.timestamps.num_gaps:
                eq.flags.append("timestamp_gaps")
            if abs(eq.timestamps.effective_fps / effective_fps - 1) > self.config.jitter_warn_ratio:
                eq.flags.append("sample_rate_mismatch")
        eq.metric_status["uniform_sampling"] = (
            "available"
            if regular
            else "unavailable: irregular timestamps; spectral/derivative metrics skipped"
        )
        if actions is not None:
            eq.dead_fraction, eq.dead_ranges = dead_action_detection(actions, self.config)
            eq.metric_status["dead_actions"] = (
                "diagnostic: zero/repeated commands can be intentional"
            )
            dims = self.config.gripper_dims
            if dims is None:
                dims = () if self.config.gripper_dim is None else (self.config.gripper_dim,)
            dims = indices(dims, actions.shape[1], "gripper_dims")
            for dim in dims:
                count, rate, flagged = gripper_chatter(
                    actions, duration, replace(self.config, gripper_dims=None, gripper_dim=dim)
                )
                eq.grippers[str(dim)] = {"transitions": count, "rate": rate, "is_chattery": flagged}
            if eq.grippers:
                eq.gripper_chatter_count = sum(g["transitions"] for g in eq.grippers.values())
                eq.gripper_chatter_rate = max(g["rate"] for g in eq.grippers.values())
                eq.is_chattery = any(g["is_chattery"] for g in eq.grippers.values())
            eq.metric_status["gripper"] = (
                "available: explicit channels"
                if dims
                else "not_applicable: no gripper channels configured"
            )
            sat, overall, saturated_dims = action_saturation(actions, self.config)
            if self.config.action_bounds is not None or self.config.action_limits is not None:
                eq.saturation_per_dim = sat.tolist()
                eq.overall_saturation = overall
                eq.saturated_dims = saturated_dims
                eq.metric_status["saturation"] = "available: supplied command limits"
            else:
                eq.observed_range_edge_fraction = overall
                eq.metric_status["saturation"] = (
                    "unavailable: no supplied limits; observed_range_edge_fraction is descriptive only"
                )
            entropy, eq.mean_entropy = action_entropy(actions)
            eq.entropy_per_dim = entropy.tolist()
            if regular:
                bands = psd_band_energy(
                    actions, 1 / dt, low_hz=self.config.psd_low_hz, high_hz=self.config.psd_high_hz
                )
                if bands is not None:
                    eq.psd_bands = bands
                    eq.psd_high_fraction = bands["high_fraction"]
                    if eq.psd_high_fraction > self.config.psd_high_flag:
                        eq.flags.append("psd_high_band_chatter")
            eq.metric_status["psd"] = (
                "available"
                if eq.psd_bands is not None
                else "unavailable: sample rate or regularity cannot support configured bands"
            )
        # Never substitute command motion for a supplied stationary state trace.
        positions = states
        if states is not None:
            dims = indices(
                self.config.state_dims
                if self.config.state_dims is not None
                else range(states.shape[1]),
                states.shape[1],
                "state_dims",
            )
            scales = (
                np.ones(states.shape[1])
                if self.config.state_scales is None
                else np.asarray(self.config.state_scales)
            )
            positions = states[:, dims] / scales[dims]
            eq.joint_path_length = eq.diagnostics["path_length_scaled"]
        elif self.config.action_semantics == "position":
            positions = actions
        if positions is not None and regular:
            eq.ldlj = log_dimensionless_jerk(positions, dt)
            eq.sparc = spectral_arc_length(positions, dt)
        eq.metric_status["smoothness"] = (
            (
                "available: state positions"
                if states is not None
                else "available: commanded positions"
            )
            if eq.ldlj is not None
            else "unavailable: no motion, no positional data, or irregular timing"
        )
        if states is not None and actions is not None:
            _, eq.state_action_consistency = state_conditioned_action_variance(
                states, actions, k=self.config.state_action_k
            )
            eq.metric_status["state_action_consistency"] = (
                "diagnostic: unnormalized within-episode neighbors; not a task-correctness score"
            )
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
        if any(e.kind == "tracking_stall_candidate" for e in eq.events):
            eq.flags.append("tracking_stall_candidate")
        eq.metric_status["score"] = (
            "heuristic: not task success or training utility; missing components omitted"
        )
        if not eq.subscores or not regular or (states is not None and eq.ldlj is None):
            eq.overall_score = None
            eq.metric_status["score"] = (
                "unavailable: no evaluable motion or irregular timebase; review diagnostics"
            )
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
            self._video_analyzer.new_accumulators(episode.fps, episode.cameras)
            if self._video_analyzer is not None
            else None
        )

        for frame_index, frame in enumerate(episode.frames()):
            actions_list.append(
                None if frame.action is None else np.asarray(frame.action, dtype=np.float64)
            )
            states_list.append(
                None if frame.state is None else np.asarray(frame.state, dtype=np.float64)
            )
            timestamps_list.append(frame.timestamp)
            if video_accums is not None:
                self._video_analyzer.consume_frame(frame, video_accums, frame_index)

        def aligned(values):
            example = next((v for v in values if v is not None), None)
            if example is None:
                return None
            return np.stack([np.full_like(example, np.nan) if v is None else v for v in values])

        actions = aligned(actions_list)
        states = aligned(states_list)
        timestamps = (
            np.array([np.nan if t is None else t for t in timestamps_list])
            if any(t is not None for t in timestamps_list)
            else None
        )

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
        from forge.io import is_remote_uri, localize

        if is_remote_uri(path):
            # Cloud URI (s3://, gs://) — download to a temp dir.
            resolved = localize(str(path))
        else:
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
            scores = [eq.overall_score for eq in report.per_episode if eq.overall_score is not None]
            report.overall_score = float(np.mean(scores)) if scores else None

            # Aggregate subscores
            all_subscore_keys = set()
            for eq in report.per_episode:
                all_subscore_keys.update(eq.subscores.keys())

            for key in all_subscore_keys:
                vals = [eq.subscores[key] for eq in report.per_episode if key in eq.subscores]
                if vals:
                    report.subscores[key] = float(np.mean(vals))

        # Generate flags and recommendations
        report.flags, report.recommendations = _generate_recommendations(report, self.config)

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
        flags.append(f"{n} episodes with gripper chatter (rate > {config.chatter_threshold}/sec)")
        recs.append(
            f"Review gripper threshold or filter: forge filter --max-gripper-chatter {config.chatter_threshold}"
        )

    if "mostly_static" in flagged:
        n = len(flagged["mostly_static"])
        flags.append(f"{n} episodes with sustained position holds (>50% observed time)")
        recs.append("Review motion-pause events; waiting and contact may be intentional.")

    if "dead_actions" in flagged:
        n = len(flagged["dead_actions"])
        flags.append(f"{n} episodes with zero/repeated commands (>10% of frames)")

    if "timestamp_jitter" in flagged:
        n = len(flagged["timestamp_jitter"])
        flags.append(f"{n} episodes with timestamp jitter")

    if "low_entropy" in flagged:
        n = len(flagged["low_entropy"])
        flags.append(f"{n} episodes with low action diversity")

    if "saturated" in flagged:
        n = len(flagged["saturated"])
        flags.append(f"{n} episodes with action saturation (>30% at bounds)")

    if "tracking_stall_candidate" in flagged:
        flags.append(
            f"{len(flagged['tracking_stall_candidate'])} episodes with position-tracking stall candidates"
        )
        recs.append(
            "Review command/state mapping, controller latency, and contact before rejecting data."
        )

    if "psd_high_band_chatter" in flagged:
        n = len(flagged["psd_high_band_chatter"])
        flags.append(
            f"{n} episodes with high-frequency action chatter "
            f"(>{int(config.psd_high_flag * 100)}% energy above {config.psd_high_hz:g} Hz)"
        )

    if report.overall_score is not None and report.overall_score < 6.0 and report.num_episodes > 0:
        recs.append(
            f"Dataset quality is low ({report.overall_score:.1f}/10). Consider: forge filter --min-quality 6.0"
        )

    if recs:
        recs.append("Export flagged IDs: forge quality <path> --export report.json")

    return flags, recs
