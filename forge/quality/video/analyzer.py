"""Camera-scoped diagnostics with explicit observation and decode coverage.

Timing describes episode observations, not independent camera capture PTS.
Unobserved samples never become trustworthy pixels or bridge motion estimates.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from forge.quality.video.config import VideoQualityConfig
from forge.quality.video.models import CameraVideoQuality, VideoEvent, VideoQuality
from forge.quality.video.pixel import (
    clip_fractions,
    colorfulness,
    frame_mae,
    laplacian_variance,
    prepare,
)


class _CameraAccumulator:
    def __init__(self, camera, config, fps=None):
        self.camera, self.config = camera, config
        self.fps = config.expected_fps or fps
        self.result = CameraVideoQuality(camera=camera)
        self.result.provenance = {
            "config": asdict(config),
            "expected_fps": self.fps,
            "timestamp_source": "episode observations, not camera capture PTS",
            "pixel_time_support": "analyzed samples only",
        }
        self.attempts = 0
        self.records = []  # (ordinal, timestamp, outcome); metadata on ALL observations
        self.samples = []  # (ordinal, time, sharpness, luma, over, under, color)
        self.prev = None
        self.motion_prev = None
        self.motion = []  # (start, end, start_time, end_time, magnitude/frame, px/s, residual, hist, displacement)
        self.frozen = []
        self._last_events = {}
        self._continuous = True

    def reset(self):
        self.prev = self.motion_prev = None
        self._last_events.clear()
        if self.samples:
            self._continuous = False

    def event(self, kind, start, end, t0, t1, evidence=None, merge=False):
        previous = self._last_events.get(kind)
        if merge and previous:
            previous.end_frame, previous.end_time_s = end, t1
        else:
            previous = VideoEvent(kind, self.camera, start, end, t0, t1, evidence or {})
            self.result.events.append(previous)
        self._last_events[kind] = previous
        return previous

    def record(self, index, timestamp, outcome):
        self.records.append((index, timestamp, outcome))
        self.attempts += outcome in ("decoded", "decode_failure", "invalid")

    def update(self, img, index=None, timestamp=None):
        # Retain the internal direct-update API used by the original audit scripts.
        index = len(self.samples) if index is None else index
        if timestamp is None and self.fps:
            timestamp = index / self.fps
        cfg, cam = self.config, self.result
        gray, rgb = prepare(img, cfg.downscale)  # raises on invalid pixels/unsupported formats
        shape = np.asarray(img).shape
        if cam.height is not None and (cam.height, cam.width) != shape[:2]:
            cam.resolution_changes += 1
            self.event("video_resolution_change", index, index + 1, timestamp, timestamp)
            self.reset()
        cam.height, cam.width = map(int, shape[:2])
        sharp, luma = laplacian_variance(gray), float(gray.mean())
        low, high = clip_fractions(gray, cfg.clip_low, cfg.clip_high)
        over, under = (
            high > cfg.clip_pixel_fraction,
            low > cfg.clip_pixel_fraction or luma < cfg.dark_luma_flag,
        )
        self.samples.append(
            (
                index,
                timestamp,
                sharp,
                luma,
                over,
                under,
                colorfulness(rgb) if rgb is not None else None,
            )
        )
        cam.num_frames += 1
        for kind, flagged in (
            ("video_blur", sharp < cfg.sharpness_flag),
            ("video_overexposure", over),
            ("video_underexposure", under),
        ):
            if flagged:
                previous = self._last_events.get(kind)
                adjacent = (
                    previous is not None and index - (previous.end_frame - 1) == cfg.sample_stride
                )
                # With sparse analysis these spans identify sampled evidence only.
                self.event(
                    kind,
                    index,
                    index + 1,
                    timestamp,
                    timestamp,
                    {"sample_stride": cfg.sample_stride, "support": "sampled pixels"},
                    merge=adjacent,
                )
            else:
                self._last_events.pop(kind, None)
        if self.prev is not None:
            pi, pt, pg = self.prev
            if pg.shape == gray.shape:
                is_frozen = frame_mae(gray, pg) < cfg.frozen_mae
                self.frozen.append((pi, index, pt, timestamp, is_frozen))
                if is_frozen:
                    self.event(
                        "video_frozen_candidate",
                        pi,
                        index + 1,
                        pt,
                        timestamp,
                        {
                            "sample_stride": cfg.sample_stride,
                            "interpretation": "near-identical samples; stationary scenes also qualify",
                        },
                        merge=True,
                    )
                else:
                    self._last_events.pop("video_frozen_candidate", None)
        self.prev = index, timestamp, gray
        if cfg.level == "motion":
            self._update_motion(img, index, timestamp)

    def _update_motion(self, img, index, timestamp):
        from forge.quality.video.motion import (
            affine_design,
            camera_scene_split,
            dense_flow,
            flow_magnitude,
            histogram_correlation,
            to_uint8_gray,
        )

        gray, _ = prepare(img, self.config.motion_downscale)
        cur = to_uint8_gray(gray)
        if cur.std() < 1:
            self.motion_prev = None  # Never bridge a blank/low-texture observation.
            self._continuous = False
            return
        prev = self.motion_prev
        self.motion_prev = index, timestamp, cur
        if prev is None or prev[2].shape != cur.shape:
            return
        pi, pt, pg = prev
        elapsed = timestamp - pt if timestamp is not None and pt is not None else None
        flow = dense_flow(pg, cur)
        magnitude = float(flow_magnitude(flow).mean())
        resid, total = camera_scene_split(flow, affine_design(*cur.shape))
        frames = elapsed * self.fps if elapsed is not None and self.fps else index - pi
        self.motion.append(
            (
                pi,
                index,
                pt,
                timestamp,
                magnitude / frames,
                magnitude / elapsed if elapsed is not None else None,
                resid / (total + 1e-6),
                histogram_correlation(pg, cur),
                (float(flow[..., 0].mean()), float(flow[..., 1].mean())),
            )
        )

    def finalize(self):
        cam, cfg = self.result, self.config
        n = len(self.records)
        if n:
            counts = {
                kind: sum(r[2] == kind for r in self.records)
                for kind in ("missing", "decoded", "decode_failure", "invalid", "stride", "cap")
            }
            cam.expected_frames = n
            cam.missing_frames = counts["missing"]
            cam.present_frames = n - cam.missing_frames
            cam.decode_failures, cam.invalid_frames = counts["decode_failure"], counts["invalid"]
            cam.decode_attempts = counts["decoded"] + cam.decode_failures + cam.invalid_frames
            cam.sampled_out_frames, cam.capped_frames = counts["stride"], counts["cap"]
            cam.missing_fraction = cam.missing_frames / n
            cam.decode_coverage = (
                cam.num_frames / cam.present_frames if cam.present_frames else None
            )
            # Collapse integrity failures only across adjacent observation ordinals.
            for outcome, kind in (
                ("missing", "video_camera_missing"),
                ("decode_failure", "video_decode_failure"),
                ("invalid", "video_invalid_pixels"),
            ):
                active = None
                for index, timestamp, status in self.records:
                    if status == outcome:
                        if active is not None:
                            active.end_frame, active.end_time_s = index + 1, timestamp
                            active.evidence["observations"] += 1
                        else:
                            active = VideoEvent(
                                kind,
                                self.camera,
                                index,
                                index + 1,
                                timestamp,
                                timestamp,
                                {"observations": 1},
                            )
                            cam.events.append(active)
                    else:
                        active = None
            for count, flag in (
                (cam.missing_frames, "video_camera_missing"),
                (cam.decode_failures, "video_decode_failure"),
                (cam.invalid_frames, "video_invalid_pixels"),
                (cam.resolution_changes, "video_resolution_change"),
            ):
                if count:
                    cam.flags.append(flag)
        cam.metric_status["coverage"] = (
            "decode_coverage = valid decoded / present references; stride and cap are reported separately"
        )
        cam.metric_status["pixels"] = (
            "available: sampled whole-image heuristics; calibrate per camera and resolution"
            if self.samples
            else "unavailable: no valid decoded samples"
        )
        cam.metric_status["frozen"] = "near-identical sampled images; not proof of an encoder stall"
        cam.metric_status["motion"] = (
            "not_requested"
            if cfg.level != "motion"
            else (
                "available: approximate global-affine decomposition"
                if self.motion
                else "unavailable: no supported textured frame pairs"
            )
        )
        cam.metric_status["motion_smoothness"] = (
            "unavailable: insufficient, interrupted, or irregular flow sequence"
        )
        if self.samples:
            sharp = [s[2] for s in self.samples]
            cam.mean_sharpness, cam.min_sharpness = float(np.mean(sharp)), float(min(sharp))
            cam.blurry_fraction = float(np.mean(np.asarray(sharp) < cfg.sharpness_flag))
            cam.mean_luma = float(np.mean([s[3] for s in self.samples]))
            cam.overexposed_fraction, cam.underexposed_fraction = [
                float(np.mean([s[i] for s in self.samples])) for i in (4, 5)
            ]
            colors = [s[6] for s in self.samples if s[6] is not None]
            cam.mean_colorfulness = float(np.mean(colors)) if colors else None
            for value, threshold, flag in (
                (cam.blurry_fraction, cfg.blurry_fraction_flag, "blurry"),
                (cam.overexposed_fraction, cfg.exposure_flag, "over_exposed"),
                (cam.underexposed_fraction, cfg.exposure_flag, "under_exposed"),
            ):
                if value > threshold:
                    cam.flags.append(flag)
        if self.frozen:
            cam.frozen_fraction = float(np.mean([r[4] for r in self.frozen]))
            run, last_end = 0, None
            for start, end, _, _, frozen in self.frozen:
                run = (run + 1 if last_end == start else 1) if frozen else 0
                cam.longest_frozen_run = max(cam.longest_frozen_run, run)
                last_end = end
            spans = [e for e in cam.events if e.kind == "video_frozen_candidate"]
            durations = [
                e.end_time_s - e.start_time_s
                for e in spans
                if e.start_time_s is not None and e.end_time_s is not None
            ]
            cam.max_frozen_duration_s = (
                max(durations, default=0.0) if any(s[1] is not None for s in self.samples) else None
            )
            long = (
                cam.max_frozen_duration_s >= cfg.frozen_duration_s
                if cam.max_frozen_duration_s is not None
                else cam.longest_frozen_run > cfg.frozen_run_flag
            )
            if long or cam.frozen_fraction > cfg.frozen_fraction_flag:
                cam.flags.append("frozen_frames")
        if self.motion:
            mags = np.array([r[4] for r in self.motion])
            cam.mean_motion = float(mags.mean())
            per_second = [r[5] for r in self.motion if r[5] is not None]
            cam.mean_motion_px_s = float(np.mean(per_second)) if per_second else None
            cam.scene_motion_fraction = float(np.mean([r[6] for r in self.motion]))
            threshold = max(cfg.cut_flow_factor * float(np.median(mags)), cfg.cut_flow_min)
            cuts = [r for r in self.motion if r[7] < cfg.cut_hist_corr and r[4] > threshold]
            cam.cut_count = len(cuts)
            for r in cuts:
                cam.events.append(
                    VideoEvent("video_cut_candidate", self.camera, r[0], r[1] + 1, r[2], r[3])
                )
            if cuts:
                cam.flags.append("cut_detected")
            times = [r[3] - r[2] for r in self.motion if r[2] is not None and r[3] is not None]
            regular = not times or (
                len(times) == len(self.motion) and np.std(times) / np.mean(times) < 0.1
            )
            if self._continuous and regular and len(self.motion) >= 3:
                from forge.quality.metrics import log_dimensionless_jerk

                path = np.cumsum([r[8] for r in self.motion], axis=0)
                cam.motion_smoothness = log_dimensionless_jerk(
                    path, dt=float(np.mean(times)) if times else 1.0
                )
                cam.metric_status["motion_smoothness"] = (
                    "available: image-space heuristic, not isolated physical camera motion"
                    if cam.motion_smoothness is not None
                    else "unavailable: degenerate image motion"
                )
                if (
                    cam.motion_smoothness is not None
                    and cam.motion_smoothness < cfg.motion_smoothness_flag
                ):
                    cam.flags.append("shaky")
        cam.events.sort(key=lambda e: (e.start_frame, e.kind))
        return cam


class _Accumulators(dict):
    def __init__(self, fps):
        super().__init__()
        self.fps = fps
        self.history = []
        self.all_stored = True
        self.any_stored = False


class VideoQualityAnalyzer:
    def __init__(self, config=None, **kwargs):
        self.config = config if config is not None else VideoQualityConfig(**kwargs)

    def new_accumulators(self, fps=None, cameras=None):
        fps = self.config.expected_fps or fps
        if fps is not None and (not np.isfinite(fps) or fps <= 0):
            raise ValueError("Video expected FPS must be finite and positive")
        accums = _Accumulators(fps)
        for name in self.config.cameras if self.config.cameras is not None else (cameras or []):
            accums[name] = _CameraAccumulator(name, self.config, fps)
        return accums

    def consume_frame(self, frame, accums, frame_index):
        """Observe EVERY episode frame, including those without camera entries."""
        cfg = self.config
        raw_time = frame.timestamp
        timestamp = float(raw_time) if raw_time is not None and np.isfinite(raw_time) else None
        previous = accums.history[-1] if accums.history else None
        # Only infer times when missing; integrity estimates below require stored time.
        display_time = (
            timestamp if timestamp is not None else frame_index / accums.fps if accums.fps else None
        )
        for name in frame.images:
            if cfg.cameras is not None and name not in cfg.cameras:
                continue
            if name not in accums:
                acc = accums[name] = _CameraAccumulator(name, cfg, accums.fps)
                for i, t, _ in accums.history:
                    acc.record(i, t, "missing")
        accums.history.append((frame_index, display_time, timestamp))
        accums.all_stored &= timestamp is not None
        accums.any_stored |= timestamp is not None
        for name, acc in accums.items():
            cam = acc.result
            cam.provenance["timebase"] = (
                "stored_episode_timestamps" if accums.all_stored else "missing_or_inferred_times"
            )
            bad_time = raw_time is not None and timestamp is None
            gap = False
            if previous is not None:
                if timestamp is None or previous[2] is None:
                    bad_time = True
                else:
                    dt = timestamp - previous[2]
                    bad_time = dt <= 0
                    if accums.fps and dt > cfg.gap_factor / accums.fps:
                        gap = True
            if bad_time:
                # Entirely timestamp-free data can still use a declared FPS grid.
                entirely_absent = not accums.any_stored and raw_time is None
                if not entirely_absent:
                    cam.timestamp_invalid_count += 1
                    if "video_invalid_timestamps" not in cam.flags:
                        cam.flags.append("video_invalid_timestamps")
                    acc.event("video_invalid_timestamps", frame_index, frame_index + 1, None, None)
                if not entirely_absent:
                    acc.reset()
            if gap:
                acc.reset()
            if frame.images.get(name) is None:
                acc.record(frame_index, display_time, "missing")
                acc.reset()
                continue
            if frame_index % cfg.sample_stride:
                acc.record(frame_index, display_time, "stride")
                continue
            if cfg.max_frames and acc.attempts >= cfg.max_frames:
                acc.record(frame_index, display_time, "cap")
                acc.reset()
                continue
            try:
                img = frame.images[name].load()
            except Exception:
                acc.record(frame_index, display_time, "decode_failure")
                acc.reset()
                continue
            try:
                acc.update(img, frame_index, display_time)
            except (ValueError, TypeError, IndexError):
                acc.record(frame_index, display_time, "invalid")
                acc.reset()
                continue
            acc.record(frame_index, display_time, "decoded")
            if bad_time and not entirely_absent:
                acc.reset()

    def finalize(self, accums):
        if not accums:
            return None
        cameras = {name: acc.finalize() for name, acc in accums.items()}
        for cam in cameras.values():
            _finalize_timing(cam, accums.history, accums.fps, self.config)
        return _build_rollup(cameras, self.config)

    def analyze_episode(self, episode):
        accums = self.new_accumulators(episode.fps, episode.cameras)
        for index, frame in enumerate(episode.frames()):
            self.consume_frame(frame, accums, index)
        return self.finalize(accums)


def _finalize_timing(cam, history, fps, config):
    """Use the full metadata timeline, including time before a camera first appears."""
    cam.events = [e for e in cam.events if e.kind != "video_frame_gap"]
    cam.flags = [f for f in cam.flags if f != "video_frame_drops"]
    cam.timestamp_gaps = 0
    cam.max_frame_gap_s = None
    stored = sum(t is not None for _, _, t in history)
    cam.timestamp_coverage = stored / len(history) if history else None
    complete = stored == len(history) and len(history) > 1
    positive = []
    invalid_indices = set()
    dropped = 0
    significant = False
    for previous, current in zip(history, history[1:]):
        i, _, t0 = previous
        j, _, t1 = current
        if t0 is None or t1 is None or t1 <= t0:
            complete = False
            if stored:
                invalid_indices.add(j)
            continue
        dt = t1 - t0
        positive.append(dt)
        if fps and dt > config.gap_factor / fps:
            count = max(0, int(round(dt * fps)) - 1)
            dropped += count
            cam.timestamp_gaps += 1
            cam.max_frame_gap_s = max(cam.max_frame_gap_s or 0.0, dt)
            large = (
                count >= config.significant_drop_frames
                or dt - 1 / fps >= config.significant_gap_s - 1e-9
            )
            significant |= large
            cam.events.append(
                VideoEvent(
                    "video_frame_gap",
                    cam.camera,
                    i,
                    j + 1,
                    t0,
                    t1,
                    {
                        "estimated_missing_observations": count,
                        "gap_s": dt,
                        "significant": large,
                        "source": "episode timestamp gap; not independent camera PTS",
                    },
                )
            )
    if invalid_indices:
        existing = {e.start_frame for e in cam.events if e.kind == "video_invalid_timestamps"}
        for index in sorted(invalid_indices - existing):
            cam.events.append(
                VideoEvent("video_invalid_timestamps", cam.camera, index, index + 1, None, None)
            )
        cam.timestamp_invalid_count = len(existing | invalid_indices)
        if "video_invalid_timestamps" not in cam.flags:
            cam.flags.append("video_invalid_timestamps")
    if positive:
        cam.effective_fps = 1 / float(np.mean(positive))
        cam.timestamp_jitter_ratio = float(np.std(positive) / np.mean(positive))
    cam.estimated_dropped_frames = dropped if fps and len(positive) else None
    cam.estimated_drop_fraction = dropped / (len(positive) + dropped) if complete and fps else None
    if significant or (
        cam.estimated_drop_fraction is not None
        and cam.estimated_drop_fraction > config.drop_fraction_flag
    ):
        cam.flags.append("video_frame_drops")
    cam.metric_status["frame_drops"] = (
        "available: estimated missing observation intervals from episode timestamps and declared FPS; uniform rate mismatch can also explain gaps; camera capture drops may be invisible"
        if complete and fps
        else "partial: counts are a lower bound over supported timestamp pairs; unknown intervals omitted"
        if fps and positive
        else "unavailable: at least two stored timestamps and declared FPS required"
    )
    if (stored and not complete) or cam.timestamp_invalid_count:
        for e in cam.events:
            if e.kind != "video_frame_gap":
                e.start_time_s = e.end_time_s = None
    cam.provenance["timebase"] = (
        "stored_episode_timestamps"
        if complete
        else "inferred_from_fps"
        if not stored and fps and not cam.timestamp_invalid_count
        else "incomplete_or_unknown"
    )
    cam.provenance["camera_reference_expectation"] = (
        "selected or discovered cameras are expected on each episode observation; asynchronous schemas require interpretation"
    )
    cam.events.sort(key=lambda e: (e.start_frame, e.kind))


def _build_rollup(
    per_camera: dict[str, CameraVideoQuality],
    config: VideoQualityConfig,
) -> VideoQuality:
    """Combine per-camera results into episode-level worst-case rollups."""
    vq = VideoQuality(
        per_camera=per_camera,
        provenance={
            "config": asdict(config),
            "aggregation": "worst-camera defect fractions; inspect camera relevance before filtering",
        },
    )

    def _vals(attr: str) -> list[float]:
        return [getattr(c, attr) for c in per_camera.values() if getattr(c, attr) is not None]

    mean_sharp = _vals("mean_sharpness")
    if mean_sharp:
        vq.min_sharpness = float(min(_vals("min_sharpness")))
        vq.mean_sharpness = float(np.mean(mean_sharp))

    blurry = _vals("blurry_fraction")
    if blurry:
        vq.blurry_fraction = float(max(blurry))
    over = _vals("overexposed_fraction")
    if over:
        vq.overexposed_fraction = float(max(over))
    under = _vals("underexposed_fraction")
    if under:
        vq.underexposed_fraction = float(max(under))
    frozen = _vals("frozen_fraction")
    if frozen:
        vq.frozen_fraction = float(max(frozen))

    vq.max_frozen_run = max((c.longest_frozen_run for c in per_camera.values()), default=0)

    color = _vals("mean_colorfulness")
    if color:
        vq.mean_colorfulness = float(np.mean(color))

    # ── Tier 1 motion rollups ──
    motion = _vals("mean_motion")
    if motion:
        # Most-active camera: an episode "has motion" if any camera does.
        vq.mean_motion = float(max(motion))
    smooth = _vals("motion_smoothness")
    if smooth:
        vq.motion_smoothness = float(min(smooth))  # most negative = jerkiest
    scene = _vals("scene_motion_fraction")
    if scene:
        vq.scene_motion_fraction = float(max(scene))
    cuts = _vals("cut_count")
    if cuts:
        vq.cut_count = int(max(cuts))

    # Union of per-camera flags (a flag fires if any camera raised it).
    seen: set[str] = set()
    for cam in per_camera.values():
        for f in cam.flags:
            if f not in seen:
                seen.add(f)
                vq.flags.append(f)

    # 'no_motion' is episode-level: flag only if the most-active camera is below
    # the threshold (so a single moving camera keeps the episode "in motion").
    if (
        len(motion) == len(per_camera)
        and vq.mean_motion is not None
        and vq.mean_motion < config.motion_low_flag
    ):
        if "no_motion" not in seen:
            vq.flags.append("no_motion")

    return vq
