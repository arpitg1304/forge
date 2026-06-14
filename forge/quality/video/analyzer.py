"""Video quality analyzer — Tier 0 (pixel statistics).

Streams frames through a per-camera accumulator so the whole episode is scored
in a single decode pass. The accumulator API (``new_accumulators`` /
``consume_frame`` / ``finalize``) lets the proprio analyzer share the same frame
loop instead of decoding the episode twice.
"""

from __future__ import annotations

import numpy as np

from forge.quality.video.config import VideoQualityConfig
from forge.quality.video.models import CameraVideoQuality, VideoQuality
from forge.quality.video.pixel import (
    clip_fractions,
    colorfulness,
    frame_mae,
    laplacian_variance,
    longest_true_run,
    prepare,
)


class _CameraAccumulator:
    """Per-camera streaming accumulator for Tier 0 (and optional Tier 1) metrics."""

    def __init__(self, camera: str, config: VideoQualityConfig) -> None:
        self.camera = camera
        self.config = config
        self.num_frames = 0
        self.width: int | None = None
        self.height: int | None = None

        self._sharp: list[float] = []
        self._luma: list[float] = []
        self._over: list[bool] = []
        self._under: list[bool] = []
        self._color: list[float] = []
        self._frozen: list[bool] = []
        self._prev_gray: np.ndarray | None = None

        # Tier 1 motion state (used only when config.level == "motion")
        self._motion_mag: list[float] = []
        self._global_vel: list[tuple[float, float]] = []
        self._scene_frac: list[float] = []
        self._hist_corr: list[float] = []
        self._prev_motion_u8: np.ndarray | None = None
        self._motion_design: np.ndarray | None = None

    def update(self, img) -> None:
        cfg = self.config
        a = np.asarray(img)
        if self.height is None and a.ndim >= 2:
            self.height = int(a.shape[0])
            self.width = int(a.shape[1])

        gray, rgb_small = prepare(a, cfg.downscale)

        self._sharp.append(laplacian_variance(gray))

        luma = float(gray.mean())
        self._luma.append(luma)
        frac_low, frac_high = clip_fractions(gray, cfg.clip_low, cfg.clip_high)
        self._over.append(frac_high > cfg.clip_pixel_fraction)
        self._under.append(
            frac_low > cfg.clip_pixel_fraction or luma < cfg.dark_luma_flag
        )

        if rgb_small is not None:
            self._color.append(colorfulness(rgb_small))

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self._frozen.append(frame_mae(gray, self._prev_gray) < cfg.frozen_mae)
        self._prev_gray = gray

        if cfg.level == "motion":
            self._update_motion(a)

        self.num_frames += 1

    def _update_motion(self, img) -> None:
        """Compute optical-flow metrics against the previous frame."""
        from forge.quality.video.motion import (
            affine_design,
            camera_scene_split,
            dense_flow,
            flow_magnitude,
            histogram_correlation,
            to_uint8_gray,
        )

        cfg = self.config
        gray_m, _ = prepare(img, cfg.motion_downscale)
        cur = to_uint8_gray(gray_m)
        # Optical flow against a blank/uniform frame is meaningless (and dodges a
        # decoder that hands back the occasional black frame): keep the last good
        # frame as the reference and skip this pair entirely.
        if float(cur.std()) < 1.0:
            return
        prev = self._prev_motion_u8
        self._prev_motion_u8 = cur
        if prev is None or prev.shape != cur.shape:
            return

        stride = max(1, cfg.sample_stride)
        flow = dense_flow(prev, cur)
        # Normalize by stride so magnitude is per original frame, not per sample.
        self._motion_mag.append(float(flow_magnitude(flow).mean()) / stride)
        self._global_vel.append((float(flow[..., 0].mean()), float(flow[..., 1].mean())))
        self._hist_corr.append(histogram_correlation(prev, cur))

        if self._motion_design is None:
            self._motion_design = affine_design(cur.shape[0], cur.shape[1])
        resid, total = camera_scene_split(flow, self._motion_design)
        self._scene_frac.append(resid / (total + 1e-6))

    def finalize(self) -> CameraVideoQuality:
        cfg = self.config
        cam = CameraVideoQuality(
            camera=self.camera,
            num_frames=self.num_frames,
            width=self.width,
            height=self.height,
        )
        if not self._sharp:
            return cam

        sharp = np.asarray(self._sharp)
        cam.mean_sharpness = float(sharp.mean())
        cam.min_sharpness = float(sharp.min())
        cam.blurry_fraction = float(np.mean(sharp < cfg.sharpness_flag))

        cam.mean_luma = float(np.mean(self._luma))
        cam.overexposed_fraction = float(np.mean(self._over))
        cam.underexposed_fraction = float(np.mean(self._under))

        if self._color:
            cam.mean_colorfulness = float(np.mean(self._color))

        if self._frozen:
            cam.frozen_fraction = float(np.mean(self._frozen))
            cam.longest_frozen_run = longest_true_run(self._frozen)

        # ── Tier 0 flags ──
        if cam.blurry_fraction > cfg.blurry_fraction_flag:
            cam.flags.append("blurry")
        if cam.overexposed_fraction > cfg.exposure_flag:
            cam.flags.append("over_exposed")
        if cam.underexposed_fraction > cfg.exposure_flag:
            cam.flags.append("under_exposed")
        if (
            cam.longest_frozen_run > cfg.frozen_run_flag
            or (cam.frozen_fraction is not None and cam.frozen_fraction > cfg.frozen_fraction_flag)
        ):
            cam.flags.append("frozen_frames")

        # ── Tier 1 motion ──
        if self._motion_mag:
            mags = np.asarray(self._motion_mag)
            cam.mean_motion = float(mags.mean())
            cam.scene_motion_fraction = float(np.mean(self._scene_frac))
            # A cut needs both a histogram discontinuity and a flow spike relative
            # to the episode's own motion — neither alone is enough.
            corr = np.asarray(self._hist_corr)
            median_mag = float(np.median(mags)) if mags.size else 0.0
            spike = mags > cfg.cut_flow_factor * max(median_mag, 1e-6)
            cam.cut_count = int(np.count_nonzero((corr < cfg.cut_hist_corr) & spike))
            if len(self._global_vel) >= 3:
                # LDLJ of the cumulative global-motion path (camera trajectory).
                from forge.quality.metrics import log_dimensionless_jerk

                path = np.cumsum(np.asarray(self._global_vel), axis=0)
                cam.motion_smoothness = log_dimensionless_jerk(path, dt=1.0)

            # 'shaky' and 'cut_detected' are per-camera; 'no_motion' is decided at
            # the episode level (rollup), since one moving camera means the
            # episode is not motionless.
            if (
                cam.motion_smoothness is not None
                and cam.motion_smoothness < cfg.motion_smoothness_flag
            ):
                cam.flags.append("shaky")
            if cam.cut_count and cam.cut_count > 0:
                cam.flags.append("cut_detected")

        return cam


class VideoQualityAnalyzer:
    """Computes Tier 0 video quality metrics for an episode.

    Usage::

        analyzer = VideoQualityAnalyzer(VideoQualityConfig(downscale=128))
        video = analyzer.analyze_episode(episode)   # VideoQuality | None
    """

    def __init__(self, config: VideoQualityConfig | None = None, **kwargs) -> None:
        self.config = config if config is not None else VideoQualityConfig(**kwargs)

    def new_accumulators(self) -> dict[str, _CameraAccumulator]:
        return {}

    def consume_frame(
        self,
        frame,
        accums: dict[str, _CameraAccumulator],
        frame_index: int,
    ) -> None:
        """Feed one frame's images into the per-camera accumulators.

        ``frame_index`` is the running index over image-bearing frames, used for
        striding. Decode failures on a single camera are skipped, not fatal.
        """
        cfg = self.config
        if not frame.images:
            return
        if frame_index % max(1, cfg.sample_stride) != 0:
            return

        for name, lazy in frame.images.items():
            if cfg.cameras is not None and name not in cfg.cameras:
                continue
            acc = accums.get(name)
            if acc is None:
                acc = _CameraAccumulator(name, cfg)
                accums[name] = acc
            if cfg.max_frames and acc.num_frames >= cfg.max_frames:
                continue
            try:
                img = lazy.load()
            except Exception:
                continue
            acc.update(img)

    def finalize(self, accums: dict[str, _CameraAccumulator]) -> VideoQuality | None:
        if not accums:
            return None
        per_camera = {name: acc.finalize() for name, acc in accums.items()}
        return _build_rollup(per_camera, self.config)

    def analyze_episode(self, episode) -> VideoQuality | None:
        """Score an episode's camera streams in a single pass.

        Returns None if the episode carries no (selected) camera images.
        """
        accums = self.new_accumulators()
        idx = -1
        for frame in episode.frames():
            if frame.images:
                idx += 1
                self.consume_frame(frame, accums, idx)
        return self.finalize(accums)


def _build_rollup(
    per_camera: dict[str, CameraVideoQuality],
    config: VideoQualityConfig,
) -> VideoQuality:
    """Combine per-camera results into episode-level worst-case rollups."""
    vq = VideoQuality(per_camera=per_camera)

    def _vals(attr: str) -> list[float]:
        return [
            getattr(c, attr)
            for c in per_camera.values()
            if getattr(c, attr) is not None
        ]

    mean_sharp = _vals("mean_sharpness")
    if mean_sharp:
        vq.min_sharpness = float(min(mean_sharp))
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
    if vq.mean_motion is not None and vq.mean_motion < config.motion_low_flag:
        if "no_motion" not in seen:
            vq.flags.append("no_motion")

    return vq
