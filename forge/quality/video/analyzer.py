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
    """Per-camera streaming accumulator for Tier 0 metrics."""

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

        self.num_frames += 1

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

        # ── Flags ──
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
        return _build_rollup(per_camera)

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


def _build_rollup(per_camera: dict[str, CameraVideoQuality]) -> VideoQuality:
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

    # Union of per-camera flags (a flag fires if any camera raised it).
    seen: set[str] = set()
    for cam in per_camera.values():
        for f in cam.flags:
            if f not in seen:
                seen.add(f)
                vq.flags.append(f)

    return vq
