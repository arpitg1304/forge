"""Tests for forge.quality.video — Tier 0 pixel metrics.

Builds synthetic episodes whose frames carry in-memory images, then checks
that each metric and flag behaves as expected. Numpy only; no decode needed.
"""

import numpy as np
import pytest

from forge.core.models import Episode, Frame, LazyImage
from forge.quality.analyzer import QualityAnalyzer
from forge.quality.video import (
    VideoQuality,
    VideoQualityAnalyzer,
    VideoQualityConfig,
)
from forge.quality.video.pixel import (
    colorfulness,
    frame_mae,
    laplacian_variance,
    longest_true_run,
    prepare,
)


# ── Image generators ─────────────────────────────────────────────


def _sharp_frame(h: int = 96, w: int = 96, seed: int = 0) -> np.ndarray:
    """High-frequency noise → high variance of Laplacian."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _blurry_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """Smooth gradient → near-zero Laplacian variance."""
    grad = np.linspace(40, 200, w, dtype=np.float64)
    img = np.tile(grad, (h, 1))
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _solid_frame(value: int, h: int = 96, w: int = 96) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _lazy(img: np.ndarray) -> LazyImage:
    h, w = img.shape[0], img.shape[1]
    c = img.shape[2] if img.ndim == 3 else 1
    return LazyImage(loader=lambda im=img: im, height=h, width=w, channels=c)


def make_episode(frames_images: list[dict[str, np.ndarray]], ep_id: str = "ep0") -> Episode:
    """Build an Episode whose frames carry the given per-camera images."""
    frames = [
        Frame(index=i, images={name: _lazy(img) for name, img in cams.items()})
        for i, cams in enumerate(frames_images)
    ]
    return Episode(episode_id=ep_id, _frame_loader=lambda: iter(frames))


# ── Pixel primitive tests ────────────────────────────────────────


def test_sharpness_distinguishes_blur():
    sharp_gray, _ = prepare(_sharp_frame(seed=1), 128)
    blur_gray, _ = prepare(_blurry_frame(), 128)
    assert laplacian_variance(sharp_gray) > laplacian_variance(blur_gray) * 10


def test_frame_mae_zero_for_identical():
    g, _ = prepare(_solid_frame(120), 64)
    assert frame_mae(g, g) == 0.0


def test_colorfulness_gray_vs_color():
    gray = _solid_frame(128).astype(np.float64)
    rng = np.random.default_rng(2)
    color = rng.integers(0, 256, size=(64, 64, 3)).astype(np.float64)
    assert colorfulness(color) > colorfulness(gray)


def test_longest_true_run():
    assert longest_true_run([False, True, True, False, True]) == 2
    assert longest_true_run([]) == 0
    assert longest_true_run([True, True, True]) == 3


def test_prepare_handles_grayscale_source():
    gray2d = _solid_frame(100)[..., 0]
    g, rgb = prepare(gray2d, 64)
    assert g.ndim == 2
    assert rgb is None


def test_prepare_rescales_float01_images():
    # A float image in [0, 1] should be treated as 0–255 internally.
    img = np.full((32, 32, 3), 0.5, dtype=np.float64)
    g, _ = prepare(img, 32)
    assert 120 < g.mean() < 135


# ── Analyzer / flag tests ────────────────────────────────────────


def test_blurry_episode_flagged():
    ep = make_episode([{"cam": _blurry_frame()} for _ in range(20)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert "blurry" in vq.flags
    assert vq.blurry_fraction == 1.0


def test_sharp_episode_not_blurry():
    ep = make_episode([{"cam": _sharp_frame(seed=i)} for i in range(20)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert "blurry" not in vq.flags


def test_frozen_frames_flagged():
    frame = _sharp_frame(seed=5)
    ep = make_episode([{"cam": frame.copy()} for _ in range(30)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert "frozen_frames" in vq.flags
    assert vq.max_frozen_run >= 29


def test_underexposed_flagged():
    ep = make_episode([{"cam": _solid_frame(5)} for _ in range(20)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert "under_exposed" in vq.flags


def test_overexposed_flagged():
    ep = make_episode([{"cam": _solid_frame(250)} for _ in range(20)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert "over_exposed" in vq.flags


def test_no_images_returns_none():
    frames = [Frame(index=i) for i in range(5)]
    ep = Episode(episode_id="empty", _frame_loader=lambda: iter(frames))
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is None


def test_multi_camera_rollup_is_worst_case():
    # One sharp camera, one blurry camera → episode should be flagged blurry.
    ep = make_episode(
        [{"good": _sharp_frame(seed=i), "bad": _blurry_frame()} for i in range(20)]
    )
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    assert vq is not None
    assert set(vq.per_camera) == {"good", "bad"}
    assert "blurry" in vq.flags
    assert vq.blurry_fraction == 1.0  # max across cameras


def test_camera_filter():
    ep = make_episode(
        [{"good": _sharp_frame(seed=i), "bad": _blurry_frame()} for i in range(10)]
    )
    cfg = VideoQualityConfig(cameras=["good"])
    vq = VideoQualityAnalyzer(cfg).analyze_episode(ep)
    assert set(vq.per_camera) == {"good"}


def test_stride_subsamples_frames():
    ep = make_episode([{"cam": _sharp_frame(seed=i)} for i in range(20)])
    cfg = VideoQualityConfig(sample_stride=5)
    vq = VideoQualityAnalyzer(cfg).analyze_episode(ep)
    assert vq.per_camera["cam"].num_frames == 4  # frames 0,5,10,15


def test_max_frames_caps_analysis():
    ep = make_episode([{"cam": _sharp_frame(seed=i)} for i in range(20)])
    cfg = VideoQualityConfig(max_frames=3)
    vq = VideoQualityAnalyzer(cfg).analyze_episode(ep)
    assert vq.per_camera["cam"].num_frames == 3


# ── Integration with QualityAnalyzer ─────────────────────────────


def test_quality_analyzer_attaches_video():
    actions = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (30, 7)), axis=0)
    frames = [
        Frame(index=i, action=actions[i], images={"cam": _lazy(_blurry_frame())})
        for i in range(30)
    ]
    ep = Episode(episode_id="ep0", _frame_loader=lambda: iter(frames))

    analyzer = QualityAnalyzer(video_config=VideoQualityConfig())
    eq = analyzer.analyze_episode(ep)

    assert eq.video is not None
    assert "blurry" in eq.video.flags
    assert "blurry" in eq.flags  # video flags propagate to episode flags


def test_quality_analyzer_video_off_by_default():
    frames = [Frame(index=i, images={"cam": _lazy(_blurry_frame())}) for i in range(10)]
    ep = Episode(episode_id="ep0", _frame_loader=lambda: iter(frames))
    eq = QualityAnalyzer().analyze_episode(ep)
    assert eq.video is None


# ── Report serialization round-trip ──────────────────────────────


def test_report_roundtrip_preserves_video(tmp_path):
    from forge.quality.models import EpisodeQuality, QualityReport

    ep = make_episode([{"cam": _blurry_frame()} for _ in range(15)])
    vq = VideoQualityAnalyzer(VideoQualityConfig()).analyze_episode(ep)
    eq = EpisodeQuality(episode_id="ep0", num_frames=15, overall_score=7.0, video=vq)
    eq.flags.extend(vq.flags)

    report = QualityReport(dataset_path="x")
    report.per_episode.append(eq)

    path = tmp_path / "report.json"
    report.to_json(path)
    loaded = QualityReport.from_json(path)

    lv = loaded.per_episode[0].video
    assert lv is not None
    assert lv.blurry_fraction == pytest.approx(vq.blurry_fraction)
    assert lv.min_sharpness == pytest.approx(vq.min_sharpness, rel=1e-3)
