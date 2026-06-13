"""Tests for video (Tier 0) criteria in forge filter."""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import DatasetInfo, Episode, Frame, LazyImage
from forge.filter.engine import FilterConfig, FilterEngine
from forge.formats.registry import FormatRegistry
from forge.quality.models import EpisodeQuality, QualityReport
from forge.quality.video.models import CameraVideoQuality, VideoQuality


# ── _needs_video_analysis ────────────────────────────────────────


def test_needs_video_analysis_numeric():
    assert FilterEngine(FilterConfig(min_sharpness=80))._needs_video_analysis()
    assert FilterEngine(FilterConfig(max_frozen_fraction=0.5))._needs_video_analysis()
    assert FilterEngine(FilterConfig(max_overexposed_fraction=0.3))._needs_video_analysis()
    assert FilterEngine(FilterConfig(max_underexposed_fraction=0.3))._needs_video_analysis()


def test_needs_video_analysis_via_video_flag():
    assert FilterEngine(FilterConfig(exclude_flags=["blurry"]))._needs_video_analysis()
    assert FilterEngine(FilterConfig(exclude_flags=["frozen_frames"]))._needs_video_analysis()


def test_no_video_analysis_for_proprio_only():
    assert not FilterEngine(FilterConfig(min_quality=6.0))._needs_video_analysis()
    assert not FilterEngine(FilterConfig(exclude_flags=["jerky"]))._needs_video_analysis()
    # ...but a proprio flag still triggers (proprio) quality analysis.
    assert FilterEngine(FilterConfig(exclude_flags=["jerky"]))._needs_quality_analysis()


# ── _evaluate_episode video criteria ─────────────────────────────


def _eq_with_video(**video_kwargs) -> EpisodeQuality:
    eq = EpisodeQuality(episode_id="ep", num_frames=30, overall_score=8.0)
    eq.video = VideoQuality(**video_kwargs)
    return eq


def test_min_sharpness_excludes_blurry():
    engine = FilterEngine(FilterConfig(min_sharpness=100))
    keep, reasons = engine._evaluate_episode("ep", _eq_with_video(min_sharpness=40))
    assert not keep
    assert any("sharpness" in r for r in reasons)


def test_min_sharpness_keeps_sharp():
    engine = FilterEngine(FilterConfig(min_sharpness=100))
    keep, _ = engine._evaluate_episode("ep", _eq_with_video(min_sharpness=500))
    assert keep


def test_max_frozen_excludes():
    engine = FilterEngine(FilterConfig(max_frozen_fraction=0.5))
    keep, reasons = engine._evaluate_episode("ep", _eq_with_video(frozen_fraction=0.8))
    assert not keep
    assert any("frozen" in r for r in reasons)


def test_max_overexposed_and_underexposed():
    engine = FilterEngine(
        FilterConfig(max_overexposed_fraction=0.2, max_underexposed_fraction=0.2)
    )
    keep, reasons = engine._evaluate_episode(
        "ep", _eq_with_video(overexposed_fraction=0.9, underexposed_fraction=0.0)
    )
    assert not keep
    assert any("overexposed" in r for r in reasons)
    assert not any("underexposed" in r for r in reasons)


def test_missing_video_metrics_skips_video_filter(caplog):
    # Criterion set but episode has no video data → keep, with a warning.
    engine = FilterEngine(FilterConfig(min_sharpness=100))
    eq = EpisodeQuality(episode_id="ep", num_frames=30, overall_score=8.0)
    keep, reasons = engine._evaluate_episode("ep", eq)
    assert keep
    assert reasons == []


def test_video_and_proprio_criteria_combine():
    engine = FilterEngine(FilterConfig(min_quality=7.0, min_sharpness=100))
    eq = _eq_with_video(min_sharpness=500)
    eq.overall_score = 5.0  # fails proprio
    keep, reasons = engine._evaluate_episode("ep", eq)
    assert not keep
    assert any("score" in r for r in reasons)


# ── from-report integration ──────────────────────────────────────


def test_from_report_reads_video_rollups(tmp_path):
    report = QualityReport(dataset_path="./x")
    sharp = EpisodeQuality(episode_id="ep_000", num_frames=30, overall_score=8.0)
    sharp.video = VideoQuality(min_sharpness=600.0, frozen_fraction=0.05)
    blurry = EpisodeQuality(episode_id="ep_001", num_frames=30, overall_score=8.0)
    blurry.video = VideoQuality(min_sharpness=30.0, frozen_fraction=0.05)
    report.per_episode.extend([sharp, blurry])

    path = tmp_path / "report.json"
    report.to_json(path)

    loaded = QualityReport.from_json(path)
    qmap = {eq.episode_id: eq for eq in loaded.per_episode}
    engine = FilterEngine(FilterConfig(from_report=path, min_sharpness=100))

    assert engine._evaluate_episode("ep_000", qmap["ep_000"])[0] is True
    assert engine._evaluate_episode("ep_001", qmap["ep_001"])[0] is False


# ── live-analysis integration ────────────────────────────────────


def _lazy(img: np.ndarray) -> LazyImage:
    return LazyImage(loader=lambda im=img: im, height=img.shape[0], width=img.shape[1], channels=3)


def _blurry_img() -> np.ndarray:
    grad = np.linspace(40, 200, 96, dtype=np.float64)
    img = np.tile(grad, (96, 1))
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _sharp_img(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)


class _VideoMockReader:
    """Two episodes: ep_000 sharp, ep_001 blurry."""

    def can_read(self, path: Path) -> bool:  # pragma: no cover
        return True

    def inspect(self, path: Path) -> DatasetInfo:
        return DatasetInfo(path=path, format="mock-video", num_episodes=2, total_frames=40)

    def read_episodes(self, path: Path):
        for idx, maker in enumerate((lambda j: _sharp_img(j), lambda j: _blurry_img())):
            frames = [Frame(index=j, images={"cam": _lazy(maker(j))}) for j in range(20)]
            yield Episode(episode_id=f"ep_{idx:03d}", _frame_loader=lambda f=frames: iter(f))


@pytest.fixture
def register_video_mock():
    FormatRegistry._readers["mock-video"] = _VideoMockReader
    yield
    del FormatRegistry._readers["mock-video"]


def test_live_filter_excludes_blurry_episode(tmp_path, register_video_mock):
    ds = tmp_path / "ds"
    ds.mkdir()
    engine = FilterEngine(FilterConfig(min_sharpness=100))
    result = engine.filter(ds, output=None, source_format="mock-video")  # dry-run

    assert "ep_000" in result.kept_ids
    assert "ep_001" in result.excluded_ids
    assert any("sharpness" in r for r in result.exclusion_reasons["ep_001"])


def test_live_filter_exclude_blurry_flag(tmp_path, register_video_mock):
    ds = tmp_path / "ds"
    ds.mkdir()
    engine = FilterEngine(FilterConfig(exclude_flags=["blurry"]))
    result = engine.filter(ds, output=None, source_format="mock-video")

    assert "ep_000" in result.kept_ids
    assert "ep_001" in result.excluded_ids
