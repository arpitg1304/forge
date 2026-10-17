"""Evidence and abstention regressions for camera quality."""

import json

import numpy as np
import pytest

from forge.core.models import Episode, Frame, LazyImage
from forge.quality import QualityAnalyzer, QualityReport
from forge.quality.video import VideoQualityAnalyzer, VideoQualityConfig


def fixture(count=30, fps=30, timestamps=True):
    img = np.random.default_rng(1).integers(30, 220, (24, 24, 3), dtype=np.uint8)
    frames = [
        Frame(
            index=i,
            timestamp=i / fps if timestamps else None,
            state=np.array([i * 0.1]),
            action=np.array([i * 0.1]),
            images={"wrist": lazy(img)},
        )
        for i in range(count)
    ]
    return Episode("test", fps=fps, _frames_cache=frames)


def lazy(img):
    return LazyImage(loader=lambda: img, height=24, width=24)


def analyze(ep, **kwargs):
    return (
        VideoQualityAnalyzer(VideoQualityConfig(**kwargs)).analyze_episode(ep).per_camera["wrist"]
    )


def test_gap_estimates_use_original_observation_times_even_with_striding():
    ep = fixture()
    for f in ep._frames_cache[10:]:
        f.timestamp += 0.3
    for stride in (1, 4):
        cam = analyze(ep, sample_stride=stride)
        assert cam.estimated_dropped_frames == 9
        assert cam.timestamp_gaps == 1
        assert "video_frame_drops" in cam.flags
        event = next(e for e in cam.events if e.kind == "video_frame_gap")
        assert (event.start_frame, event.end_frame) == (9, 11)
        assert event.evidence["significant"]
        assert event.evidence["estimated_missing_observations"] == 9
        assert all(
            not (e.start_frame < 10 < e.end_frame)
            for e in cam.events
            if e.kind == "video_frozen_candidate"
        )


def test_sampling_and_cap_are_not_dropped_frames():
    cam = analyze(fixture(40), sample_stride=4, max_frames=3)
    assert cam.num_frames == cam.decode_attempts == 3
    assert cam.sampled_out_frames == 30
    assert cam.capped_frames == 7
    assert cam.estimated_dropped_frames == 0
    assert cam.missing_frames == cam.decode_failures == 0
    assert cam.decode_coverage == pytest.approx(3 / 40)
    assert "video_frame_drops" not in cam.flags


def test_decode_failures_are_observable_and_break_frozen_sequences():
    ep = fixture(20)

    def broken():
        raise OSError("codec failed")

    for f in ep._frames_cache[7:10]:
        f.images["wrist"] = LazyImage(loader=broken, height=24, width=24)
    cam = analyze(ep)
    assert cam.decode_attempts == 20 and cam.decode_failures == 3
    assert "video_decode_failure" in cam.flags
    e = next(e for e in cam.events if e.kind == "video_decode_failure")
    assert (e.start_frame, e.end_frame) == (7, 10)
    assert all(
        not (e.start_frame < 7 and e.end_frame > 10)
        for e in cam.events
        if e.kind == "video_frozen_candidate"
    )


def test_all_decode_failures_never_look_healthy():
    ep = fixture(10)

    def broken():
        raise OSError("codec failed")

    for f in ep._frames_cache:
        f.images["wrist"] = LazyImage(loader=broken, height=24, width=24)
    cam = analyze(ep, max_frames=3)
    assert cam.decode_attempts == cam.decode_failures == 3
    assert cam.num_frames == 0 and cam.blurry_fraction is None
    assert cam.capped_frames == 7
    assert cam.metric_status["pixels"].startswith("unavailable")


@pytest.mark.parametrize(
    "image",
    [
        np.full((24, 24, 3), np.nan),
        np.zeros((24, 24), dtype=np.uint16),
        np.full((24, 24, 3), 300.0),
        np.zeros((0, 4, 3)),
    ],
)
def test_invalid_pixels_are_not_exposure_observations(image):
    ep = fixture(1)
    ep._frames_cache[0].images["wrist"] = lazy(image)
    cam = analyze(ep)
    assert cam.invalid_frames == 1 and cam.num_frames == 0
    assert cam.underexposed_fraction is None
    assert cam.flags == ["video_invalid_pixels"]


def test_missing_cameras_include_leading_and_trailing_absence():
    ep = fixture(20)
    for i in [0, 1, 2, 9, 10, 19]:
        ep._frames_cache[i].images = {}
    cam = analyze(ep)
    assert cam.expected_frames == 20 and cam.missing_frames == 6
    assert [
        (e.start_frame, e.end_frame) for e in cam.events if e.kind == "video_camera_missing"
    ] == [(0, 3), (9, 11), (19, 20)]
    assert cam.estimated_dropped_frames == 0


def test_explicit_camera_absent_entirely_is_reported():
    ep = fixture(10)
    result = VideoQualityAnalyzer(cameras=["overhead"]).analyze_episode(ep)
    cam = result.per_camera["overhead"]
    assert cam.missing_frames == 10 and cam.num_frames == 0
    assert cam.flags == ["video_camera_missing"]


def test_frozen_duration_uses_seconds_not_sample_count():
    for fps, stride in [(15, 1), (30, 1), (30, 3)]:
        ep = fixture(fps + 1, fps)
        cam = analyze(ep, sample_stride=stride)
        assert cam.max_frozen_duration_s == pytest.approx(1.0)
        assert "frozen_frames" in cam.flags


def test_no_stored_timestamps_abstains_on_drop_detection():
    cam = analyze(fixture(timestamps=False))
    assert cam.estimated_dropped_frames is None
    assert cam.metric_status["frame_drops"].startswith("unavailable")
    assert "video_invalid_timestamps" not in cam.flags
    assert cam.max_frozen_duration_s > 0


def test_invalid_time_breaks_flow_and_abstains_on_smoothness():
    ep = fixture()
    ep._frames_cache[10].timestamp = ep._frames_cache[9].timestamp
    cam = analyze(ep, level="motion")
    assert "video_invalid_timestamps" in cam.flags
    assert cam.motion_smoothness is None
    json.dumps(cam.to_dict(), allow_nan=False)


def test_blank_frames_do_not_bridge_flow():
    ep = fixture(3)
    ep._frames_cache[1].images["wrist"] = lazy(np.zeros((24, 24, 3), dtype=np.uint8))
    cam = analyze(ep, level="motion")
    assert cam.mean_motion is None
    assert cam.underexposed_fraction == pytest.approx(1 / 3)


def test_resolution_change_breaks_flow_without_crashing():
    ep = fixture(4)
    ep._frames_cache[2].images["wrist"] = lazy(
        np.random.default_rng(3).integers(0, 255, (12, 30, 3), dtype=np.uint8)
    )
    cam = analyze(ep, level="motion")
    assert cam.resolution_changes == 2
    assert "video_resolution_change" in cam.flags
    assert cam.motion_smoothness is None


def test_report_roundtrip_keeps_camera_events_coverage_and_configuration(tmp_path):
    ep = fixture()
    ep._frames_cache[10].images = {}
    eq = QualityAnalyzer(video_config=VideoQualityConfig()).analyze_episode(ep)
    assert eq.video.per_camera["wrist"].missing_frames == 1
    report = QualityReport("fixture", per_episode=[eq])
    path = tmp_path / "report.json"
    report.to_json(path)
    loaded = QualityReport.from_json(path).per_episode[0]
    assert loaded.video.to_dict() == eq.video.to_dict()
    assert loaded.video.scorer_version == "video-v2-integrity"
    assert "video_camera_missing" in loaded.flags
    json.dumps(loaded.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kw",
    [
        {"sample_stride": 0},
        {"downscale": 0},
        {"gap_factor": 1},
        {"expected_fps": -1},
        {"max_frames": -1},
        {"level": "semantic"},
        {"frozen_duration_s": float("nan")},
    ],
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        VideoQualityConfig(**kw)


def test_late_camera_still_gets_full_observation_gap_evidence():
    ep = fixture(30)
    for f in ep._frames_cache[:12]:
        f.images = {}
    for f in ep._frames_cache[5:]:
        f.timestamp += 0.3
    cam = analyze(ep)
    assert cam.estimated_dropped_frames == 9
    assert any(e.kind == "video_frame_gap" and e.start_frame == 4 for e in cam.events)


def test_repeated_small_drops_have_a_fraction_and_warning():
    ep = fixture(20)
    for f in ep._frames_cache:
        f.timestamp = (f.index + f.index // 3) / 30
    cam = analyze(ep, significant_drop_frames=3, significant_gap_s=1)
    assert cam.estimated_dropped_frames == 6
    assert cam.estimated_drop_fraction == pytest.approx(6 / 25)
    assert "video_frame_drops" in cam.flags


def test_timestamp_free_unknown_fps_has_no_invented_seconds():
    ep = fixture(timestamps=False)
    ep.fps = None
    cam = analyze(ep)
    assert cam.estimated_dropped_frames is None
    assert cam.max_frozen_duration_s is None
    assert all(e.start_time_s is None and e.end_time_s is None for e in cam.events)


def test_min_sharpness_is_a_true_minimum_and_legacy_reports_still_load():
    from forge.quality.models import EpisodeQuality
    from forge.quality.video.analyzer import _build_rollup
    from forge.quality.video.models import CameraVideoQuality

    video = _build_rollup(
        {
            "a": CameraVideoQuality("a", min_sharpness=2, mean_sharpness=100),
            "b": CameraVideoQuality("b", min_sharpness=5, mean_sharpness=80),
        },
        VideoQualityConfig(),
    )
    assert video.min_sharpness == 2
    legacy = EpisodeQuality.from_dict(
        {"episode_id": "old", "video_num_cameras": 2, "video_min_sharpness": 80}
    )
    assert legacy.video.min_sharpness == 80
    assert legacy.video.scorer_version == "video-v1"
    assert legacy.video.per_camera == {}


def test_no_motion_does_not_hide_an_unavailable_camera():
    ep = fixture()
    video = VideoQualityAnalyzer(level="motion", cameras=["wrist", "absent"]).analyze_episode(ep)
    assert video.per_camera["wrist"].mean_motion < 0.05
    assert "no_motion" not in video.flags
    assert "video_camera_missing" in video.flags
