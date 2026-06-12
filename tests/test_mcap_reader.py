"""Integration tests for MCAPReader against the Rerun fixture corpus.

trossen_transfer_cube.mcap is the most ROS-manipulation-shaped fixture
(8-DOF JointStates per arm, 4 wrist+external cameras, task description on
foxglove.KeyValuePair). supported_ros2_messages.mcap exercises the ROS2 CDR
path. r2b_galileo.mcap exercises images-only (no JointState).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")

from forge.formats.mcap import MCAPReader, generate_config  # noqa: E402
from forge.formats.mcap.topic_config import (  # noqa: E402
    EpisodeSplit,
    FieldMapping,
    SyncPolicy,
    TopicConfig,
)

FIXTURES = Path(__file__).parent.parent / "sample_data" / "mcap"
TROSSEN = FIXTURES / "trossen_transfer_cube.mcap"
ROS2 = FIXTURES / "supported_ros2_messages.mcap"
GALILEO = FIXTURES / "r2b_galileo.mcap"

if not all(p.exists() for p in (TROSSEN, ROS2, GALILEO)):
    pytest.skip(
        "MCAP fixtures missing — run `python scripts/download_mcap_fixtures.py`",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _silence_sync_warnings():
    """Reader logs per-frame skew warnings — don't pollute test output."""
    logging.getLogger("forge.formats.mcap.reader").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Hand-tuned config — Trossen (the ROS-manipulation-shaped fixture)
# ---------------------------------------------------------------------------


def _trossen_config(source: Path = TROSSEN) -> TopicConfig:
    return TopicConfig(
        source=source,
        episodes=EpisodeSplit(strategy="single"),
        fields={
            "observation.state": FieldMapping(
                topic="/robot_left/joint_states", field="position", dtype="float32"
            ),
            "action": FieldMapping(
                topic="/robot_right/joint_states", field="position", dtype="float32"
            ),
            "observation.images.cam_high": FieldMapping(
                topic="/external/cam_high/video_compressed", encoding="video"
            ),
            "observation.images.left_wrist": FieldMapping(
                topic="/robot_left/wrist_camera/video_compressed", encoding="video"
            ),
        },
        sync=SyncPolicy(primary="observation.state", method="nearest", max_skew_ms=200.0),
    )


class TestTrossenWithHandTunedConfig:
    def test_yields_one_episode_with_expected_shape(self):
        eps = list(MCAPReader().read_episodes(TROSSEN, config=_trossen_config()))
        assert len(eps) == 1
        ep = eps[0]
        frames = ep.load_frames()
        assert len(frames) > 100, "Trossen has ~600 left-arm joint_states frames"
        assert ep.state_dim == 8  # 8-DOF Trossen left arm
        assert ep.fps is not None and ep.fps > 5
        assert set(ep.cameras.keys()) == {"cam_high", "left_wrist"}

    def test_first_frame_is_first(self):
        ep = next(MCAPReader().read_episodes(TROSSEN, config=_trossen_config()))
        f0 = next(ep.frames())
        assert f0.is_first
        assert f0.index == 0

    def test_state_and_action_are_numpy(self):
        ep = next(MCAPReader().read_episodes(TROSSEN, config=_trossen_config()))
        f = ep.load_frames()[10]
        assert isinstance(f.state, np.ndarray)
        assert f.state.dtype == np.float32
        assert f.state.shape == (8,)
        assert isinstance(f.action, np.ndarray)
        assert f.action.shape == (8,)

    def test_images_are_lazy(self):
        ep = next(MCAPReader().read_episodes(TROSSEN, config=_trossen_config()))
        f = ep.load_frames()[5]
        assert "cam_high" in f.images
        img = f.images["cam_high"]
        assert not img.is_loaded


# ---------------------------------------------------------------------------
# Auto-generated config — verify it produces a usable Episode
# ---------------------------------------------------------------------------


class TestTrossenWithGeneratedConfig:
    def test_generated_config_yields_episode(self):
        res = generate_config(TROSSEN)
        assert res.config is not None
        eps = list(MCAPReader().read_episodes(TROSSEN, config=res.config))
        assert len(eps) >= 1
        ep = eps[0]
        frames = ep.load_frames()
        assert len(frames) > 0
        assert (
            ep.language_instruction
            == "Pick and transfer the cube back and forth between grippers."
        )


# ---------------------------------------------------------------------------
# ROS2 CDR fixture
# ---------------------------------------------------------------------------


class TestSupportedRos2Messages:
    def test_reads_with_generated_config(self):
        eps = list(MCAPReader().read_episodes(ROS2))
        assert len(eps) >= 1
        ep = eps[0]
        frames = ep.load_frames()
        assert len(frames) > 0
        # /joint_state has 4 joints.
        assert ep.state_dim == 4
        f0 = frames[0]
        assert isinstance(f0.state, np.ndarray)
        assert f0.state.shape == (4,)


# ---------------------------------------------------------------------------
# Galileo — images only, no JointState (state_dim should be None)
# ---------------------------------------------------------------------------


class TestGalileoImagesOnly:
    def test_no_state_field(self):
        eps = list(MCAPReader().read_episodes(GALILEO))
        assert len(eps) >= 1
        ep = eps[0]
        assert ep.state_dim is None
        # Should pick up at least one camera.
        assert len(ep.cameras) >= 1


# ---------------------------------------------------------------------------
# Episode strategies
# ---------------------------------------------------------------------------


class TestEpisodeStrategies:
    def test_time_gap_falls_to_single_when_no_gap(self):
        cfg = _trossen_config()
        cfg.episodes = EpisodeSplit(strategy="time_gap", time_gap_seconds=10.0)
        eps = list(MCAPReader().read_episodes(TROSSEN, config=cfg))
        # Trossen recording is one continuous take — no 10s gap, one episode.
        assert len(eps) == 1

    def test_drop_first_n_frames(self):
        cfg = _trossen_config()
        baseline = next(MCAPReader().read_episodes(TROSSEN, config=cfg))
        baseline_n = len(baseline.load_frames())
        cfg.episodes = EpisodeSplit(strategy="single", drop_first_n_frames=10)
        ep = next(MCAPReader().read_episodes(TROSSEN, config=cfg))
        assert len(ep.load_frames()) == baseline_n - 10

    def test_min_length_filters_episode(self):
        cfg = _trossen_config()
        cfg.episodes = EpisodeSplit(strategy="single", min_length_frames=10**6)
        assert list(MCAPReader().read_episodes(TROSSEN, config=cfg)) == []


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_fields_raises(self):
        cfg = TopicConfig()  # no fields
        with pytest.raises(ValueError, match="fields"):
            list(MCAPReader().read_episodes(TROSSEN, config=cfg))

    def test_primary_missing_topic_raises(self):
        cfg = TopicConfig(
            fields={
                "observation.state": FieldMapping(
                    topic="/does/not/exist", field="position"
                )
            },
            sync=SyncPolicy(primary="observation.state"),
        )
        with pytest.raises(ValueError, match="primary topic"):
            list(MCAPReader().read_episodes(TROSSEN, config=cfg))


# ---------------------------------------------------------------------------
# Public API contract — all fixtures readable through the registry
# ---------------------------------------------------------------------------


def test_reader_is_registered():
    from forge.formats import FormatRegistry

    reader = FormatRegistry.get_reader("mcap")
    assert isinstance(reader, MCAPReader)
