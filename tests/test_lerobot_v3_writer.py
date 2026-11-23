"""Tests for LeRobot V3 writer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from forge.core.exceptions import ConversionError
from forge.core.models import CameraInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample episode for testing."""

    def create_frame(idx: int) -> Frame:
        def make_image_loader(index: int):
            def loader():
                # Create a simple gradient image
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (index * 10) % 256
                img[:, :, 1] = np.arange(64).reshape(-1, 1)
                img[:, :, 2] = np.arange(64).reshape(1, -1)
                return img

            return loader

        return Frame(
            index=idx,
            timestamp=idx / 30.0,
            images={
                "camera0": LazyImage(
                    loader=make_image_loader(idx),
                    height=64,
                    width=64,
                    channels=3,
                ),
            },
            state=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            action=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], dtype=np.float32),
        )

    def frame_loader():
        for i in range(30):  # 1 second at 30 fps
            yield create_frame(i)

    return Episode(
        episode_id="ep_001",
        language_instruction="Pick up the red block",
        cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
        fps=30.0,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def sample_episodes(sample_episode: Episode) -> list[Episode]:
    """Create multiple sample episodes."""

    def create_episode(ep_idx: int) -> Episode:
        def create_frame(idx: int) -> Frame:
            def make_image_loader(index: int):
                def loader():
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    img[:, :, 0] = (index * 10) % 256
                    return img

                return loader

            return Frame(
                index=idx,
                timestamp=idx / 30.0,
                images={
                    "camera0": LazyImage(
                        loader=make_image_loader(idx),
                        height=64,
                        width=64,
                        channels=3,
                    ),
                },
                state=np.array([0.1] * 7, dtype=np.float32),
                action=np.array([0.01] * 7, dtype=np.float32),
            )

        def frame_loader():
            for i in range(20 + ep_idx * 5):  # Variable length episodes
                yield create_frame(i)

        return Episode(
            episode_id=f"ep_{ep_idx:03d}",
            language_instruction=f"Task {ep_idx % 2}",  # Two different tasks
            cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

    return [create_episode(i) for i in range(3)]


class TestLeRobotV3WriterConfig:
    """Tests for LeRobotV3WriterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LeRobotV3WriterConfig()
        assert config.fps == 30.0
        assert config.robot_type == "unknown"
        # LeRobot's own encoder defaults, so outputs merge with its recordings.
        assert config.video_codec == "libsvtav1"
        assert config.video_crf == 30
        assert config.video_gop_size == 2
        assert config.chunks_size == 1000
        assert config.task is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LeRobotV3WriterConfig(
            fps=60.0,
            robot_type="franka",
            video_crf=18,
            chunks_size=500,
            camera_name_mapping={"cam0": "observation.images.front"},
        )
        assert config.fps == 60.0
        assert config.robot_type == "franka"
        assert config.video_crf == 18
        assert config.chunks_size == 500
        assert config.camera_name_mapping == {"cam0": "observation.images.front"}


class TestLeRobotV3Writer:
    """Tests for LeRobotV3Writer."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_single_episode(self, tmp_path: Path, sample_episode: Episode):
        """Test writing a single episode."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)  # Flush to write data

        # Check parquet file exists (chunked structure)
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        assert parquet_path.exists()

        # Check video file exists (chunked structure)
        video_path = output_dir / "videos" / "observation.images.camera0" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_dataset(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test writing full dataset with multiple episodes."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Check structure
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "data" / "chunk-000").exists()
        assert (output_dir / "videos").exists()

        # Check data parquet exists
        parquet_files = list((output_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(parquet_files) == 1

        # Check metadata content
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["fps"] == 30
        assert info["total_episodes"] == 3
        assert info["splits"]["train"] == "0:3"
        assert "data_path" in info
        assert "video_path" in info

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_episodes_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test episodes parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read episodes parquet
        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(episodes_path)
        episodes_data = table.to_pylist()

        assert len(episodes_data) == 3
        assert episodes_data[0]["episode_index"] == 0
        assert episodes_data[1]["episode_index"] == 1
        assert episodes_data[2]["episode_index"] == 2
        assert all("length" in ep for ep in episodes_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_tasks_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test tasks.parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read tasks.parquet
        tasks_path = output_dir / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_path)
        tasks_data = table.to_pylist()

        # Should have 2 unique tasks (Task 0 and Task 1)
        assert len(tasks_data) == 2
        assert any(t["task"] == "Task 0" for t in tasks_data)
        assert any(t["task"] == "Task 1" for t in tasks_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_camera_name_mapping(self, tmp_path: Path, sample_episode: Episode):
        """Test custom camera name mapping."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(
            fps=30.0,
            camera_name_mapping={"camera0": "observation.images.front_cam"},
        )
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Check video is in mapped directory
        video_path = output_dir / "videos" / "observation.images.front_cam" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_parquet_has_required_columns(self, tmp_path: Path, sample_episode: Episode):
        """Test parquet file has required columns."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Read parquet and check columns
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(parquet_path)
        columns = set(table.column_names)

        assert "episode_index" in columns
        assert "frame_index" in columns
        assert "index" in columns
        assert "task_index" in columns
        assert "timestamp" in columns
        assert "observation.state" in columns
        assert "action" in columns

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_default_camera_name_stripping(self, tmp_path: Path):
        """Test that _image and _rgb suffixes are stripped by default."""
        config = LeRobotV3WriterConfig()
        writer = LeRobotV3Writer(config)

        assert writer._map_camera_name("agentview_image") == "observation.images.agentview"
        assert writer._map_camera_name("front_rgb") == "observation.images.front"
        assert writer._map_camera_name("wrist") == "observation.images.wrist"

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_info_json_structure(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test info.json has correct LeRobot v3 structure."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        # Check required fields
        assert "codebase_version" in info
        assert "robot_type" in info
        assert "total_episodes" in info
        assert "total_frames" in info
        assert "total_tasks" in info
        assert "chunks_size" in info
        assert "fps" in info
        assert "splits" in info
        assert "data_path" in info
        assert "video_path" in info
        assert "features" in info

        # Check features have correct structure
        features = info["features"]
        assert "episode_index" in features
        assert "frame_index" in features
        assert "timestamp" in features
        assert "action" in features

        # Check video features have video_info
        for key, feat in features.items():
            if feat.get("dtype") == "video":
                # lerobot >= 0.5 writes video metadata under "info".
                assert "info" in feat
                assert "video.fps" in feat["info"]
                assert feat["names"] == ["height", "width", "channels"]


# ---------------------------------------------------------------------------
# Spec fixes from the v3 audit (no lerobot needed; parity with lerobot itself
# is in test_lerobot_v3_spec_parity.py)
# ---------------------------------------------------------------------------


def _proprio_episode(ep_id: str, task: str | None, n: int = 10, dim: int = 3, names=None) -> Episode:
    rng = np.random.default_rng(abs(hash(ep_id)) % 2**31)
    states = rng.standard_normal((n, dim)).astype(np.float32)

    def gen():
        for i in range(n):
            yield Frame(index=i, timestamp=i / 10, state=states[i], action=states[i] * 2)

    return Episode(
        episode_id=ep_id,
        fps=10,
        language_instruction=task,
        state_names=names,
        action_names=names,
        _frame_loader=gen,
    )


class TestV3AuditFixes:
    def test_joint_names_written(self, tmp_path):
        names = ["a", "b", "c"]
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        w.write_dataset(iter([_proprio_episode("e0", "t", names=names)]), tmp_path)
        feats = json.loads((tmp_path / "meta/info.json").read_text())["features"]
        assert feats["observation.state"]["names"] == names
        assert feats["action"]["names"] == names

    def test_config_names_override_and_validate(self, tmp_path):
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10, state_names=["x", "y", "z"]))
        w.write_dataset(iter([_proprio_episode("e0", "t", names=["a", "b", "c"])]), tmp_path / "ok")
        feats = json.loads((tmp_path / "ok/meta/info.json").read_text())["features"]
        assert feats["observation.state"]["names"] == ["x", "y", "z"]

        bad = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10, state_names=["x"]))
        with pytest.raises(ConversionError, match="3 dimensions but 1 names"):
            bad.write_dataset(iter([_proprio_episode("e0", "t")]), tmp_path / "bad")

    def test_missing_task_is_an_error_not_default(self, tmp_path):
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        with pytest.raises(ConversionError, match="--task"):
            w.write_dataset(iter([_proprio_episode("e0", None)]), tmp_path)

    def test_config_task_fills_episodes_without_language(self, tmp_path):
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10, task="push the block"))
        w.write_dataset(iter([_proprio_episode("e0", None), _proprio_episode("e1", "own task")]), tmp_path)
        tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
        assert list(tasks.index) == ["push the block", "own task"]

    def test_quantiles_in_stats(self, tmp_path):
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        w.write_dataset(iter([_proprio_episode("e0", "t")]), tmp_path)
        stats = json.loads((tmp_path / "meta/stats.json").read_text())
        for feat in ("observation.state", "action", "timestamp", "index"):
            assert {"q01", "q10", "q50", "q90", "q99"} <= set(stats[feat]), feat

    def test_data_columns_are_float32_fixed_size(self, tmp_path):
        import pyarrow as pa

        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        w.write_dataset(iter([_proprio_episode("e0", "t")]), tmp_path)
        schema = pq.read_schema(tmp_path / "data/chunk-000/file-000.parquet")
        assert schema.field("observation.state").type == pa.list_(pa.float32(), 3)
        assert schema.field("timestamp").type == pa.float32()

    def test_separate_writers_finalize_like_parallel_workers(self, tmp_path):
        """--workers N: each worker numbers its own tasks from 0 and finalize
        runs in a fresh writer. Tasks and global indices must still be right."""
        eps = [_proprio_episode(f"e{i}", t) for i, t in enumerate(["pick", "place", "pick", "stack"])]
        LeRobotV3Writer(LeRobotV3WriterConfig(fps=10)).begin(tmp_path)
        for i, ep in reversed(list(enumerate(eps))):  # out of order, like as_completed
            LeRobotV3Writer(LeRobotV3WriterConfig(fps=10)).write_episode(ep, tmp_path, episode_index=i)
        from forge.core.models import DatasetInfo

        LeRobotV3Writer(LeRobotV3WriterConfig(fps=10)).finalize(
            tmp_path, DatasetInfo(path=tmp_path, format="lerobot-v3")
        )
        tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
        assert list(tasks.index) == ["pick", "place", "stack"]
        episodes = pd.read_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
        assert [list(t) for t in episodes["tasks"]] == [["pick"], ["place"], ["pick"], ["stack"]]
        for i in range(4):
            data = pd.read_parquet(tmp_path / f"data/chunk-{i:03d}/file-000.parquet")
            assert set(data["task_index"]) == {tasks.loc[eps[i].language_instruction, "task_index"]}
            assert list(data["index"]) == list(range(10 * i, 10 * i + 10))
        assert not (tmp_path / "meta/.forge_episodes").exists()

    def test_gap_from_failed_episode_is_closed(self, tmp_path):
        from forge.core.models import DatasetInfo

        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        w.begin(tmp_path)
        w.write_episode(_proprio_episode("e0", "a"), tmp_path, episode_index=0)
        w.write_episode(_proprio_episode("e2", "b"), tmp_path, episode_index=2)  # 1 failed
        w.finalize(tmp_path, DatasetInfo(path=tmp_path, format="lerobot-v3"))
        info = json.loads((tmp_path / "meta/info.json").read_text())
        assert info["total_episodes"] == 2
        data = pd.read_parquet(tmp_path / "data/chunk-001/file-000.parquet")
        assert set(data["episode_index"]) == {1}
        assert not (tmp_path / "data/chunk-002").exists()

    def test_tiny_frames_fall_back_from_svtav1(self, tmp_path):
        """SVT-AV1 never returns on frames ~16 px wide; must not hang."""
        w = LeRobotV3Writer(LeRobotV3WriterConfig(fps=10))
        assert w._video_encoder(16, 16).config.codec == "libx264"
        assert w._video_encoder(640, 480).config.codec in ("libsvtav1", "libx264")
