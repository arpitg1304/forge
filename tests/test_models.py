"""Tests for core data models."""

from pathlib import Path

import numpy as np

from forge.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyImage,
)


class TestDtype:
    """Tests for Dtype enum."""

    def test_dtype_values(self):
        """Test that expected dtypes exist."""
        assert Dtype.FLOAT32.value == "float32"
        assert Dtype.FLOAT64.value == "float64"
        assert Dtype.INT32.value == "int32"
        assert Dtype.INT64.value == "int64"
        assert Dtype.UINT8.value == "uint8"
        assert Dtype.BOOL.value == "bool"
        assert Dtype.STRING.value == "string"

    def test_to_numpy_dtype(self):
        """Test conversion to numpy dtype."""
        assert Dtype.FLOAT32.to_numpy_dtype() == "float32"
        assert Dtype.STRING.to_numpy_dtype() == "object"


class TestFieldSchema:
    """Tests for FieldSchema dataclass."""

    def test_basic_schema(self):
        """Test creating a basic field schema."""
        schema = FieldSchema(name="state", shape=(7,), dtype=Dtype.FLOAT32)
        assert schema.name == "state"
        assert schema.shape == (7,)
        assert schema.dtype == Dtype.FLOAT32
        assert schema.description is None

    def test_schema_with_description(self):
        """Test schema with description."""
        schema = FieldSchema(
            name="action",
            shape=(6,),
            dtype=Dtype.FLOAT32,
            description="End-effector delta pose",
        )
        assert schema.description == "End-effector delta pose"


class TestCameraInfo:
    """Tests for CameraInfo dataclass."""

    def test_rgb_camera(self):
        """Test RGB camera info."""
        cam = CameraInfo(name="wrist_cam", height=480, width=640, channels=3)
        assert cam.name == "wrist_cam"
        assert cam.height == 480
        assert cam.width == 640
        assert cam.channels == 3

    def test_depth_camera(self):
        """Test depth camera info."""
        cam = CameraInfo(name="depth", height=480, width=640, channels=1)
        assert cam.channels == 1

    def test_storage_defaults_to_unknown(self):
        cam = CameraInfo(name="cam", height=64, width=64)
        assert cam.storage == "unknown"

    def test_storage_inline_image(self):
        cam = CameraInfo(name="cam", height=64, width=64, storage="image")
        assert cam.storage == "image"

    def test_storage_mp4(self):
        cam = CameraInfo(name="cam", height=64, width=64, storage="mp4")
        assert cam.storage == "mp4"


class TestLazyImage:
    """Tests for LazyImage class."""

    def test_lazy_loading(self):
        """Test that image is loaded lazily."""
        load_count = [0]

        def loader():
            load_count[0] += 1
            return np.zeros((480, 640, 3), dtype=np.uint8)

        img = LazyImage(loader=loader, height=480, width=640, channels=3)

        # Should not load yet
        assert load_count[0] == 0
        assert img.height == 480
        assert img.width == 640

        # Should load on first access
        data = img.load()
        assert load_count[0] == 1
        assert data.shape == (480, 640, 3)

        # Should cache and not reload
        data2 = img.load()
        assert load_count[0] == 1
        assert data2 is data


class TestFrame:
    """Tests for Frame dataclass."""

    def test_basic_frame(self):
        """Test creating a basic frame."""
        frame = Frame(index=0, is_first=True)
        assert frame.index == 0
        assert frame.is_first is True
        assert frame.is_last is False
        assert frame.images == {}
        assert frame.state is None
        assert frame.action is None

    def test_frame_with_data(self):
        """Test frame with all data."""
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.1, 0.2])

        frame = Frame(
            index=5,
            state=state,
            action=action,
            reward=1.0,
            timestamp=0.5,
            is_first=False,
            is_last=True,
            is_terminal=True,
        )

        assert frame.index == 5
        assert np.array_equal(frame.state, state)
        assert np.array_equal(frame.action, action)
        assert frame.reward == 1.0
        assert frame.timestamp == 0.5
        assert frame.is_last is True
        assert frame.is_terminal is True


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_basic_episode(self):
        """Test creating a basic episode."""
        episode = Episode(episode_id="ep_001")
        assert episode.episode_id == "ep_001"
        assert episode.language_instruction is None

    def test_episode_with_language(self):
        """Test episode with language instruction."""
        episode = Episode(
            episode_id="ep_002",
            language_instruction="Pick up the red cube",
        )
        assert episode.language_instruction == "Pick up the red cube"

    def test_episode_frames_iterator(self):
        """Test episode frame iterator."""
        frames = [
            Frame(index=0, is_first=True),
            Frame(index=1),
            Frame(index=2, is_last=True),
        ]

        def loader():
            yield from frames

        episode = Episode(episode_id="test", _frame_loader=loader)

        # Iterate through frames
        loaded_frames = list(episode.frames())
        assert len(loaded_frames) == 3
        assert loaded_frames[0].index == 0
        assert loaded_frames[2].is_last is True


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_basic_info(self):
        """Test creating basic dataset info."""
        info = DatasetInfo(path=Path("/data/test"), format="zarr")
        assert info.path == Path("/data/test")
        assert info.format == "zarr"
        assert info.num_episodes == 0
        assert info.cameras == {}
        assert info.observation_schema == {}

    def test_info_with_cameras(self):
        """Test dataset info with cameras."""
        info = DatasetInfo(path=Path("/data/test"), format="lerobot-v3")
        info.cameras["wrist"] = CameraInfo(name="wrist", height=480, width=640, channels=3)
        info.cameras["overhead"] = CameraInfo(name="overhead", height=720, width=1280, channels=3)

        assert len(info.cameras) == 2
        assert info.cameras["wrist"].height == 480
        assert info.cameras["overhead"].width == 1280

    def test_missing_required_default_empty(self):
        """Test missing_required defaults to empty list."""
        info = DatasetInfo(path=Path("/data/test"), format="zarr")
        # missing_required is set by the inspector, defaults to empty
        assert info.missing_required == []

    def test_missing_required_set_externally(self):
        """Test missing_required can be set."""
        info = DatasetInfo(path=Path("/data/test"), format="zarr")
        info.missing_required = ["fps", "robot_type"]

        assert "fps" in info.missing_required
        assert "robot_type" in info.missing_required

    def test_is_ready_for_conversion(self):
        """Test conversion readiness check."""
        info = DatasetInfo(path=Path("/data/test"), format="zarr")

        # Empty missing_required means ready
        assert info.is_ready_for_conversion("lerobot-v3") is True

        # With missing items, not ready
        info.missing_required = ["fps"]
        assert info.is_ready_for_conversion("lerobot-v3") is False

        # Clear missing, ready again
        info.missing_required = []
        assert info.is_ready_for_conversion("lerobot-v3") is True
