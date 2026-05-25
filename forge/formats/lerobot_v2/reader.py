"""LeRobot v2 format reader for Forge.

LeRobot v2 datasets use HuggingFace datasets with Parquet files for metadata
and MP4 video files for camera streams.

Structure:
    dataset/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   └── stats.json
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   └── ...
    │   └── ...
    └── videos/
        ├── chunk-000/
        │   ├── observation.images.top/
        │   │   ├── episode_000000.mp4
        │   │   └── ...
        │   └── ...
        └── ...
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.core.exceptions import (
    EpisodeNotFoundError,
    InspectionError,
    MissingDependencyError,
)
from forge.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyImage,
)
from forge.formats.registry import FormatRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_pyarrow() -> None:
    """Check if PyArrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="LeRobot v2 format support",
            install_hint="pip install forge-robotics[lerobot]",
        )


@FormatRegistry.register_reader("lerobot-v2")
class LeRobotV2Reader:
    """Reader for LeRobot v2 datasets.

    LeRobot v2 uses HuggingFace datasets format with:
    - Parquet files for tabular data (states, actions)
    - MP4 video files for camera streams
    - JSON metadata files
    """

    @property
    def format_name(self) -> str:
        return "lerobot-v2"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for LeRobot v2 markers.

        Args:
            path: Path to potential LeRobot v2 dataset.

        Returns:
            True if LeRobot v2 markers found.
        """
        if not path.exists() or not path.is_dir():
            return False

        # Check for meta/info.json (v2 structure)
        meta_dir = path / "meta"
        if meta_dir.exists():
            if (meta_dir / "info.json").exists():
                return True

        # Check for data/ directory with parquet files
        data_dir = path / "data"
        if data_dir.exists():
            if any(data_dir.rglob("*.parquet")):
                return True

        # Check for videos/ directory
        videos_dir = path / "videos"
        if videos_dir.exists():
            if any(videos_dir.rglob("*.mp4")):
                return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect LeRobot version from metadata.

        Args:
            path: Path to dataset.

        Returns:
            Version string or None.
        """
        info_path = path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                return info.get("codebase_version", "2.0")
            except (json.JSONDecodeError, KeyError):
                pass
        return "2.0"

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze LeRobot v2 dataset structure.

        Args:
            path: Path to dataset.

        Returns:
            DatasetInfo with schema and metadata.
        """
        path = Path(path)
        info = DatasetInfo(path=path, format="lerobot-v2")
        info.format_version = self.detect_version(path)

        # Load info.json
        self._load_info_json(path, info)

        # Analyze parquet schema
        self._analyze_parquet_schema(path, info)

        # Detect video streams
        self._detect_video_streams(path, info)

        return info

    def _load_info_json(self, path: Path, info: DatasetInfo) -> None:
        """Load metadata from info.json."""
        info_path = path / "meta" / "info.json"
        if not info_path.exists():
            return

        try:
            with open(info_path) as f:
                data = json.load(f)

            info.num_episodes = data.get("total_episodes", 0)
            info.total_frames = data.get("total_frames", 0)
            info.inferred_fps = data.get("fps")

            # Extract robot type
            info.inferred_robot_type = data.get("robot_type")

            # Check for features info
            if "features" in data:
                features = data["features"]
                for key, spec in features.items():
                    if "image" in key.lower():
                        shape = spec.get("shape", [480, 640, 3])
                        # Extract camera name from key like "observation.images.top"
                        parts = key.split(".")
                        cam_name = parts[-1] if len(parts) > 1 else key
                        dtype_str = spec.get("dtype", "")
                        storage = (
                            "mp4" if dtype_str == "video"
                            else "image" if dtype_str == "image"
                            else "unknown"
                        )
                        info.cameras[cam_name] = CameraInfo(
                            name=cam_name,
                            height=shape[0] if len(shape) > 0 else 480,
                            width=shape[1] if len(shape) > 1 else 640,
                            channels=shape[2] if len(shape) > 2 else 3,
                            storage=storage,
                        )
                    elif "state" in key.lower() or "action" in key.lower():
                        shape = tuple(spec.get("shape", []))
                        dtype_str = spec.get("dtype", "float32")
                        field = FieldSchema(
                            name=key,
                            shape=shape,
                            dtype=self._str_to_dtype(dtype_str),
                        )
                        if "action" in key.lower():
                            info.action_schema = field
                        else:
                            info.observation_schema[key] = field

        except (json.JSONDecodeError, KeyError) as e:
            raise InspectionError(path, f"Failed to parse info.json: {e}")

    def _str_to_dtype(self, dtype_str: str) -> Dtype:
        """Convert string dtype to Forge Dtype."""
        dtype_str = dtype_str.lower()
        mapping = {
            "float32": Dtype.FLOAT32,
            "float64": Dtype.FLOAT64,
            "int32": Dtype.INT32,
            "int64": Dtype.INT64,
            "uint8": Dtype.UINT8,
            "bool": Dtype.BOOL,
            "string": Dtype.STRING,
        }
        return mapping.get(dtype_str, Dtype.FLOAT32)

    def _analyze_parquet_schema(self, path: Path, info: DatasetInfo) -> None:
        """Analyze schema from parquet files."""
        _check_pyarrow()
        import pyarrow.parquet as pq

        # Find first parquet file
        data_dir = path / "data"
        if not data_dir.exists():
            return

        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            return

        try:
            # Read schema from first file
            schema = pq.read_schema(parquet_files[0])

            for field in schema:
                name = field.name
                # Skip index columns
                if name in ("episode_index", "frame_index", "index", "timestamp"):
                    if name == "timestamp":
                        info.has_timestamps = True
                    continue

                # Infer shape and dtype from arrow type
                arrow_type = field.type
                dtype = self._arrow_to_dtype(arrow_type)

                # Try to get shape from metadata or type
                shape = self._infer_shape_from_arrow(arrow_type)

                if "action" in name.lower() and info.action_schema is None:
                    info.action_schema = FieldSchema(name=name, shape=shape, dtype=dtype)
                elif "image" not in name.lower():
                    info.observation_schema[name] = FieldSchema(name=name, shape=shape, dtype=dtype)

        except Exception:
            pass  # Schema analysis is optional

    def _arrow_to_dtype(self, arrow_type: Any) -> Dtype:
        """Convert PyArrow type to Forge Dtype."""
        import pyarrow as pa

        if pa.types.is_float32(arrow_type):
            return Dtype.FLOAT32
        elif pa.types.is_float64(arrow_type):
            return Dtype.FLOAT64
        elif pa.types.is_int32(arrow_type):
            return Dtype.INT32
        elif pa.types.is_int64(arrow_type):
            return Dtype.INT64
        elif pa.types.is_uint8(arrow_type):
            return Dtype.UINT8
        elif pa.types.is_boolean(arrow_type):
            return Dtype.BOOL
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return Dtype.STRING
        elif pa.types.is_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
            # Recurse into list type
            return self._arrow_to_dtype(arrow_type.value_type)
        else:
            return Dtype.FLOAT32

    def _infer_shape_from_arrow(self, arrow_type: Any) -> tuple[int, ...]:
        """Infer array shape from PyArrow type."""
        import pyarrow as pa

        if pa.types.is_fixed_size_list(arrow_type):
            inner_shape = self._infer_shape_from_arrow(arrow_type.value_type)
            return (arrow_type.list_size,) + inner_shape
        elif pa.types.is_list(arrow_type):
            # Variable length list - unknown size
            return (-1,)
        else:
            return ()

    def _detect_video_streams(self, path: Path, info: DatasetInfo) -> None:
        """Detect camera streams from video directories."""
        videos_dir = path / "videos"
        if not videos_dir.exists():
            return

        # Look for camera directories
        for chunk_dir in videos_dir.iterdir():
            if not chunk_dir.is_dir():
                continue

            for cam_dir in chunk_dir.iterdir():
                if not cam_dir.is_dir():
                    continue

                # Extract camera name from path like "observation.images.top"
                cam_name = cam_dir.name.split(".")[-1]

                if cam_name not in info.cameras:
                    # Try to get dimensions from first video
                    mp4_files = list(cam_dir.glob("*.mp4"))
                    if mp4_files:
                        dims = self._get_video_dimensions(mp4_files[0])
                        info.cameras[cam_name] = CameraInfo(
                            name=cam_name,
                            height=dims[0],
                            width=dims[1],
                            channels=3,
                            storage="mp4",
                        )

    def _get_video_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Get video dimensions using ffprobe or av."""
        try:
            import av

            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                return (stream.height, stream.width)
        except ImportError:
            pass

        # Fallback to default
        return (480, 640)

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over LeRobot v2 episodes.

        Args:
            path: Path to dataset.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_pyarrow()

        path = Path(path)
        data_dir = path / "data"

        if not data_dir.exists():
            return

        # Find all parquet files
        parquet_files = sorted(data_dir.rglob("*.parquet"))

        for pq_file in parquet_files:
            # Extract episode ID from filename
            episode_id = pq_file.stem  # e.g., "episode_000000"

            yield self._load_episode(path, pq_file, episode_id)

    def _load_episode(self, dataset_path: Path, parquet_path: Path, episode_id: str) -> Episode:
        """Load a single episode from parquet file."""
        import pyarrow.parquet as pq

        # Read parquet data
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Find video files for this episode
        videos_dir = dataset_path / "videos"
        video_paths: dict[str, Path] = {}

        if videos_dir.exists():
            for chunk_dir in videos_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue
                for cam_dir in chunk_dir.iterdir():
                    if not cam_dir.is_dir():
                        continue
                    cam_name = cam_dir.name.split(".")[-1]
                    video_file = cam_dir / f"{episode_id}.mp4"
                    if video_file.exists():
                        video_paths[cam_name] = video_file

        def load_frames() -> Iterator[Frame]:
            for idx, row in df.iterrows():
                # Create lazy image loaders for each camera
                images: dict[str, LazyImage] = {}
                for cam_name, video_path in video_paths.items():
                    frame_idx = int(idx)

                    def make_loader(vp: Path = video_path, fi: int = frame_idx) -> NDArray[Any]:
                        return self._extract_frame(vp, fi)

                    images[cam_name] = LazyImage(
                        loader=make_loader,
                        height=480,  # Would get from video metadata
                        width=640,
                        channels=3,
                    )

                # Extract state and action
                state = None
                action = None

                for col in df.columns:
                    if "state" in col.lower():
                        val = row[col]
                        if hasattr(val, "__len__"):
                            import numpy as np

                            state = np.array(val, dtype=np.float32)
                    elif "action" in col.lower():
                        val = row[col]
                        if hasattr(val, "__len__"):
                            import numpy as np

                            action = np.array(val, dtype=np.float32)

                timestamp = row.get("timestamp", None)

                yield Frame(
                    index=int(idx),
                    timestamp=float(timestamp) if timestamp is not None else None,
                    images=images,
                    state=state,
                    action=action,
                    is_first=(idx == 0),
                    is_last=(idx == len(df) - 1),
                )

        return Episode(
            episode_id=episode_id,
            _frame_loader=load_frames,
            metadata={"parquet_path": str(parquet_path)},
        )

    def _extract_frame(self, video_path: Path, frame_index: int) -> NDArray[Any]:
        """Extract a single frame from video using efficient seeking."""
        try:
            import av
            import numpy as np

            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]

                # Use keyframe seeking for efficiency
                if frame_index > 0 and stream.duration:
                    time_base = stream.time_base
                    fps = float(stream.average_rate) if stream.average_rate else 30.0
                    target_pts = int(frame_index / fps / time_base)
                    container.seek(target_pts, stream=stream, backward=True, any_frame=False)

                # Decode from keyframe to target frame
                for frame in container.decode(stream):
                    if stream.average_rate:
                        fps = float(stream.average_rate)
                        current_frame = int(frame.pts * time_base * fps) if frame.pts else 0
                    else:
                        current_frame = 0

                    if current_frame >= frame_index:
                        return frame.to_ndarray(format="rgb24")

            # If we didn't find the frame, return black
            return np.zeros((480, 640, 3), dtype=np.uint8)

        except (ImportError, Exception):
            import numpy as np

            return np.zeros((480, 640, 3), dtype=np.uint8)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to dataset.
            episode_id: Episode identifier.

        Returns:
            The requested Episode.

        Raises:
            EpisodeNotFoundError: If episode not found.
        """
        path = Path(path)
        data_dir = path / "data"

        # Look for matching parquet file
        for pq_file in data_dir.rglob("*.parquet"):
            if pq_file.stem == episode_id:
                return self._load_episode(path, pq_file, episode_id)

        raise EpisodeNotFoundError(episode_id, path)
