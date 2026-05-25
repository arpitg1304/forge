"""LeRobot v3 format reader for Forge.

LeRobot v3 is the platform's lingua franca - the standard format that
all downstream tools expect. It builds on v2 with improved metadata
and standardized naming conventions.

Structure:
    dataset/
    ├── meta/
    │   ├── info.json           # Dataset metadata, features, robot info
    │   ├── episodes.jsonl      # Per-episode metadata
    │   ├── stats.json          # Statistics for normalization
    │   └── tasks.jsonl         # Task/language annotations
    ├── data/
    │   └── train/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── train/
            ├── observation.images.top/
            │   ├── episode_000000.mp4
            │   └── ...
            └── ...
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
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


# Global video frame cache for efficient sequential access
_video_cache: dict[str, "VideoFrameCache"] = {}
_video_cache_lock = Lock()


class VideoFrameCache:
    """Cache for efficient sequential video frame access.

    Keeps the video container open and decodes frames sequentially
    for much faster playback than seeking for each frame.
    """

    def __init__(self, video_path: Path, max_cached_frames: int = 32):
        self.video_path = video_path
        self.max_cached = max_cached_frames
        self._container = None
        self._stream = None
        self._current_frame_idx = -1
        self._frame_cache: dict[int, Any] = {}  # frame_idx -> ndarray
        self._lock = Lock()
        self._fps = 30.0
        self._total_frames = 0

    def _open(self) -> bool:
        """Open video container if not already open."""
        if self._container is not None:
            return True
        try:
            import av
            self._container = av.open(str(self.video_path))
            self._stream = self._container.streams.video[0]
            self._fps = float(self._stream.average_rate) if self._stream.average_rate else 30.0
            self._total_frames = self._stream.frames or 0
            return True
        except Exception:
            return False

    def get_frame(self, frame_idx: int, dims: tuple[int, int, int] = (480, 640, 3)) -> Any:
        """Get a frame, using cache or sequential decode."""
        import numpy as np

        with self._lock:
            # Check cache first
            if frame_idx in self._frame_cache:
                return self._frame_cache[frame_idx]

            if not self._open():
                return np.zeros(dims, dtype=np.uint8)

            try:
                # If we need to go backwards or skip far ahead, seek
                if frame_idx < self._current_frame_idx or frame_idx > self._current_frame_idx + 30:
                    self._seek_to_frame(frame_idx)

                # Decode frames until we reach target
                for frame in self._container.decode(self._stream):
                    self._current_frame_idx += 1
                    img = frame.to_ndarray(format="rgb24")

                    # Cache this frame
                    self._frame_cache[self._current_frame_idx] = img

                    # Prune cache if too large
                    if len(self._frame_cache) > self.max_cached:
                        # Remove frames far from current
                        to_remove = [k for k in self._frame_cache
                                   if abs(k - frame_idx) > self.max_cached // 2]
                        for k in to_remove[:len(self._frame_cache) - self.max_cached]:
                            del self._frame_cache[k]

                    if self._current_frame_idx >= frame_idx:
                        return img

                # Reached end of video
                return np.zeros(dims, dtype=np.uint8)

            except Exception:
                return np.zeros(dims, dtype=np.uint8)

    def _seek_to_frame(self, target_frame: int) -> None:
        """Seek to a frame position."""
        if self._stream is None:
            return
        try:
            time_base = self._stream.time_base
            target_pts = int(target_frame / self._fps / time_base)
            self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
            # Reset current position - we'll update it as we decode
            self._current_frame_idx = max(0, target_frame - 10)  # Estimate
        except Exception:
            pass

    def close(self) -> None:
        """Close the video container."""
        with self._lock:
            if self._container is not None:
                self._container.close()
                self._container = None
                self._stream = None
            self._frame_cache.clear()


def get_video_cache(video_path: Path) -> VideoFrameCache:
    """Get or create a video frame cache for a path."""
    key = str(video_path)
    with _video_cache_lock:
        if key not in _video_cache:
            _video_cache[key] = VideoFrameCache(video_path)
        return _video_cache[key]


def _check_pyarrow() -> None:
    """Check if PyArrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="LeRobot v3 format support",
            install_hint="pip install forge-robotics[lerobot]",
        )


@FormatRegistry.register_reader("lerobot-v3")
class LeRobotV3Reader:
    """Reader for LeRobot v3 datasets.

    LeRobot v3 is the standardized format for the robotics data platform.
    Key differences from v2:
    - Stricter naming conventions
    - Required fields in info.json
    - Task annotations in tasks.jsonl
    - Train/test split organization
    """

    @property
    def format_name(self) -> str:
        return "lerobot-v3"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for LeRobot v3 markers.

        V3 is distinguished from v2 by:
        - codebase_version >= 3.0 in info.json
        - Presence of tasks.jsonl
        - train/ subdirectory structure

        Args:
            path: Path to potential LeRobot v3 dataset.

        Returns:
            True if LeRobot v3 markers found.
        """
        if not path.exists() or not path.is_dir():
            return False

        # Check for meta/info.json with version >= 3.0
        info_path = path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)

                version = info.get("codebase_version", "")
                if isinstance(version, str):
                    # Parse version string like "3.0" or "v3.0"
                    version = version.lstrip("v")
                    try:
                        major = int(version.split(".")[0])
                        if major >= 3:
                            return True
                    except (ValueError, IndexError):
                        pass
                elif isinstance(version, (int, float)) and version >= 3:
                    return True

            except (json.JSONDecodeError, KeyError):
                pass

        # Check for v3-specific files (tasks.jsonl or tasks.parquet)
        if (path / "meta" / "tasks.jsonl").exists():
            return True
        if (path / "meta" / "tasks.parquet").exists():
            return True

        # Check for train/ subdirectory structure (v3 convention)
        data_train = path / "data" / "train"
        if data_train.exists() and any(data_train.glob("*.parquet")):
            return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect specific LeRobot version.

        Args:
            path: Path to dataset.

        Returns:
            Version string like "3.0".
        """
        info_path = path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                version = info.get("codebase_version", "3.0")
                if isinstance(version, str):
                    return version.lstrip("v")
                return str(version)
            except (json.JSONDecodeError, KeyError):
                pass
        return "3.0"

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze LeRobot v3 dataset structure.

        Args:
            path: Path to dataset.

        Returns:
            DatasetInfo with schema and metadata.
        """
        path = Path(path)
        info = DatasetInfo(path=path, format="lerobot-v3")
        info.format_version = self.detect_version(path)

        # Load info.json (required for v3)
        self._load_info_json(path, info)

        # Load episodes metadata (jsonl or parquet)
        self._load_episodes_metadata(path, info)

        # Load tasks (jsonl or parquet)
        self._load_tasks(path, info)

        # Analyze parquet schema
        self._analyze_parquet_schema(path, info)

        # Detect video streams
        self._detect_video_streams(path, info)

        return info

    def _load_info_json(self, path: Path, info: DatasetInfo) -> None:
        """Load metadata from info.json."""
        info_path = path / "meta" / "info.json"
        if not info_path.exists():
            raise InspectionError(path, "Missing meta/info.json (required for v3)")

        try:
            with open(info_path) as f:
                data = json.load(f)

            # Required fields in v3
            info.num_episodes = data.get("total_episodes", 0)
            info.total_frames = data.get("total_frames", 0)
            info.inferred_fps = data.get("fps")
            info.inferred_robot_type = data.get("robot_type")

            # Features specification
            if "features" in data:
                for key, spec in data["features"].items():
                    dtype_str = spec.get("dtype", "float32")
                    shape = tuple(spec.get("shape", []))

                    # Skip index/metadata columns
                    if key in ("timestamp", "episode_index", "frame_index", "index", "task_index"):
                        if key == "timestamp":
                            info.has_timestamps = True
                        continue

                    # Detect cameras (dtype == "video" is the v3 convention;
                    # dtype == "image" means inline image bytes in parquet)
                    is_video = dtype_str == "video"
                    is_inline_image = dtype_str == "image"
                    if is_video or is_inline_image or ("image" in key.lower() and len(shape) == 3):
                        cam_name = key.split(".")[-1]
                        # Shape is [height, width, channels] in v3
                        if len(shape) >= 2:
                            h, w = shape[0], shape[1]
                            c = shape[2] if len(shape) > 2 else 3
                            storage = "mp4" if is_video else ("image" if is_inline_image else "unknown")
                            info.cameras[cam_name] = CameraInfo(
                                name=cam_name,
                                height=h,
                                width=w,
                                channels=c,
                                storage=storage,
                            )
                    elif "action" in key.lower():
                        info.action_schema = FieldSchema(
                            name=key,
                            shape=shape,
                            dtype=self._str_to_dtype(dtype_str),
                        )
                    elif key not in ("next.reward", "next.done"):
                        # Regular observation field
                        info.observation_schema[key] = FieldSchema(
                            name=key,
                            shape=shape,
                            dtype=self._str_to_dtype(dtype_str),
                        )

            # Check for timestamps
            info.has_timestamps = data.get("has_timestamps", True)

        except json.JSONDecodeError as e:
            raise InspectionError(path, f"Invalid JSON in info.json: {e}")

    def _str_to_dtype(self, dtype_str: str) -> Dtype:
        """Convert string dtype to Forge Dtype."""
        dtype_str = str(dtype_str).lower()
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

    def _load_episodes_metadata(self, path: Path, info: DatasetInfo) -> None:
        """Load episode metadata from episodes.jsonl or episodes/ directory."""
        # Check for episodes.jsonl
        episodes_jsonl = path / "meta" / "episodes.jsonl"
        if episodes_jsonl.exists():
            self._load_episodes_jsonl(episodes_jsonl, info)
            return

        # Check for episodes/ directory with parquet files (v3 convention)
        episodes_dir = path / "meta" / "episodes"
        if episodes_dir.exists():
            self._load_episodes_parquet_dir(episodes_dir, info)

    def _load_episodes_jsonl(self, episodes_path: Path, info: DatasetInfo) -> None:
        """Load episodes from JSONL format."""
        try:
            episode_count = 0
            total_frames = 0
            has_success = False

            with open(episodes_path) as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        episode_count += 1
                        total_frames += ep.get("length", 0)
                        if "success" in ep:
                            has_success = True

            if episode_count > 0 and info.num_episodes == 0:
                info.num_episodes = episode_count
            if total_frames > 0 and info.total_frames == 0:
                info.total_frames = total_frames
            info.has_success_labels = has_success

        except (json.JSONDecodeError, KeyError):
            pass

    def _load_episodes_parquet_dir(self, episodes_dir: Path, info: DatasetInfo) -> None:
        """Load episodes from parquet files in episodes/ directory."""
        try:
            _check_pyarrow()
            import pyarrow.parquet as pq

            parquet_files = list(episodes_dir.glob("*.parquet"))
            if not parquet_files:
                return

            episode_count = 0
            has_success = False

            for pq_file in parquet_files:
                table = pq.read_table(pq_file)
                episode_count += len(table)

                # Check for success column
                if "success" in table.column_names:
                    has_success = True

            if episode_count > 0 and info.num_episodes == 0:
                info.num_episodes = episode_count
            info.has_success_labels = has_success

        except Exception:
            pass

    def _load_tasks(self, path: Path, info: DatasetInfo) -> None:
        """Load task/language annotations from tasks.jsonl or tasks.parquet."""
        # Try jsonl first
        tasks_jsonl = path / "meta" / "tasks.jsonl"
        tasks_parquet = path / "meta" / "tasks.parquet"

        if tasks_jsonl.exists():
            self._load_tasks_jsonl(tasks_jsonl, info)
        elif tasks_parquet.exists():
            self._load_tasks_parquet(tasks_parquet, info)

    def _load_tasks_jsonl(self, tasks_path: Path, info: DatasetInfo) -> None:
        """Load tasks from JSONL format."""
        try:
            task_count = 0
            sample_task = None

            with open(tasks_path) as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        task_count += 1
                        if sample_task is None:
                            sample_task = task.get("task", task.get("language"))

            if task_count > 0:
                info.has_language = True
                info.language_coverage = 1.0
                info.sample_language = sample_task

        except (json.JSONDecodeError, KeyError):
            pass

    def _load_tasks_parquet(self, tasks_path: Path, info: DatasetInfo) -> None:
        """Load tasks from Parquet format."""
        try:
            _check_pyarrow()
            import pyarrow.parquet as pq

            table = pq.read_table(tasks_path)
            df = table.to_pandas()

            if len(df) > 0:
                info.has_language = True
                info.language_coverage = 1.0

                # In v3 format, task text may be in index or in a column
                # Look for task column first
                for col in ["task", "language", "instruction"]:
                    if col in df.columns:
                        info.sample_language = str(df[col].iloc[0])
                        return

                # If task is in index (common v3 pattern)
                if df.index.dtype == object:  # String index
                    info.sample_language = str(df.index[0])

        except Exception:
            pass

    def _analyze_parquet_schema(self, path: Path, info: DatasetInfo) -> None:
        """Analyze schema from parquet files."""
        _check_pyarrow()
        import pyarrow.parquet as pq

        # V3 uses data/train/ structure
        data_dir = path / "data" / "train"
        if not data_dir.exists():
            data_dir = path / "data"

        if not data_dir.exists():
            return

        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            return

        try:
            schema = pq.read_schema(parquet_files[0])

            for field in schema:
                name = field.name

                # Skip index/metadata columns
                if name in ("episode_index", "frame_index", "index", "timestamp", "task_index"):
                    continue

                dtype = self._arrow_to_dtype(field.type)
                shape = self._infer_shape_from_arrow(field.type)

                # Update existing schemas or add new ones
                if "action" in name.lower():
                    if info.action_schema is None:
                        info.action_schema = FieldSchema(name=name, shape=shape, dtype=dtype)
                elif "image" not in name.lower() and name not in info.observation_schema:
                    info.observation_schema[name] = FieldSchema(name=name, shape=shape, dtype=dtype)

            # Get sample episode info
            if parquet_files:
                info.sample_episode_id = parquet_files[0].stem
                table = pq.read_table(parquet_files[0])
                info.sample_num_frames = len(table)

        except Exception:
            pass

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
            return (-1,)
        else:
            return ()

    def _detect_video_streams(self, path: Path, info: DatasetInfo) -> None:
        """Detect camera streams from video directories."""
        videos_dir = path / "videos"
        if not videos_dir.exists():
            return

        for cam_dir in videos_dir.iterdir():
            if not cam_dir.is_dir():
                continue

            # Skip non-camera directories (train/test splits, etc.)
            cam_dir_name = cam_dir.name
            if cam_dir_name in ("train", "test", "val", "validation"):
                # This is a split directory, look inside
                for sub_cam_dir in cam_dir.iterdir():
                    if sub_cam_dir.is_dir():
                        self._add_camera_from_dir(sub_cam_dir, info)
                continue

            # Parse camera name from "observation.images.exterior_image_1_left" format
            self._add_camera_from_dir(cam_dir, info)

    def _add_camera_from_dir(self, cam_dir: Path, info: DatasetInfo) -> None:
        """Add camera info from a camera directory."""
        cam_dir_name = cam_dir.name

        # Skip chunk directories
        if cam_dir_name.startswith("chunk-"):
            return

        # Parse camera name from "observation.images.XXX" format
        cam_name = cam_dir_name.split(".")[-1]

        if cam_name in info.cameras:
            return  # Already have this camera from info.json

        # Find video files (may be in chunk subdirs)
        mp4_files = list(cam_dir.rglob("*.mp4"))
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
        """Get video dimensions."""
        try:
            import av

            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                return (stream.height, stream.width)
        except ImportError:
            pass

        return (480, 640)

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over LeRobot v3 episodes.

        Args:
            path: Path to dataset.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_pyarrow()
        import pyarrow.parquet as pq

        path = Path(path)

        # V3 uses data/train/ structure
        data_dir = path / "data" / "train"
        if not data_dir.exists():
            data_dir = path / "data"

        if not data_dir.exists():
            return

        # Load tasks for language annotations
        tasks = self._load_all_tasks(path)

        # Load episode metadata
        episode_meta = self._load_episode_metadata(path)

        # Find all parquet files
        parquet_files = sorted(data_dir.rglob("*.parquet"))

        # Track which episodes we've already yielded (in case same episode spans files)
        yielded_episodes: set[int] = set()

        for pq_file in parquet_files:
            # Read parquet to find all episode indices in this file
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            # Check if this file has episode_index column (multi-episode file)
            if "episode_index" in df.columns:
                # Get unique episode indices in this file, sorted
                episode_indices = sorted(df["episode_index"].unique())

                for episode_idx in episode_indices:
                    # Skip if we've already yielded this episode
                    if episode_idx in yielded_episodes:
                        continue
                    yielded_episodes.add(episode_idx)

                    episode_id = f"episode_{episode_idx:06d}"

                    # Get language instruction for this episode
                    language = None
                    if episode_meta and episode_idx in episode_meta:
                        meta = episode_meta[episode_idx]
                        task_idx = meta.get("task_index", 0)
                        if tasks and task_idx < len(tasks):
                            language = tasks[task_idx]

                    yield self._load_episode_from_dataframe(
                        path, pq_file, df, episode_idx, episode_id, language
                    )
            else:
                # Single episode per file (old format)
                episode_id = pq_file.stem
                episode_idx = self._parse_episode_index(episode_id)

                if episode_idx in yielded_episodes:
                    continue
                yielded_episodes.add(episode_idx)

                # Get language instruction for this episode
                language = None
                if episode_meta and episode_idx in episode_meta:
                    meta = episode_meta[episode_idx]
                    task_idx = meta.get("task_index", 0)
                    if tasks and task_idx < len(tasks):
                        language = tasks[task_idx]

                yield self._load_episode(path, pq_file, episode_id, language)

    def _parse_episode_index(self, episode_id: str) -> int:
        """Parse episode index from ID like 'episode_000123'."""
        try:
            return int(episode_id.replace("episode_", ""))
        except ValueError:
            return 0

    def _load_all_tasks(self, path: Path) -> list[str]:
        """Load all task descriptions."""
        tasks_path = path / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            return []

        tasks = []
        try:
            with open(tasks_path) as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        tasks.append(task.get("task", task.get("language", "")))
        except (json.JSONDecodeError, KeyError):
            pass

        return tasks

    def _load_episode_metadata(self, path: Path) -> dict[int, dict]:
        """Load per-episode metadata."""
        episodes_path = path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            return {}

        metadata = {}
        try:
            with open(episodes_path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        metadata[i] = json.loads(line)
        except (json.JSONDecodeError, KeyError):
            pass

        return metadata

    def _load_episode_from_dataframe(
        self,
        dataset_path: Path,
        parquet_path: Path,
        df: Any,  # pandas DataFrame
        episode_idx: int,
        episode_id: str,
        language: str | None,
    ) -> Episode:
        """Load a single episode from an already-loaded dataframe.

        This is used when a parquet file contains multiple episodes.
        We filter the dataframe to just the rows for this episode.
        """
        # Filter to just this episode's rows
        ep_df = df[df["episode_index"] == episode_idx].reset_index(drop=True)

        return self._build_episode(
            dataset_path, parquet_path, ep_df, episode_idx, episode_id, language
        )

    def _load_episode(
        self,
        dataset_path: Path,
        parquet_path: Path,
        episode_id: str,
        language: str | None,
    ) -> Episode:
        """Load a single episode from a parquet file."""
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        episode_idx = self._parse_episode_index(episode_id)

        # Filter to just this episode if multiple in file
        if "episode_index" in df.columns:
            ep_df = df[df["episode_index"] == episode_idx].reset_index(drop=True)
        else:
            ep_df = df

        return self._build_episode(
            dataset_path, parquet_path, ep_df, episode_idx, episode_id, language
        )

    def _build_episode(
        self,
        dataset_path: Path,
        parquet_path: Path,
        ep_df: Any,  # pandas DataFrame filtered to one episode
        episode_idx: int,
        episode_id: str,
        language: str | None,
    ) -> Episode:
        """Build an Episode object from a filtered dataframe."""

        # Find video files - handle both old and chunked formats
        video_paths: dict[str, Path] = {}
        video_dims: dict[str, tuple[int, int, int]] = {}

        # Get camera info from info.json for dimensions
        info_path = dataset_path / "meta" / "info.json"
        camera_features: dict[str, dict] = {}
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                for key, spec in info.get("features", {}).items():
                    if spec.get("dtype") == "video":
                        camera_features[key] = spec

        # Get video path template from info.json
        video_path_template = None
        if info_path.exists():
            with open(info_path) as f:
                info_data = json.load(f)
                video_path_template = info_data.get("video_path")

        videos_dir = dataset_path / "videos"
        if videos_dir.exists():
            # Calculate episode chunk for chunked formats
            info_path2 = dataset_path / "meta" / "info.json"
            chunks_size = 1000
            if info_path2.exists():
                with open(info_path2) as f:
                    info_data = json.load(f)
                    chunks_size = info_data.get("chunks_size", 1000)
            episode_chunk = episode_idx // chunks_size

            # Handle different video path patterns
            # Pattern 1: videos/chunk-XXX/{video_key}/episode_XXX.mp4 (chunk-first)
            # Pattern 2: videos/{video_key}/chunk-XXX/file-XXX.mp4 (camera-first)
            # Pattern 3: videos/train/{video_key}/episode_XXX.mp4 (split-based)

            # Check for chunk-first structure: videos/chunk-XXX/...
            chunk_dir = videos_dir / f"chunk-{episode_chunk:03d}"
            if chunk_dir.exists() and chunk_dir.is_dir():
                # Chunk-first format: videos/chunk-XXX/{video_key}/episode_XXX.mp4
                for cam_dir in sorted(chunk_dir.iterdir(), key=lambda p: p.name):
                    if not cam_dir.is_dir():
                        continue
                    cam_key = cam_dir.name
                    cam_name = cam_key.split(".")[-1]
                    video_file = cam_dir / f"{episode_id}.mp4"
                    if video_file.exists():
                        video_paths[cam_name] = video_file
                        if cam_key in camera_features:
                            shape = camera_features[cam_key].get("shape", [480, 640, 3])
                            video_dims[cam_name] = (shape[0], shape[1], shape[2] if len(shape) > 2 else 3)
            else:
                # Sort camera directories for consistent ordering across runs
                for cam_dir in sorted(videos_dir.iterdir(), key=lambda p: p.name):
                    if not cam_dir.is_dir():
                        continue

                    cam_dir_name = cam_dir.name

                    # Skip non-camera directories
                    if cam_dir_name in ("train", "test", "val", "validation"):
                        # Old format: videos/train/{camera}/episode_XXX.mp4
                        for sub_cam_dir in sorted(cam_dir.iterdir(), key=lambda p: p.name):
                            if sub_cam_dir.is_dir():
                                cam_name = sub_cam_dir.name.split(".")[-1]
                                video_file = sub_cam_dir / f"{episode_id}.mp4"
                                if video_file.exists():
                                    video_paths[cam_name] = video_file
                                    # Get dimensions from features
                                    full_key = f"observation.images.{cam_name}"
                                    if full_key in camera_features:
                                        shape = camera_features[full_key].get("shape", [480, 640, 3])
                                        video_dims[cam_name] = (shape[0], shape[1], shape[2] if len(shape) > 2 else 3)
                        continue

                    # Skip chunk directories at this level (handled above)
                    if cam_dir_name.startswith("chunk-"):
                        continue

                    # Camera-first format: videos/observation.images.{cam}/chunk-XXX/file-XXX.mp4
                    cam_name = cam_dir_name.split(".")[-1]

                    # Find ALL video files in the chunk directory for this camera
                    # (handles split videos like file-000.mp4, file-001.mp4)
                    parquet_rel = parquet_path.relative_to(dataset_path / "data")
                    chunk_dir = cam_dir / parquet_rel.parent

                    if chunk_dir.exists():
                        video_files = sorted(chunk_dir.glob("file-*.mp4"))
                        if video_files:
                            # For simple case, just use the first file
                            # We'll handle split files in load_frames
                            video_paths[cam_name] = video_files[0]
                            # Store all video files for this camera
                            video_paths[f"_all_{cam_name}"] = video_files
                            # Get dimensions from features
                            if cam_dir_name in camera_features:
                                shape = camera_features[cam_dir_name].get("shape", [480, 640, 3])
                                video_dims[cam_name] = (shape[0], shape[1], shape[2] if len(shape) > 2 else 3)

        # Determine if this is a multi-episode video format
        # Multi-episode: video_path template has {file_index} (e.g., file-000.mp4)
        # Single-episode: video_path template has episode_{episode_index} (e.g., episode_000000.mp4)
        is_multi_episode_video = False
        if video_path_template and "{file_index" in video_path_template:
            is_multi_episode_video = True

        # Build video file frame ranges for split video handling
        # Maps camera_name -> list of (start_frame, end_frame, video_path)
        video_frame_ranges: dict[str, list[tuple[int, int, Path]]] = {}
        for cam_name in list(video_paths.keys()):
            if cam_name.startswith("_all_"):
                continue
            all_key = f"_all_{cam_name}"
            if all_key in video_paths:
                video_files = video_paths[all_key]
                ranges = []
                cumulative_frames = 0
                for vf in video_files:
                    try:
                        import av
                        with av.open(str(vf)) as container:
                            stream = container.streams.video[0]
                            frame_count = stream.frames or 0
                            if frame_count == 0:
                                # Estimate from duration
                                fps = float(stream.average_rate) if stream.average_rate else 30.0
                                if stream.duration:
                                    frame_count = int(stream.duration * stream.time_base * fps)
                        ranges.append((cumulative_frames, cumulative_frames + frame_count - 1, vf))
                        cumulative_frames += frame_count
                    except Exception:
                        ranges.append((cumulative_frames, cumulative_frames + 100000, vf))
                        cumulative_frames += 100000
                video_frame_ranges[cam_name] = ranges

        def find_video_for_frame(cam_name: str, global_idx: int) -> tuple[Path, int]:
            """Find the video file and local frame index for a global frame index."""
            if cam_name in video_frame_ranges:
                for start, end, vpath in video_frame_ranges[cam_name]:
                    if start <= global_idx <= end:
                        return vpath, global_idx - start
            # Fallback to single video file
            return video_paths.get(cam_name, video_paths.get(list(video_paths.keys())[0])), global_idx

        def load_frames() -> Iterator[Frame]:
            for idx in range(len(ep_df)):
                row = ep_df.iloc[idx]
                images: dict[str, LazyImage] = {}

                # Get the frame index within the video
                if is_multi_episode_video and "index" in ep_df.columns:
                    # Multi-episode video: use global index for seeking
                    global_frame_idx = int(row["index"])
                elif "frame_index" in ep_df.columns:
                    # Single-episode video: use relative frame_index
                    global_frame_idx = int(row["frame_index"])
                elif "index" in ep_df.columns:
                    # Fallback: use index relative to first frame in episode
                    first_idx = int(ep_df.iloc[0]["index"])
                    global_frame_idx = int(row["index"]) - first_idx
                else:
                    global_frame_idx = idx

                for cam_name in video_paths.keys():
                    # Skip internal entries
                    if cam_name.startswith("_all_"):
                        continue

                    dims = video_dims.get(cam_name, (480, 640, 3))

                    # Find the right video file for this frame
                    if is_multi_episode_video and video_frame_ranges.get(cam_name):
                        video_path, local_frame_idx = find_video_for_frame(cam_name, global_frame_idx)
                    else:
                        video_path = video_paths[cam_name]
                        local_frame_idx = global_frame_idx

                    def make_loader(vp: Path = video_path, fi: int = local_frame_idx, d: tuple = dims) -> NDArray[Any]:
                        return self._extract_frame(vp, fi, d)

                    images[cam_name] = LazyImage(
                        loader=make_loader,
                        height=dims[0],
                        width=dims[1],
                        channels=dims[2],
                    )

                # Extract state and action
                state = None
                action = None

                for col in ep_df.columns:
                    col_lower = col.lower()
                    if "state" in col_lower or "observation" in col_lower:
                        val = row[col]
                        if hasattr(val, "__len__") and "image" not in col_lower:
                            import numpy as np

                            state = np.array(val, dtype=np.float32)
                            break

                for col in ep_df.columns:
                    if "action" in col.lower():
                        val = row[col]
                        if hasattr(val, "__len__"):
                            import numpy as np

                            action = np.array(val, dtype=np.float32)
                            break

                timestamp = row.get("timestamp", None)

                yield Frame(
                    index=idx,
                    timestamp=float(timestamp) if timestamp is not None else None,
                    images=images,
                    state=state,
                    action=action,
                    is_first=(idx == 0),
                    is_last=(idx == len(ep_df) - 1),
                )

        return Episode(
            episode_id=episode_id,
            language_instruction=language,
            _frame_loader=load_frames,
        )

    def _extract_frame(
        self,
        video_path: Path,
        frame_index: int,
        dims: tuple[int, int, int] = (480, 640, 3),
    ) -> NDArray[Any]:
        """Extract a single frame from video using cached sequential access."""
        # Use global video cache for efficient sequential playback
        cache = get_video_cache(video_path)
        return cache.get_frame(frame_index, dims)

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

        data_dir = path / "data" / "train"
        if not data_dir.exists():
            data_dir = path / "data"

        for pq_file in data_dir.rglob("*.parquet"):
            if pq_file.stem == episode_id:
                tasks = self._load_all_tasks(path)
                episode_meta = self._load_episode_metadata(path)
                episode_idx = self._parse_episode_index(episode_id)

                language = None
                if episode_meta and episode_idx in episode_meta:
                    meta = episode_meta[episode_idx]
                    task_idx = meta.get("task_index", 0)
                    if tasks and task_idx < len(tasks):
                        language = tasks[task_idx]

                return self._load_episode(path, pq_file, episode_id, language)

        raise EpisodeNotFoundError(episode_id, path)
