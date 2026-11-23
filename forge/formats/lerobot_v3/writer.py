"""LeRobot v3 format writer for Forge.

Writes datasets in LeRobot v3 format:
    dataset/
    ├── meta/
    │   ├── info.json           # Dataset metadata, features, paths
    │   ├── stats.json          # Aggregate stats (incl. quantiles, image stats)
    │   ├── tasks.parquet       # Task strings, index named "task"
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet  # One row per episode
    ├── data/
    │   └── chunk-{episode:03d}/
    │       └── file-000.parquet
    └── videos/
        └── observation.images.{camera}/
            └── chunk-{episode:03d}/
                └── file-000.mp4

The feature schema, tasks, episode rows and stats follow what
``lerobot.datasets.LeRobotDataset.create`` writes (checked against lerobot
0.5.1 and 0.7.0 in tests/test_lerobot_v3_spec_parity.py). One difference
remains: each episode gets its own data/video chunk, where LeRobot packs
episodes into files up to ``data_files_size_in_mb``. Both layouts load.

Every episode needs a task string. Episodes without a language instruction
take ``LeRobotV3WriterConfig.task``; if that is unset the writer raises
rather than invent one.

Each ``write_episode`` call stages a small JSON record under
``meta/.forge_episodes/``. ``finalize`` builds all metadata from those
records, so episodes written by separate processes (``forge convert
--workers N``) finalize the same way as a sequential run.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.exceptions import ConversionError, MissingDependencyError
from forge.core.models import CameraInfo, DatasetInfo, Episode, LazyImage
from forge.formats.lerobot_v3 import stats as v3stats
from forge.formats.registry import FormatRegistry
from forge.video.encoder import VideoEncoder, VideoEncoderConfig

logger = logging.getLogger(__name__)

STAGING_DIR = ".forge_episodes"
# SVT-AV1 3.0, bundled with PyAV 15 (the version lerobot[dataset] pins with
# av<16), never returns on frames narrower than ~28 px: 16, 20 and 24 hang,
# 28 and 31 encode. SVT-AV1 4.x (PyAV 17-19) is fine. Smaller cameras use h264.
_SVTAV1_MIN_WIDTH = 32
_SVTAV1_MIN_HEIGHT = 16
# Episode rows per meta/episodes file; LeRobot's default chunk size.
_EPISODES_PER_META_FILE = 1000

# PyAV encoder name -> codec name LeRobot records in info.json.
_CODEC_NAMES = {
    "libx264": "h264",
    "h264": "h264",
    "libx265": "hevc",
    "hevc": "hevc",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "av1": "av1",
    "libvpx-vp9": "vp9",
}


def _codec_available(codec: str) -> bool:
    try:
        import av
    except ImportError:
        return True  # VideoEncoder raises the missing-dependency error itself
    return codec in av.codecs_available


def _check_pyarrow() -> None:
    """Check if PyArrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="LeRobot v3 format writing",
            install_hint="pip install forge-robotics[lerobot]",
        )


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Same conversion VideoEncoder applies before encoding."""
    if img.dtype == np.uint8:
        return img
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def _scalar_feature(dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


# Per-frame index columns, in the order LeRobot declares them.
_INDEX_FEATURES: dict[str, str] = {
    "timestamp": "float32",
    "frame_index": "int64",
    "episode_index": "int64",
    "index": "int64",
    "task_index": "int64",
}


@dataclass
class LeRobotV3WriterConfig:
    """Configuration for LeRobot v3 writer.

    Attributes:
        fps: Frames per second (required).
        robot_type: Robot type identifier (e.g., "franka", "so100").
        video_codec: PyAV encoder (default "libsvtav1", LeRobot's default).
            Falls back to "libx264" with a warning when the installed PyAV
            lacks it; info.json records the codec actually used.
        video_crf: Constant rate factor for video quality (default: 30).
        video_preset: Encoder preset, codec-specific (default "12" for
            SVT-AV1; use e.g. "medium" with libx264). None omits it.
        video_gop_size: Keyframe interval in frames (default 2, as LeRobot).
        chunks_size: Max files per chunk directory, recorded in info.json.
        repo_id: Optional HuggingFace repository ID.
        camera_name_mapping: Optional mapping from source to target camera names.
        tokenized_action_feature: If set, frames carrying this key in
            ``frame.extras`` are written as an integer column of that name
            (dtype int64) and declared in info.json. Token ids are excluded
            from stats. Default None = current behavior (no token column).
        task: Task string for episodes that have no language instruction.
            None (default) makes such episodes an error.
        state_names: Names for the state dimensions, overriding any the
            source episodes carry.
        action_names: Names for the action dimensions, overriding any the
            source episodes carry.
    """

    fps: float = 30.0
    robot_type: str = "unknown"
    # LeRobot's own defaults (lerobot.configs.video.RGBEncoderConfig). Datasets
    # merge with aggregate_datasets only if their codecs match.
    video_codec: str = "libsvtav1"
    video_crf: int = 30
    video_preset: str | None = "12"
    video_gop_size: int = 2
    chunks_size: int = 1000
    # Upstream v3 target sizes for chunk-file packing (info.json requires both).
    data_files_size_in_mb: int = 100
    video_files_size_in_mb: int = 200
    repo_id: str | None = None
    camera_name_mapping: dict[str, str] = field(default_factory=dict)
    tokenized_action_feature: str | None = None
    task: str | None = None
    state_names: list[str] | None = None
    action_names: list[str] | None = None


@FormatRegistry.register_writer("lerobot-v3")
class LeRobotV3Writer:
    """Writer for LeRobot v3 format.

    Example:
        >>> writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30, task="pick the cube"))
        >>> writer.write_dataset(episodes, Path("./output"))
    """

    def __init__(self, config: LeRobotV3WriterConfig | None = None):
        self.config = config or LeRobotV3WriterConfig()
        self._encoders: dict[tuple[Any, ...], VideoEncoder] = {}
        self._codec_warned: set[tuple[str, int, int]] = set()
        self._reset()

    def _video_encoder(self, width: int, height: int) -> VideoEncoder:
        """Encoder for one camera under the current config (config fields
        may be set after construction, e.g. by Converter._configure_writer)."""
        cfg = self.config
        codec, preset = cfg.video_codec, cfg.video_preset
        fallback = None
        if not _codec_available(codec):
            fallback = f"PyAV has no {codec!r} encoder"
        elif codec == "libsvtav1" and (width < _SVTAV1_MIN_WIDTH or height < _SVTAV1_MIN_HEIGHT):
            fallback = f"SVT-AV1 hangs on frames this small ({width}x{height})"
        if fallback:
            if (codec, width, height) not in self._codec_warned:
                self._codec_warned.add((codec, width, height))
                logger.warning(
                    "%s; encoding this camera as h264. Datasets whose codecs differ "
                    "cannot be merged by lerobot's aggregate_datasets.",
                    fallback,
                )
            codec, preset = "libx264", "medium"
        key = (codec, cfg.video_crf, preset, cfg.video_gop_size)
        if key not in self._encoders:
            self._encoders[key] = VideoEncoder(
                VideoEncoderConfig(codec=codec, crf=cfg.video_crf, preset=preset, gop_size=cfg.video_gop_size)
            )
        return self._encoders[key]

    def _reset(self) -> None:
        self._num_episodes_written = 0
        self._tasks_seen: dict[str, int] = {}
        self._cameras: dict[str, CameraInfo] = {}
        self._total_frames = 0
        # Pending data for the episode being written; flushed by _flush_chunk.
        self._current_chunk_frames: dict[str, Any] | None = None
        self._current_chunk_videos: dict[str, list[LazyImage]] = {}
        self._current_chunk_index = 0

    @property
    def format_name(self) -> str:
        return "lerobot-v3"

    def _map_camera_name(self, source_name: str) -> str:
        """Map source camera name to LeRobot v3 naming convention.

        LeRobot v3 uses 'observation.images.{camera}' naming.
        """
        if source_name in self.config.camera_name_mapping:
            return self.config.camera_name_mapping[source_name]

        clean_name = source_name
        if clean_name.endswith("_image"):
            clean_name = clean_name[:-6]
        if clean_name.endswith("_rgb"):
            clean_name = clean_name[:-4]

        return f"observation.images.{clean_name}"

    def _resolve_names(self, override: list[str] | None, from_episode: list[str] | None, dim: int, what: str) -> list[str] | None:
        names = override if override is not None else from_episode
        if names is None:
            return None
        names = [str(n) for n in names]
        if len(names) != dim:
            raise ConversionError(
                "source",
                "lerobot-v3",
                f"{what} has {dim} dimensions but {len(names)} names were given: {names}",
            )
        return names

    def _video_feature(self, height: int, width: int, channels: int, fps: float) -> dict[str, Any]:
        enc = self._video_encoder(width, height).config
        return {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": _CODEC_NAMES.get(enc.codec, enc.codec),
                "video.pix_fmt": enc.pixel_format,
                "video.is_depth_map": False,
                "video.fps": int(fps) if float(fps).is_integer() else float(fps),
                "video.channels": channels,
                "has_audio": False,
            },
        }

    def _staging_dir(self, output_path: Path) -> Path:
        return output_path / "meta" / STAGING_DIR

    def _flush_chunk(self, output_path: Path) -> None:
        """Write the pending episode's data parquet and videos to disk."""
        if self._current_chunk_frames is None:
            return

        _check_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        chunk_index = self._current_chunk_index
        file_index = 0

        data_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            pq.write_table(
                pa.table(self._current_chunk_frames),
                data_dir / f"file-{file_index:03d}.parquet",
            )
        except Exception as e:
            raise ConversionError("source", "lerobot-v3", f"Failed to write parquet: {e}")

        fps = self.config.fps
        for cam_name, lazy_images in self._current_chunk_videos.items():
            if not lazy_images:
                continue
            video_path = output_path / "videos" / cam_name / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            first_img = lazy_images[0]
            try:
                self._video_encoder(first_img.width, first_img.height).encode_frames(
                    iter(lazy_images),
                    video_path,
                    fps=fps,
                    width=first_img.width,
                    height=first_img.height,
                )
            except Exception as e:
                raise ConversionError("source", "lerobot-v3", f"Failed to encode video: {e}")

        self._current_chunk_frames = None
        self._current_chunk_videos = {}

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Write a single episode: data parquet, videos, and a staging record.

        Raises:
            ConversionError: If the episode has no task, no frames, or
                writing fails.
        """
        _check_pyarrow()
        import pyarrow as pa

        output_path = Path(output_path)
        if episode_index is None:
            episode_index = self._num_episodes_written

        fps = episode.fps or self.config.fps

        task = episode.language_instruction or self.config.task
        if not task:
            raise ConversionError(
                "source",
                "lerobot-v3",
                f"Episode {episode.episode_id!r} has no language instruction, and LeRobot v3 "
                "stores a task string for every episode. Pass --task \"...\" (forge convert, "
                "forge tokenize write) or set writer_config.task for episodes without language.",
            )
        if task not in self._tasks_seen:
            self._tasks_seen[task] = len(self._tasks_seen)
        task_index = self._tasks_seen[task]

        try:
            frame_list = list(episode.frames())
        except Exception as e:
            raise ConversionError("source", "lerobot-v3", f"Failed to read frames: {e}")
        if not frame_list:
            raise ConversionError("source", "lerobot-v3", "Episode has no frames")

        n = len(frame_list)
        if progress_callback:
            for i in range(n):
                progress_callback(i, n)

        # --- numeric columns ---------------------------------------------
        timestamps = np.asarray(
            [f.timestamp if f.timestamp is not None else i / fps for i, f in enumerate(frame_list)],
            dtype=np.float32,
        )
        columns: dict[str, Any] = {}
        features: dict[str, dict[str, Any]] = {}
        numeric: dict[str, np.ndarray] = {}

        def add_vector(key: str, values: list[Any], dtype: Any, pa_type: Any, names: list[str] | None) -> None:
            arr = np.stack([np.asarray(v) for v in values]).astype(dtype)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            dim = arr.shape[1]
            flat = pa.array(arr.reshape(-1), type=pa_type)
            columns[key] = pa.FixedSizeListArray.from_arrays(flat, dim)
            features[key] = {"dtype": np.dtype(dtype).name, "shape": [dim], "names": names}
            numeric[key] = arr

        states = [f.state for f in frame_list]
        if all(s is not None for s in states):
            dim = int(np.asarray(states[0]).size)
            names = self._resolve_names(self.config.state_names, episode.state_names, dim, "observation.state")
            add_vector("observation.state", states, np.float32, pa.float32(), names)
        elif any(s is not None for s in states):
            raise ConversionError("source", "lerobot-v3", f"Episode {episode.episode_id!r}: state is missing on some frames")

        actions = [f.action for f in frame_list]
        if all(a is not None for a in actions):
            dim = int(np.asarray(actions[0]).size)
            names = self._resolve_names(self.config.action_names, episode.action_names, dim, "action")
            add_vector("action", actions, np.float32, pa.float32(), names)
        elif any(a is not None for a in actions):
            raise ConversionError("source", "lerobot-v3", f"Episode {episode.episode_id!r}: action is missing on some frames")

        # Action tokens (opt-in): int column from frame.extras, kept out of stats.
        tok_key = self.config.tokenized_action_feature
        tokenized = False
        if tok_key and all(tok_key in f.extras for f in frame_list):
            add_vector(tok_key, [f.extras[tok_key] for f in frame_list], np.int64, pa.int64(), None)
            numeric.pop(tok_key)
            tokenized = True

        index_values = {
            "timestamp": timestamps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "index": np.arange(self._total_frames, self._total_frames + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        }
        for key, values in index_values.items():
            columns[key] = pa.array(values)
            numeric[key] = values

        # --- cameras --------------------------------------------------------
        videos: dict[str, list[LazyImage]] = {}
        for frame in frame_list:
            for cam_name, lazy_img in frame.images.items():
                mapped = self._map_camera_name(cam_name)
                videos.setdefault(mapped, []).append(lazy_img)
                if cam_name not in self._cameras:
                    self._cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=lazy_img.height,
                        width=lazy_img.width,
                        channels=lazy_img.channels,
                    )
                if mapped not in features:
                    features[mapped] = self._video_feature(lazy_img.height, lazy_img.width, lazy_img.channels, fps)

        # --- per-episode stats ---------------------------------------------
        ep_stats: dict[str, v3stats.Stats] = {
            key: v3stats.vector_stats(values) for key, values in numeric.items()
        }
        for key, images in videos.items():
            if len(images) != n:
                raise ConversionError(
                    "source", "lerobot-v3", f"Episode {episode.episode_id!r}: camera {key!r} is missing on some frames"
                )
            sampled = []
            for i in v3stats.sample_indices(n):
                sampled.append(_to_uint8(np.asarray(images[i].load())))
                images[i].clear_cache()
            ep_stats[key] = v3stats.image_stats(sampled, n)

        # --- flush data + videos -------------------------------------------
        self._current_chunk_index = episode_index
        self._current_chunk_frames = columns
        self._current_chunk_videos = videos
        self._flush_chunk(output_path)

        record = {
            "episode_index": episode_index,
            "length": n,
            "task": task,
            "fps": float(fps),
            "features": features,
            "video_keys": list(videos),
            "tokenized": tokenized,
            "stats": v3stats.to_jsonable(ep_stats),
        }
        staging = self._staging_dir(output_path)
        staging.mkdir(parents=True, exist_ok=True)
        tmp = staging / f"episode-{episode_index:06d}.json.tmp"
        tmp.write_text(json.dumps(record))
        tmp.replace(staging / f"episode-{episode_index:06d}.json")

        self._total_frames += n
        self._num_episodes_written += 1

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        """Write full dataset from episode iterator."""
        output_path = Path(output_path)
        self.begin(output_path)

        if dataset_info:
            if dataset_info.inferred_fps:
                self.config.fps = dataset_info.inferred_fps
            if dataset_info.inferred_robot_type:
                self.config.robot_type = dataset_info.inferred_robot_type

        for episode_idx, episode in enumerate(episodes):
            if progress_callback:
                progress_callback(episode_idx, episode.episode_id)
            self.write_episode(episode, output_path, episode_index=episode_idx)

        if dataset_info is None:
            dataset_info = DatasetInfo(
                path=output_path,
                format="lerobot-v3",
                num_episodes=self._num_episodes_written,
                total_frames=self._total_frames,
                inferred_fps=self.config.fps,
                inferred_robot_type=self.config.robot_type,
                cameras=self._cameras,
            )
        self.finalize(output_path, dataset_info)

    def begin(self, output_path: Path) -> None:
        """Start a fresh dataset at ``output_path``: drop any staging records
        left by an earlier, unfinished run. Call once before the first
        ``write_episode`` (``write_dataset`` and ``Converter`` do)."""
        self._reset()
        shutil.rmtree(self._staging_dir(Path(output_path)), ignore_errors=True)

    def _load_records(self, output_path: Path) -> list[dict[str, Any]]:
        """Read staging records, renumbering episodes to close any gaps.

        A gap appears when an episode failed and the conversion carried on
        (``fail_on_error`` off). LeRobot needs contiguous episode indices, so
        later episodes move down: their chunk directories are renamed and
        their ``episode_index`` column rewritten.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        staging = self._staging_dir(output_path)
        records = [json.loads(p.read_text()) for p in staging.glob("episode-*.json")]
        records.sort(key=lambda r: r["episode_index"])
        for new_idx, rec in enumerate(records):
            old_idx = rec["episode_index"]
            if old_idx == new_idx:
                continue
            # Ascending order means chunk-{new_idx} is already free.
            dirs = [output_path / "data"] + [output_path / "videos" / vk for vk in rec["video_keys"]]
            for base in dirs:
                src = base / f"chunk-{old_idx:03d}"
                if src.exists():
                    src.rename(base / f"chunk-{new_idx:03d}")
            data_file = output_path / "data" / f"chunk-{new_idx:03d}" / "file-000.parquet"
            table = pq.read_table(data_file)
            col = table.schema.get_field_index("episode_index")
            table = table.set_column(
                col, "episode_index", pa.array(np.full(table.num_rows, new_idx, dtype=np.int64))
            )
            pq.write_table(table, data_file)
            rec["episode_index"] = new_idx
            rec["stats"]["episode_index"] = v3stats.to_jsonable(
                {"x": v3stats.vector_stats(np.full(rec["length"], new_idx, dtype=np.int64))}
            )["x"]
        return records

    @staticmethod
    def _merge_features(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for rec in records:
            for key, feat in rec["features"].items():
                if key not in merged:
                    merged[key] = feat
                    continue
                prev = merged[key]
                if prev["shape"] != feat["shape"] or prev["dtype"] != feat["dtype"]:
                    raise ConversionError(
                        "source",
                        "lerobot-v3",
                        f"Feature {key!r} is {prev['dtype']}{prev['shape']} in one episode and "
                        f"{feat['dtype']}{feat['shape']} in episode {rec['episode_index']}; "
                        "LeRobot v3 needs one shape per feature.",
                    )
                if prev.get("names") != feat.get("names"):
                    raise ConversionError(
                        "source",
                        "lerobot-v3",
                        f"Feature {key!r} has names {prev.get('names')} in one episode and "
                        f"{feat.get('names')} in episode {rec['episode_index']}.",
                    )
        keys = {k for rec in records for k in rec["features"]}
        for key in keys:
            if any(key not in rec["features"] for rec in records):
                raise ConversionError("source", "lerobot-v3", f"Feature {key!r} is missing from some episodes")
        return merged

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write meta/info.json, stats.json, tasks.parquet and episodes parquet.

        Builds everything from the staging records that ``write_episode``
        left in ``meta/.forge_episodes/``, then removes them.
        """
        self._flush_chunk(output_path)
        _check_pyarrow()
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path = Path(output_path)
        meta_dir = output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        records = self._load_records(output_path)
        if not records:
            raise ConversionError("source", "lerobot-v3", "No episodes were written")

        fps = float(self.config.fps or dataset_info.inferred_fps or 30)
        # Keep fractional rates (29.97, 12.5 Hz); write whole rates as ints like LeRobot does.
        fps_out: int | float = int(fps) if fps.is_integer() else fps

        # Tasks get indices in order of first appearance across episodes, as
        # LeRobot assigns them. Episodes written by separate worker processes
        # numbered their tasks locally, so fix up any data file that differs.
        task_to_index: dict[str, int] = {}
        for rec in records:
            task_to_index.setdefault(rec["task"], len(task_to_index))

        # Global frame offsets, likewise only known once every episode is in.
        offset = 0
        for rec in records:
            rec["dataset_from_index"] = offset
            offset += rec["length"]
            rec["dataset_to_index"] = offset
        total_frames = offset

        for rec in records:
            ep = rec["episode_index"]
            data_file = output_path / "data" / f"chunk-{ep:03d}" / "file-000.parquet"
            table = pq.read_table(data_file)
            want_task = task_to_index[rec["task"]]
            want_index0 = rec["dataset_from_index"]
            if (
                table["task_index"][0].as_py() != want_task
                or table["index"][0].as_py() != want_index0
            ):
                n = table.num_rows
                table = table.set_column(
                    table.schema.get_field_index("task_index"),
                    "task_index",
                    pa.array(np.full(n, want_task, dtype=np.int64)),
                )
                table = table.set_column(
                    table.schema.get_field_index("index"),
                    "index",
                    pa.array(np.arange(want_index0, want_index0 + n, dtype=np.int64)),
                )
                pq.write_table(table, data_file)
                stats = rec["stats"]
                stats["task_index"] = v3stats.to_jsonable(
                    {"x": v3stats.vector_stats(np.full(n, want_task, dtype=np.int64))}
                )["x"]
                stats["index"] = v3stats.to_jsonable(
                    {"x": v3stats.vector_stats(np.arange(want_index0, want_index0 + n, dtype=np.int64))}
                )["x"]

        merged = self._merge_features(records)
        features: dict[str, dict[str, Any]] = {}
        for key in ("observation.state", "action"):
            if key in merged:
                features[key] = merged[key]
        for key, feat in merged.items():
            if feat["dtype"] == "video":
                features[key] = feat
        for key, feat in merged.items():
            features.setdefault(key, feat)
        for key, dtype in _INDEX_FEATURES.items():
            features[key] = _scalar_feature(dtype)

        video_keys = [k for k, f in features.items() if f["dtype"] == "video"]

        info: dict[str, Any] = {
            "codebase_version": "v3.0",
            "robot_type": self.config.robot_type or dataset_info.inferred_robot_type or "unknown",
            "total_episodes": len(records),
            "total_frames": total_frames,
            "total_tasks": len(task_to_index),
            "chunks_size": int(self.config.chunks_size),
            "data_files_size_in_mb": int(self.config.data_files_size_in_mb),
            "video_files_size_in_mb": int(self.config.video_files_size_in_mb),
            "fps": fps_out,
            "splits": {"train": f"0:{len(records)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        if self.config.repo_id:
            info["repo_id"] = self.config.repo_id
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        # Episode rows, one per episode, packed into meta/episodes files.
        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            ep = rec["episode_index"]
            meta_file_seq = i // _EPISODES_PER_META_FILE
            meta_chunk = meta_file_seq // self.config.chunks_size
            meta_file = meta_file_seq % self.config.chunks_size
            row: dict[str, Any] = {
                "episode_index": ep,
                "tasks": [rec["task"]],
                "length": rec["length"],
                "data/chunk_index": ep,
                "data/file_index": 0,
                "dataset_from_index": rec["dataset_from_index"],
                "dataset_to_index": rec["dataset_to_index"],
            }
            for vk in video_keys:
                row[f"videos/{vk}/chunk_index"] = ep
                row[f"videos/{vk}/file_index"] = 0
                row[f"videos/{vk}/from_timestamp"] = 0.0
                row[f"videos/{vk}/to_timestamp"] = float(rec["length"]) / fps
            for feat_name in features:
                for stat_name, value in rec["stats"].get(feat_name, {}).items():
                    row[f"stats/{feat_name}/{stat_name}"] = value
            row["meta/episodes/chunk_index"] = meta_chunk
            row["meta/episodes/file_index"] = meta_file
            rows.append(row)

        shutil.rmtree(meta_dir / "episodes", ignore_errors=True)
        for start in range(0, len(rows), _EPISODES_PER_META_FILE):
            batch = rows[start : start + _EPISODES_PER_META_FILE]
            chunk, file = batch[0]["meta/episodes/chunk_index"], batch[0]["meta/episodes/file_index"]
            path = meta_dir / "episodes" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(batch), path)

        aggregate = v3stats.aggregate([v3stats.from_jsonable(r["stats"]) for r in records])
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(v3stats.to_jsonable(aggregate), f, indent=4)

        # Upstream io_utils.write_tasks: index named "task", column task_index.
        tasks_df = pd.DataFrame(
            {"task_index": list(task_to_index.values())},
            index=pd.Index(list(task_to_index.keys()), name="task"),
        )
        tasks_df.to_parquet(meta_dir / "tasks.parquet")

        shutil.rmtree(self._staging_dir(output_path), ignore_errors=True)
