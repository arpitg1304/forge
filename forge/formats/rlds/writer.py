"""RLDS format writer for Forge.

Writes datasets in RLDS (TensorFlow Datasets) format, commonly used for
Open X-Embodiment robotics datasets.

Output structure:
    dataset/
    ├── dataset_info.json       # Dataset metadata
    ├── features.json           # TFDS feature specification
    └── {name}-train.tfrecord-XXXXX-of-YYYYY  # Sharded TFRecord files
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Suppress TensorFlow logging BEFORE any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

from forge.core.exceptions import ConversionError, MissingDependencyError
from forge.core.models import CameraInfo, DatasetInfo, Episode
from forge.formats.registry import FormatRegistry


def _check_tensorflow() -> None:
    """Check if TensorFlow is available."""
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
    except ImportError:
        raise MissingDependencyError(
            dependency="tensorflow",
            feature="RLDS format writing",
            install_hint="pip install forge-robotics[rlds]",
        )


@dataclass
class RLDSWriterConfig:
    """Configuration for RLDS writer.

    Attributes:
        dataset_name: Name of the dataset (used in filenames).
        fps: Frames per second.
        robot_type: Robot type identifier.
        image_encoding: Image encoding format ("jpeg" or "png").
        image_quality: JPEG quality (1-100).
        episodes_per_shard: Number of episodes per TFRecord shard.
        description: Dataset description.
        citation: BibTeX citation for the dataset.
        camera_name_mapping: Optional mapping from source to target camera names.
    """

    dataset_name: str = "forge_dataset"
    fps: float = 30.0
    robot_type: str = "unknown"
    image_encoding: str = "jpeg"
    image_quality: int = 95
    episodes_per_shard: int = 10
    description: str = "Dataset converted with Forge"
    citation: str = ""
    camera_name_mapping: dict[str, str] = field(default_factory=dict)


@FormatRegistry.register_writer("rlds")
class RLDSWriter:
    """Writer for RLDS format.

    Converts Episode/Frame data to RLDS format with:
    - TFRecord files containing serialized episodes
    - dataset_info.json with metadata
    - features.json with TFDS feature specification

    Example:
        >>> writer = RLDSWriter(RLDSWriterConfig(dataset_name="my_robot_data"))
        >>> writer.write_dataset(episodes, Path("./output"))
    """

    def __init__(self, config: RLDSWriterConfig | None = None):
        """Initialize writer with configuration.

        Args:
            config: Writer configuration. Uses defaults if None.
        """
        self.config = config or RLDSWriterConfig()

        # Track written episodes for metadata
        self._episode_metadata: list[dict[str, Any]] = []
        self._total_frames: int = 0
        self._total_episodes: int = 0
        self._shard_lengths: list[int] = []
        self._current_shard_episodes: int = 0
        self._current_shard_index: int = 0
        self._current_writer: Any = None

        # Schema inference
        self._action_shape: tuple[int, ...] | None = None
        self._state_shape: tuple[int, ...] | None = None
        self._cameras: dict[str, CameraInfo] = {}
        self._has_language: bool = False
        self._has_reward: bool = False

    @property
    def format_name(self) -> str:
        return "rlds"

    def _map_camera_name(self, source_name: str) -> str:
        """Map source camera name to RLDS naming convention.

        Args:
            source_name: Original camera name.

        Returns:
            RLDS camera name.
        """
        if source_name in self.config.camera_name_mapping:
            return self.config.camera_name_mapping[source_name]

        # Clean up LeRobot-style names and ensure _image suffix for RLDS compatibility
        clean_name = source_name
        if clean_name.startswith("observation.images."):
            clean_name = clean_name.replace("observation.images.", "")

        # Ensure camera names have _image suffix for RLDS reader compatibility
        if not any(clean_name.endswith(suffix) for suffix in ["_image", "_rgb", "_img", "_frame"]):
            clean_name = f"{clean_name}_image"

        return clean_name

    def _encode_image(self, image_array: Any) -> bytes:
        """Encode image array to bytes.

        Args:
            image_array: NumPy array of image data.

        Returns:
            Encoded image bytes.
        """
        import numpy as np
        from PIL import Image

        # Ensure uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        img = Image.fromarray(image_array)
        buffer = io.BytesIO()

        if self.config.image_encoding == "jpeg":
            img.save(buffer, format="JPEG", quality=self.config.image_quality)
        else:
            img.save(buffer, format="PNG")

        return buffer.getvalue()

    def _get_tfrecord_path(self, output_path: Path, shard_index: int, total_shards: int) -> Path:
        """Get path for a TFRecord shard file.

        Args:
            output_path: Base output directory.
            shard_index: Index of current shard.
            total_shards: Total number of shards.

        Returns:
            Path to TFRecord file.
        """
        name = self.config.dataset_name
        return output_path / f"{name}-train.tfrecord-{shard_index:05d}-of-{total_shards:05d}"

    def _create_tf_example(self, episode: Episode) -> Any:
        """Create a TensorFlow Example from an Episode.

        RLDS format stores each episode as a single tf.train.Example with
        nested steps structure.

        Args:
            episode: Episode to convert.

        Returns:
            tf.train.Example proto.
        """
        import numpy as np
        import tensorflow as tf

        frames = list(episode.frames())
        if not frames:
            raise ConversionError("source", "rlds", "Episode has no frames")

        # Build steps data
        steps_data: dict[str, list[Any]] = {
            "action": [],
            "reward": [],
            "discount": [],
            "is_first": [],
            "is_last": [],
            "is_terminal": [],
        }

        # Observation fields
        obs_data: dict[str, list[Any]] = {}

        for frame in frames:
            # Action
            if frame.action is not None:
                steps_data["action"].append(frame.action.astype(np.float32))
                if self._action_shape is None:
                    self._action_shape = frame.action.shape
            else:
                # Pad with zeros if no action
                if self._action_shape:
                    steps_data["action"].append(np.zeros(self._action_shape, dtype=np.float32))

            # Reward
            reward = frame.reward if frame.reward is not None else 0.0
            steps_data["reward"].append(float(reward))
            if frame.reward is not None:
                self._has_reward = True

            # Discount (default 1.0)
            steps_data["discount"].append(1.0)

            # Step flags
            steps_data["is_first"].append(frame.is_first)
            steps_data["is_last"].append(frame.is_last)
            steps_data["is_terminal"].append(frame.is_terminal if frame.is_terminal else False)

            # State/proprio
            if frame.state is not None:
                if "state" not in obs_data:
                    obs_data["state"] = []
                obs_data["state"].append(frame.state.astype(np.float32))
                if self._state_shape is None:
                    self._state_shape = frame.state.shape

            # Images
            for cam_name, lazy_img in frame.images.items():
                mapped_name = self._map_camera_name(cam_name)
                if mapped_name not in obs_data:
                    obs_data[mapped_name] = []
                    self._cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=lazy_img.height,
                        width=lazy_img.width,
                        channels=lazy_img.channels,
                    )

                # Encode image
                img_array = lazy_img.load()
                img_bytes = self._encode_image(img_array)
                obs_data[mapped_name].append(img_bytes)

        # Language instruction (repeated for each step in RLDS format)
        language = episode.language_instruction or ""
        if language:
            self._has_language = True
            obs_data["language_instruction"] = [language.encode("utf-8")] * len(frames)

        # Create feature dict for tf.train.Example
        # RLDS uses SequenceExample or nested structure
        # For simplicity, we'll use the flat serialization approach
        feature_dict: dict[str, tf.train.Feature] = {}

        # Serialize steps data
        for key, values in steps_data.items():
            if key in ["is_first", "is_last", "is_terminal"]:
                feature_dict[f"steps/{key}"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(v) for v in values])
                )
            elif key in ["reward", "discount"]:
                feature_dict[f"steps/{key}"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=values)
                )
            elif key == "action" and values:
                # Flatten action array
                flat_actions = np.concatenate([a.flatten() for a in values])
                feature_dict[f"steps/{key}"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=flat_actions.tolist())
                )

        # Serialize observation data
        for key, values in obs_data.items():
            if key == "state" and values:
                flat_state = np.concatenate([s.flatten() for s in values])
                feature_dict[f"steps/observation/{key}"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=flat_state.tolist())
                )
            elif key == "language_instruction":
                # Store as bytes list
                feature_dict[f"steps/{key}"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=values)
                )
            elif isinstance(values[0], bytes):
                # Image data
                feature_dict[f"steps/observation/{key}"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=values)
                )

        # Episode metadata
        feature_dict["episode_metadata/file_path"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[episode.episode_id.encode("utf-8")])
        )
        # Carry the source episode's metadata into RLDS episode_metadata. RLDS
        # supports per-episode metadata first-class; without this it is dropped on
        # conversion. Stored as a JSON string so arbitrary keys fit one fixed
        # feature ("{}" when there is none, keeping the schema uniform).
        feature_dict["episode_metadata/metadata_json"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[json.dumps(episode.metadata or {}, default=str).encode("utf-8")]
            )
        )

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def _open_shard_writer(self, output_path: Path, shard_index: int, total_shards: int) -> Any:
        """Open a TFRecord writer for a shard.

        Args:
            output_path: Base output directory.
            shard_index: Index of current shard.
            total_shards: Total number of shards.

        Returns:
            TFRecord writer.
        """
        import tensorflow as tf

        tfrecord_path = self._get_tfrecord_path(output_path, shard_index, total_shards)
        return tf.io.TFRecordWriter(str(tfrecord_path))

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Write a single episode to the output directory.

        Args:
            episode: Episode to write.
            output_path: Base output directory.
            episode_index: Optional explicit episode index.
            progress_callback: Optional callback for progress.

        Raises:
            ConversionError: If writing fails.
        """
        _check_tensorflow()

        if episode_index is None:
            episode_index = self._total_episodes

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Convert episode to TF Example
            tf_example = self._create_tf_example(episode)

            # Count frames
            frames = list(episode.frames())
            num_frames = len(frames)

            # Track metadata
            self._episode_metadata.append(
                {
                    "episode_index": episode_index,
                    "num_frames": num_frames,
                    "shard_index": self._current_shard_index,
                }
            )
            self._total_frames += num_frames
            self._total_episodes += 1
            self._current_shard_episodes += 1

            # Write to temp file per episode (we'll consolidate in finalize)
            temp_path = output_path / f"_temp_episode_{episode_index:05d}_shard_{self._current_shard_index:05d}.tfrecord"

            import tensorflow as tf

            # Write without compression (standard TFDS format)
            with tf.io.TFRecordWriter(str(temp_path)) as writer:
                writer.write(tf_example.SerializeToString())

            # Check if we need to start a new shard
            if self._current_shard_episodes >= self.config.episodes_per_shard:
                self._shard_lengths.append(self._current_shard_episodes)
                self._current_shard_index += 1
                self._current_shard_episodes = 0

        except Exception as e:
            raise ConversionError("source", "rlds", f"Failed to write episode: {e}")

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        """Write full dataset from episode iterator.

        Args:
            episodes: Iterator of episodes to write.
            output_path: Base output directory.
            dataset_info: Optional dataset metadata.
            progress_callback: Optional callback for progress.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Reset tracking state
        self._episode_metadata = []
        self._total_frames = 0
        self._total_episodes = 0
        self._shard_lengths = []
        self._current_shard_episodes = 0
        self._current_shard_index = 0
        self._action_shape = None
        self._state_shape = None
        self._cameras = {}
        self._has_language = False
        self._has_reward = False

        # Override config from dataset_info if provided
        if dataset_info:
            if dataset_info.inferred_fps:
                self.config.fps = dataset_info.inferred_fps
            if dataset_info.inferred_robot_type:
                self.config.robot_type = dataset_info.inferred_robot_type

        # Process each episode
        for episode_idx, episode in enumerate(episodes):
            if progress_callback:
                progress_callback(episode_idx, episode.episode_id)

            self.write_episode(episode, output_path, episode_index=episode_idx)

        # Finalize
        if dataset_info:
            self.finalize(output_path, dataset_info)
        else:
            minimal_info = DatasetInfo(
                path=output_path,
                format="rlds",
                num_episodes=self._total_episodes,
                total_frames=self._total_frames,
                inferred_fps=self.config.fps,
                inferred_robot_type=self.config.robot_type,
                cameras=self._cameras,
            )
            self.finalize(output_path, minimal_info)

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write metadata files and consolidate TFRecord shards.

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
        """
        _check_tensorflow()
        import tensorflow as tf

        output_path = Path(output_path)

        # Finalize last shard
        if self._current_shard_episodes > 0:
            self._shard_lengths.append(self._current_shard_episodes)

        total_shards = len(self._shard_lengths) if self._shard_lengths else 1

        # Consolidate temp files into properly named shards
        self._consolidate_shards(output_path, total_shards)

        # Write dataset_info.json
        self._write_dataset_info(output_path, dataset_info, total_shards)

        # Write features.json
        self._write_features_json(output_path, dataset_info)

    def _consolidate_shards(self, output_path: Path, total_shards: int) -> None:
        """Consolidate temporary shard files into final format.

        Args:
            output_path: Base output directory.
            total_shards: Total number of shards.
        """
        import tensorflow as tf

        # Group episodes by shard
        shard_episodes: dict[int, list[bytes]] = {}

        # Read all temp files (format: _temp_episode_XXXXX_shard_YYYYY.tfrecord)
        for temp_file in sorted(output_path.glob("_temp_episode_*_shard_*.tfrecord")):
            # Extract shard index from filename
            parts = temp_file.stem.split("_")
            shard_idx = int(parts[-1])  # Last part after "shard_"
            if shard_idx not in shard_episodes:
                shard_episodes[shard_idx] = []

            # Read records from temp file (no compression)
            dataset = tf.data.TFRecordDataset([str(temp_file)])
            for record in dataset:
                shard_episodes[shard_idx].append(record.numpy())

            # Remove temp file
            temp_file.unlink()

        # Write consolidated shard files
        for shard_idx in range(total_shards):
            records = shard_episodes.get(shard_idx, [])
            if not records:
                continue

            shard_path = self._get_tfrecord_path(output_path, shard_idx, total_shards)

            # Write without compression (standard TFDS format)
            with tf.io.TFRecordWriter(str(shard_path)) as writer:
                for record in records:
                    writer.write(record)

    def _write_dataset_info(
        self, output_path: Path, dataset_info: DatasetInfo, total_shards: int
    ) -> None:
        """Write dataset_info.json metadata file.

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
            total_shards: Total number of shards.
        """
        # Calculate total bytes (approximate)
        total_bytes = sum(
            f.stat().st_size
            for f in output_path.glob(f"{self.config.dataset_name}-train.tfrecord-*")
        )

        info = {
            "citation": self.config.citation or "Dataset converted with Forge",
            "description": self.config.description,
            "fileFormat": "tfrecord",
            "moduleName": f"{self.config.dataset_name}.{self.config.dataset_name}",
            "name": self.config.dataset_name,
            "releaseNotes": {"1.0.0": "Converted with Forge"},
            "splits": [
                {
                    "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                    "name": "train",
                    "numBytes": str(total_bytes),
                    "shardLengths": [str(length) for length in self._shard_lengths],
                }
            ],
            "version": "1.0.0",
        }

        with open(output_path / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_features_json(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write features.json TFDS feature specification.

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
        """
        # Build observation features
        obs_features: dict[str, Any] = {}

        # State feature
        if self._state_shape:
            obs_features["state"] = {
                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                "tensor": {
                    "shape": {"dimensions": [str(d) for d in self._state_shape]},
                    "dtype": "float32",
                    "encoding": "none",
                },
                "description": "Robot state/proprioception.",
            }

        # Camera features
        for cam_name, cam_info in self._cameras.items():
            mapped_name = self._map_camera_name(cam_name)
            obs_features[mapped_name] = {
                "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                "image": {
                    "shape": {
                        "dimensions": [
                            str(cam_info.height),
                            str(cam_info.width),
                            str(cam_info.channels),
                        ]
                    },
                    "dtype": "uint8",
                    "encodingFormat": self.config.image_encoding,
                },
                "description": f"RGB camera observation from {cam_name}.",
            }

        # Build step features
        step_features: dict[str, Any] = {
            "action": {
                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                "tensor": {
                    "shape": {
                        "dimensions": [str(d) for d in (self._action_shape or (7,))]
                    },
                    "dtype": "float32",
                    "encoding": "none",
                },
                "description": "Robot action.",
            },
            "is_terminal": {
                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                "description": "True on last step if terminal.",
            },
            "is_last": {
                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                "description": "True on last step of episode.",
            },
            "is_first": {
                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                "description": "True on first step of episode.",
            },
            "discount": {
                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                "description": "Discount factor.",
            },
            "reward": {
                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                "description": "Reward signal.",
            },
            "observation": {
                "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                "featuresDict": {"features": obs_features},
            },
        }

        # Add language instruction if present
        if self._has_language:
            step_features["language_instruction"] = {
                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                "text": {},
                "description": "Language instruction.",
            }

        # Build full features spec
        features = {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {
                "features": {
                    "steps": {
                        "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                        "sequence": {
                            "feature": {
                                "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                "featuresDict": {"features": step_features},
                            },
                            "length": "-1",
                        },
                    },
                    "episode_metadata": {
                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                        "featuresDict": {
                            "features": {
                                "file_path": {
                                    "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                    "text": {},
                                    "description": "Original episode identifier.",
                                },
                                "metadata_json": {
                                    "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                    "text": {},
                                    "description": "Source episode metadata, JSON-encoded ('{}' if none).",
                                },
                            }
                        },
                    },
                }
            },
        }

        with open(output_path / "features.json", "w") as f:
            json.dump(features, f, indent=4)
