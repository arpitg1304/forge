"""Core domain models for Forge.

This module defines the canonical intermediate representations used throughout
the Forge library for robotics episode data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Dtype(Enum):
    """Data types supported by Forge."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"
    STRING = "string"

    def to_numpy_dtype(self) -> str:
        """Convert to numpy dtype string."""
        if self == Dtype.STRING:
            return "object"
        return self.value


@dataclass(frozen=True)
class FieldSchema:
    """Schema for a single field in the dataset.

    Attributes:
        name: Field identifier.
        shape: Tuple describing the array shape (excluding batch dimension).
        dtype: Data type of the field.
        description: Optional human-readable description.
    """

    name: str
    shape: tuple[int, ...]
    dtype: Dtype
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Field name cannot be empty")


@dataclass(frozen=True)
class CameraInfo:
    """Metadata about a camera stream.

    Attributes:
        name: Camera identifier (e.g., "wrist_cam", "overhead").
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels (default 3 for RGB).
        encoding: Color encoding format ("rgb", "bgr", "depth", etc.).
        storage: How frames are physically stored — "mp4" (external video
            file), "image" (inline image bytes in the row, e.g. parquet
            with dtype=image), or "unknown" if the reader can't tell.
    """

    name: str
    height: int
    width: int
    channels: int = 3
    encoding: str = "rgb"
    storage: str = "unknown"

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Camera dimensions must be positive")
        if self.channels <= 0:
            raise ValueError("Channels must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels) tuple."""
        return (self.height, self.width, self.channels)


# ============================================================
# Lazy Loading Primitives
# ============================================================


@dataclass
class LazyArray:
    """Lazy-loaded array - only loads data when accessed.

    Enables streaming without OOM on large datasets. The loader function
    is only called when load() or __array__() is invoked.

    Attributes:
        loader: Callable that returns the numpy array when invoked.
        shape: Expected shape of the array.
        dtype: Expected data type.
    """

    loader: Callable[[], NDArray[Any]]
    shape: tuple[int, ...]
    dtype: Dtype
    _cached: NDArray[Any] | None = field(default=None, repr=False, compare=False)

    def load(self) -> NDArray[Any]:
        """Load and return the array, caching the result."""
        if self._cached is None:
            object.__setattr__(self, "_cached", self.loader())
        return self._cached  # type: ignore[return-value]

    def __array__(self) -> NDArray[Any]:
        """Support numpy array protocol."""
        return self.load()

    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded into cache."""
        return self._cached is not None

    def clear_cache(self) -> None:
        """Clear cached data to free memory."""
        object.__setattr__(self, "_cached", None)


@dataclass
class LazyImage:
    """Lazy-loaded image frame.

    Can load from: file path, byte buffer, or extraction function.
    Designed for memory-efficient streaming of video frames.

    Attributes:
        loader: Callable that returns the image as a numpy array.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels (default 3).
    """

    loader: Callable[[], NDArray[Any]]
    height: int
    width: int
    channels: int = 3
    _cached: NDArray[Any] | None = field(default=None, repr=False, compare=False)

    def load(self) -> NDArray[Any]:
        """Load and return the image, caching the result."""
        if self._cached is None:
            object.__setattr__(self, "_cached", self.loader())
        return self._cached  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels) tuple."""
        return (self.height, self.width, self.channels)

    @property
    def is_loaded(self) -> bool:
        """Check if image has been loaded into cache."""
        return self._cached is not None

    def clear_cache(self) -> None:
        """Clear cached data to free memory."""
        object.__setattr__(self, "_cached", None)


# ============================================================
# Frame - Single Timestep
# ============================================================


@dataclass
class Frame:
    """A single timestep in an episode.

    All fields are optional to handle varying dataset schemas. This allows
    Forge to work with datasets that have different observation structures.

    Attributes:
        index: Zero-based frame index within the episode.
        timestamp: Optional timestamp in seconds.
        images: Dictionary mapping camera names to lazy-loaded images.
        state: Full proprioception vector (raw).
        joint_positions: Parsed joint positions (if schema known).
        joint_velocities: Parsed joint velocities (if schema known).
        gripper_state: Parsed gripper state in [0, 1] (if schema known).
        action: Full action vector (raw).
        action_joints: Parsed joint actions (if schema known).
        action_gripper: Parsed gripper action (if schema known).
        reward: Optional reward signal.
        is_terminal: True if this is a terminal state.
        is_first: True if this is the first frame.
        is_last: True if this is the last frame.
    """

    index: int
    timestamp: float | None = None

    # Observations
    images: dict[str, LazyImage] = field(default_factory=dict)
    state: NDArray[Any] | None = None

    # Parsed proprioception (optional, if schema known)
    joint_positions: NDArray[Any] | None = None
    joint_velocities: NDArray[Any] | None = None
    gripper_state: float | None = None

    # Action
    action: NDArray[Any] | None = None

    # Parsed action (optional)
    action_joints: NDArray[Any] | None = None
    action_gripper: float | None = None

    # Extras
    reward: float | None = None
    is_terminal: bool = False
    is_first: bool = False
    is_last: bool = False

    # Format-specific extra fields keyed by dotted name (e.g. "observation.ee_pose").
    # Used by readers that surface fields outside the canonical state/action/images slots.
    extras: dict[str, Any] = field(default_factory=dict)

    def get_image(self, camera_name: str) -> NDArray[Any]:
        """Load and return image from specified camera.

        Args:
            camera_name: Name of the camera.

        Returns:
            Loaded image as numpy array.

        Raises:
            KeyError: If camera not found.
        """
        if camera_name not in self.images:
            raise KeyError(
                f"Camera '{camera_name}' not found. Available: {list(self.images.keys())}"
            )
        return self.images[camera_name].load()


# ============================================================
# Episode - Sequence of Frames
# ============================================================


@dataclass
class Episode:
    """Canonical intermediate representation of a robotics episode.

    Design choices:
    - Lazy frame iteration for memory efficiency
    - Metadata separate from frame data
    - Flexible schema (not all fields required)

    Attributes:
        episode_id: Unique identifier for this episode.
        metadata: Arbitrary metadata dictionary.
        language_instruction: Optional natural language task description.
        success: Optional success label.
        robot_type: Optional robot identifier (e.g., "franka", "widow_x").
        cameras: Dictionary mapping camera names to CameraInfo.
        state_dim: Dimensionality of state vector.
        action_dim: Dimensionality of action vector.
        fps: Frames per second.
    """

    episode_id: str

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    language_instruction: str | None = None
    success: bool | None = None
    robot_type: str | None = None

    # Schema info (inferred or provided)
    cameras: dict[str, CameraInfo] = field(default_factory=dict)
    state_dim: int | None = None
    action_dim: int | None = None
    fps: float | None = None

    # Frame access - lazy iterator
    _frame_loader: Callable[[], Iterator[Frame]] | None = field(default=None, repr=False)
    _frames_cache: list[Frame] | None = field(default=None, repr=False)

    def frames(self) -> Iterator[Frame]:
        """Iterate over frames. Loads lazily if possible.

        Yields:
            Frame objects in sequence.

        Raises:
            ValueError: If no frame data is available.
        """
        if self._frames_cache is not None:
            yield from self._frames_cache
        elif self._frame_loader is not None:
            yield from self._frame_loader()
        else:
            raise ValueError("Episode has no frame data")

    def load_frames(self) -> list[Frame]:
        """Force-load all frames into memory.

        Returns:
            List of all frames in the episode.
        """
        if self._frames_cache is None:
            self._frames_cache = list(self.frames())
        return self._frames_cache

    @property
    def num_frames(self) -> int | None:
        """Number of frames if known without loading.

        Returns:
            Frame count if cached or available in metadata, else None.
        """
        if self._frames_cache is not None:
            return len(self._frames_cache)
        return self.metadata.get("num_frames")

    def clear_cache(self) -> None:
        """Clear cached frames to free memory."""
        self._frames_cache = None


# ============================================================
# Dataset Info - Inspection Result
# ============================================================


@dataclass
class DatasetInfo:
    """Result of inspecting a dataset.

    Contains everything needed to understand structure and plan conversion.
    This is the primary output of the inspect() operation.

    Attributes:
        path: Path to the dataset.
        format: Detected format identifier ("rlds", "lerobot-v2", etc.).
        format_version: Specific version if applicable.
        num_episodes: Total number of episodes.
        total_frames: Total frames across all episodes.
        observation_schema: Dictionary mapping field names to schemas.
        action_schema: Schema for the action field.
        cameras: Dictionary mapping camera names to CameraInfo.
        has_timestamps: Whether timestamps are present.
        has_language: Whether language instructions are present.
        language_coverage: Fraction of episodes with language (0.0 to 1.0).
        has_rewards: Whether reward signals are present.
        has_success_labels: Whether success labels are present.
        inferred_fps: FPS inferred from data.
        inferred_gripper_index: Detected gripper index in state vector.
        inferred_robot_type: Detected robot type.
        missing_required: List of required fields missing for conversion.
        sample_episode_id: ID of a sample episode.
        sample_num_frames: Frame count of sample episode.
        sample_language: Language instruction from sample episode.
    """

    path: Path
    format: str
    format_version: str | None = None

    # Counts
    num_episodes: int = 0
    total_frames: int = 0

    # Schema
    observation_schema: dict[str, FieldSchema] = field(default_factory=dict)
    action_schema: FieldSchema | None = None
    cameras: dict[str, CameraInfo] = field(default_factory=dict)

    # Internal metadata (not displayed in summary)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Inferred properties
    has_timestamps: bool = False
    has_language: bool = False
    language_coverage: float = 0.0
    has_rewards: bool = False
    has_success_labels: bool = False

    # Detected but uncertain
    inferred_fps: float | None = None
    inferred_gripper_index: int | None = None
    inferred_robot_type: str | None = None

    # What user needs to specify
    missing_required: list[str] = field(default_factory=list)

    # Sample data for display
    sample_episode_id: str | None = None
    sample_num_frames: int | None = None
    sample_language: str | None = None

    def is_ready_for_conversion(self, target_format: str) -> bool:
        """Check if we have everything needed to convert.

        Args:
            target_format: Target format identifier.

        Returns:
            True if no required fields are missing.
        """
        return len(self.missing_required) == 0

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Multi-line string summarizing the dataset.
        """
        lines = [
            f"Dataset: {self.path}",
            f"Format: {self.format}"
            + (f" (v{self.format_version})" if self.format_version else ""),
            f"Episodes: {self.num_episodes}",
            f"Total frames: {self.total_frames}",
        ]

        if self.cameras:
            lines.append(f"Cameras: {', '.join(self.cameras.keys())}")

        if self.inferred_fps:
            lines.append(f"Inferred FPS: {self.inferred_fps}")

        if self.missing_required:
            lines.append(f"Missing for conversion: {', '.join(self.missing_required)}")

        return "\n".join(lines)
