"""Configuration models for Forge.

Defines configuration dataclasses for conversion jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FieldMapping:
    """Mapping configuration for a single field.

    Supports mapping source field names to target names, with optional
    transformations.

    Attributes:
        source: Source field name (can include path like 'steps/observation/foo').
        target: Target field name in output format.
        transform: Optional transformation ('normalize', 'binary', etc.).
    """

    source: str
    target: str | None = None
    transform: str | None = None

    def get_target(self) -> str:
        """Get target name, defaulting to normalized source name."""
        if self.target:
            return self.target
        # Normalize: strip common prefixes and use last component
        name = self.source
        for prefix in ["steps/observation/", "observation.", "steps/", "observation/"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        # Replace slashes and dots with underscores for flat names
        return name.replace("/", "_").replace(".", "_")


@dataclass
class ConversionConfig:
    """Configuration for a conversion job.

    Attributes:
        source_format: Source format (usually auto-detected).
        target_format: Target format.
        camera_mapping: Maps original camera names to target names.
        field_mapping: Maps source field names to target names.
        action_field: Source field name for action data.
        state_field: Source field name for state/proprioception data.
        state_joint_indices: Indices for joint positions in state vector.
        state_gripper_index: Index for gripper state in state vector.
        action_joint_indices: Indices for joint actions in action vector.
        action_gripper_index: Index for gripper action in action vector.
        fps: Frames per second (if not in source data).
        robot_type: Robot identifier (e.g., "franka", "widow_x").
        video_codec: Video codec for encoding (default h264).
        video_crf: Video quality (lower = better, 18-28 typical).
        fail_on_error: Stop on first error vs continue.
        skip_existing: Skip episodes that already exist in output.
        compress_videos: Whether to compress video streams.
        include_depth: Whether to include depth camera streams.
        writer_config: Additional writer-specific configuration.
    """

    # Format hints (usually auto-detected)
    source_format: str | None = None
    target_format: str | None = None

    # Camera mapping: original_name -> target_name
    # Supports fuzzy matching (strips common prefixes)
    camera_mapping: dict[str, str] = field(default_factory=dict)

    # Field mapping for observations and other data
    field_mapping: dict[str, FieldMapping] = field(default_factory=dict)

    # Explicit field names for action/state
    action_field: str | None = None
    state_field: str | None = None

    # Proprioception parsing
    state_joint_indices: list[int] | None = None
    state_gripper_index: int | None = None

    action_joint_indices: list[int] | None = None
    action_gripper_index: int | None = None

    # Required metadata (if not in source)
    fps: float | None = None
    robot_type: str | None = None

    # Video encoding
    video_codec: str = "h264"
    video_crf: int = 23

    # Behavior
    fail_on_error: bool = False
    skip_existing: bool = True

    # Output options
    compress_videos: bool = True
    include_depth: bool = True

    # Additional writer-specific config
    writer_config: dict[str, Any] = field(default_factory=dict)

    # Parallelism
    num_workers: int = 1  # Number of parallel workers for episode processing

    def normalize_camera_name(self, name: str) -> str:
        """Normalize a camera name by stripping common prefixes.

        This enables fuzzy matching in camera_mapping.
        E.g., 'steps/observation/agentview_image' -> 'agentview_image'
        """
        for prefix in [
            "steps/observation/",
            "observation.images.",
            "observation/",
            "steps/",
        ]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    def get_camera_target(self, source_name: str) -> str:
        """Get target camera name for a source camera.

        Tries exact match first, then normalized match.
        """
        # Exact match
        if source_name in self.camera_mapping:
            return self.camera_mapping[source_name]

        # Normalized match
        normalized = self.normalize_camera_name(source_name)
        if normalized in self.camera_mapping:
            return self.camera_mapping[normalized]

        # Check if any mapping key is a suffix of the source name
        for key, target in self.camera_mapping.items():
            if source_name.endswith(key) or normalized.endswith(key):
                return target

        # Default: use normalized name
        return normalized

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ConversionConfig":
        """Load configuration from a YAML file.

        Example YAML:
            target_format: lerobot-v3
            fps: 30
            robot_type: franka

            cameras:
              agentview_image: front_cam
              robot0_eye_in_hand_image: wrist_cam

            fields:
              action: steps/action
              state: observation/robot_state

            video:
              codec: h264
              crf: 23

        Args:
            path: Path to YAML config file.

        Returns:
            ConversionConfig instance.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversionConfig":
        """Create config from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            ConversionConfig instance.
        """
        config = cls()

        # Basic settings
        config.source_format = data.get("source_format")
        config.target_format = data.get("target_format")
        config.fps = data.get("fps")
        config.robot_type = data.get("robot_type")
        config.fail_on_error = data.get("fail_on_error", False)
        config.skip_existing = data.get("skip_existing", True)

        # Camera mapping
        cameras = data.get("cameras", {})
        if isinstance(cameras, dict):
            config.camera_mapping = cameras

        # Field mapping
        fields = data.get("fields", {})
        if isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(value, str):
                    config.field_mapping[key] = FieldMapping(source=value)
                elif isinstance(value, dict):
                    config.field_mapping[key] = FieldMapping(
                        source=value.get("source", key),
                        target=value.get("target"),
                        transform=value.get("transform"),
                    )

        # Explicit action/state fields
        # First check explicit config, then look in field_mapping
        action_from_data = data.get("action_field")
        if action_from_data:
            config.action_field = action_from_data
        elif "action" in config.field_mapping:
            config.action_field = config.field_mapping["action"].source

        state_from_data = data.get("state_field")
        if state_from_data:
            config.state_field = state_from_data
        elif "state" in config.field_mapping:
            config.state_field = config.field_mapping["state"].source

        # Index mappings
        config.state_joint_indices = data.get("state_joint_indices")
        config.state_gripper_index = data.get("state_gripper_index")
        config.action_joint_indices = data.get("action_joint_indices")
        config.action_gripper_index = data.get("action_gripper_index")

        # Video settings
        video = data.get("video", {})
        if isinstance(video, dict):
            config.video_codec = video.get("codec", "h264")
            config.video_crf = video.get("crf", 23)
            config.compress_videos = video.get("compress", True)

        # Output options
        config.include_depth = data.get("include_depth", True)

        # Writer-specific config
        config.writer_config = data.get("writer_config", {})

        # Parallelism
        config.num_workers = data.get("num_workers", 1)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        result: dict[str, Any] = {}

        if self.source_format:
            result["source_format"] = self.source_format
        if self.target_format:
            result["target_format"] = self.target_format
        if self.fps:
            result["fps"] = self.fps
        if self.robot_type:
            result["robot_type"] = self.robot_type

        if self.camera_mapping:
            result["cameras"] = self.camera_mapping

        if self.field_mapping:
            fields = {}
            for key, mapping in self.field_mapping.items():
                if mapping.target or mapping.transform:
                    fields[key] = {
                        "source": mapping.source,
                        "target": mapping.target,
                        "transform": mapping.transform,
                    }
                else:
                    fields[key] = mapping.source
            result["fields"] = fields

        if self.action_field:
            result["action_field"] = self.action_field
        if self.state_field:
            result["state_field"] = self.state_field

        if self.video_codec != "h264" or self.video_crf != 23:
            result["video"] = {
                "codec": self.video_codec,
                "crf": self.video_crf,
            }

        if self.fail_on_error:
            result["fail_on_error"] = True
        if not self.skip_existing:
            result["skip_existing"] = False
        if not self.include_depth:
            result["include_depth"] = False

        if self.num_workers > 1:
            result["num_workers"] = self.num_workers

        if self.writer_config:
            result["writer_config"] = dict(self.writer_config)

        return result

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to output YAML file.
        """
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
