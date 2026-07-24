"""TLabel format reader for Forge.

Reads .tlabel JSON files containing tactile sensor annotations.
TLabel Schema V2 defines 14 semantic dimensions for tactile data
(contact, force, slip, vibration, texture, etc.).

See: https://github.com/liesliy/tlabel
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.core.exceptions import (
    EpisodeNotFoundError,
    InspectionError,
)
from forge.core.models import (
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
)
from forge.formats.registry import FormatRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _load_tlabel(path: Path) -> dict[str, Any]:
    """Load and validate a .tlabel JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"TLabel file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic structure check
    if "frames" not in data and "frame" not in data:
        raise InspectionError(f"Not a valid TLabel file: missing 'frames' field")

    return data


def _frame_to_forge(tlabel_frame: dict[str, Any]) -> Frame:
    """Convert a single TLabel frame dict to a Forge Frame."""
    schema_v2 = tlabel_frame.get("schema_v2", {})

    return Frame(
        index=tlabel_frame.get("frame_idx", 0),
        timestamp=tlabel_frame.get("timestamp_s"),
        is_first=tlabel_frame.get("is_first", False),
        is_last=tlabel_frame.get("is_last", False),
        extras={
            "tlabel.schema_v2": schema_v2,
            "tlabel.confidence": tlabel_frame.get("confidence"),
            "tlabel.manipulation_phase": tlabel_frame.get("manipulation_phase"),
        },
    )


@FormatRegistry.register_reader("tlabel")
class TLabelReader:
    """Reader for .tlabel tactile annotation files.

    TLabel files contain per-frame tactile sensor annotations with a
    14-dimensional semantic schema (Schema V2). Each frame maps to a
    Forge Frame with tactile data stored in Frame.extras.
    """

    @property
    def format_name(self) -> str:
        return "tlabel"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check if path looks like a .tlabel file or directory containing them."""
        path = Path(path)

        # Single .tlabel file
        if path.is_file() and path.suffix == ".tlabel":
            return True

        # Directory with .tlabel files
        if path.is_dir():
            return any(path.glob("*.tlabel"))

        # JSON file with TLabel structure (some use .json extension)
        if path.is_file() and path.suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return "frames" in data and "schema_version" in data
            except (json.JSONDecodeError, OSError):
                return False

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect TLabel schema version."""
        path = Path(path)
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("schema_version_v2", data.get("schema_version"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze a .tlabel file's structure."""
        path = Path(path)
        data = _load_tlabel(path)

        frames = data.get("frames", [])
        num_frames = len(frames)
        sensor = data.get("sensor", {})

        # Build observation schema from capabilities
        capabilities = data.get("capabilities", {})
        obs_schema = {}
        for dim_name, enabled in capabilities.items():
            if enabled:
                obs_schema[f"tlabel.{dim_name}"] = FieldSchema(
                    name=f"tlabel.{dim_name}",
                    shape=(1,),
                    dtype=Dtype.FLOAT32,
                    description=f"TLabel dimension: {dim_name}",
                )

        # Infer FPS from timestamps
        inferred_fps = None
        if num_frames >= 2:
            t0 = frames[0].get("timestamp_s", 0)
            t1 = frames[1].get("timestamp_s", 0)
            dt = t1 - t0
            if dt > 0:
                inferred_fps = round(1.0 / dt, 1)

        return DatasetInfo(
            path=path,
            format="tlabel",
            format_version=data.get("schema_version"),
            num_episodes=1,  # One file = one episode
            total_frames=num_frames,
            observation_schema=obs_schema,
            cameras={},
            has_timestamps=all("timestamp_s" in f for f in frames[:10]),
            has_language=data.get("episode", {}).get("task") is not None,
            has_rewards=False,
            has_success_labels=data.get("episode", {}).get("success") is not None,
            inferred_fps=inferred_fps,
            inferred_robot_type=sensor.get("name"),
            sample_num_frames=num_frames,
            sample_language=data.get("episode", {}).get("task"),
        )

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Yield episodes from a .tlabel file (one file = one episode)."""
        path = Path(path)

        # If directory, iterate .tlabel files
        if path.is_dir():
            for tlabel_file in sorted(path.glob("*.tlabel")):
                yield from self._read_single(tlabel_file)
            for json_file in sorted(path.glob("*.json")):
                if self.can_read(json_file):
                    yield from self._read_single(json_file)
        else:
            yield from self._read_single(path)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID."""
        for ep in self.read_episodes(path):
            if ep.episode_id == episode_id:
                return ep
        raise EpisodeNotFoundError(episode_id, str(path))

    def _read_single(self, file_path: Path) -> Iterator[Episode]:
        """Read a single .tlabel file as one Episode."""
        data = _load_tlabel(file_path)
        episode_meta = data.get("episode", {})
        frames_data = data.get("frames", [])

        # Build Episode
        ep_id = episode_meta.get("id", file_path.stem)
        episode = Episode(
            episode_id=ep_id,
            metadata={
                "sensor": data.get("sensor", {}),
                "capabilities": data.get("capabilities", {}),
                "schema_version": data.get("schema_version"),
            },
            language_instruction=episode_meta.get("task"),
            success=episode_meta.get("success"),
            fps=self._infer_fps(frames_data),
        )

        # Lazy frame loader
        def _load_frames() -> Iterator[Frame]:
            for fd in frames_data:
                yield _frame_to_forge(fd)

        episode._frame_loader = _load_frames
        yield episode

    @staticmethod
    def _infer_fps(frames_data: list[dict]) -> float | None:
        if len(frames_data) >= 2:
            t0 = frames_data[0].get("timestamp_s", 0)
            t1 = frames_data[1].get("timestamp_s", 0)
            dt = t1 - t0
            if dt > 0:
                return round(1.0 / dt, 1)
        return None
