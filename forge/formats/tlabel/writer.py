"""TLabel format writer for Forge.

Writes Forge Episodes to .tlabel JSON files with Schema V2 annotations.
Tactile data stored in Frame.extras["tlabel.schema_v2"] is preserved.

See: https://github.com/liesliy/tlabel
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.core.models import DatasetInfo, Episode
from forge.formats.registry import FormatRegistry

if TYPE_CHECKING:
    pass


def _frame_to_tlabel(frame_obj: Any) -> dict[str, Any]:
    """Convert a Forge Frame to a TLabel frame dict."""
    extras = getattr(frame_obj, "extras", {}) or {}
    schema_v2 = extras.get("tlabel.schema_v2", {})

    return {
        "frame_idx": frame_obj.index,
        "timestamp_s": frame_obj.timestamp,
        "schema_v2": schema_v2,
        "confidence": extras.get("tlabel.confidence"),
        "manipulation_phase": extras.get("tlabel.manipulation_phase"),
        "is_first": getattr(frame_obj, "is_first", False),
        "is_last": getattr(frame_obj, "is_last", False),
    }


@FormatRegistry.register_writer("tlabel")
class TLabelWriter:
    """Writer for .tlabel tactile annotation files.

    Converts Forge Episodes back to .tlabel JSON format.
    Tactile data from Frame.extras["tlabel.schema_v2"] is written
    to the standard TLabel Schema V2 frame structure.
    """

    @property
    def format_name(self) -> str:
        return "tlabel"

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
    ) -> None:
        """Write a single episode to a .tlabel file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        ep_suffix = f"_{episode_index:04d}" if episode_index is not None else ""
        out_file = output_path / f"{episode.episode_id}{ep_suffix}.tlabel"

        frames = [_frame_to_tlabel(f) for f in episode.frames()]

        data = {
            "schema_version": episode.metadata.get("schema_version", "0.17.0"),
            "sensor": episode.metadata.get("sensor", {}),
            "capabilities": episode.metadata.get("capabilities", {}),
            "episode": {
                "id": episode.episode_id,
                "task": episode.language_instruction,
                "num_frames": len(frames),
                "success": episode.success,
            },
            "frames": frames,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        """Write multiple episodes to .tlabel files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            self.write_episode(episode, output_path, episode_index=i)

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write a summary manifest.json for the dataset."""
        output_path = Path(output_path)
        tlabel_files = sorted(output_path.glob("*.tlabel"))

        manifest = {
            "format": "tlabel",
            "num_episodes": len(tlabel_files),
            "files": [f.name for f in tlabel_files],
        }

        with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
