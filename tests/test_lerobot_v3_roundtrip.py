"""Round-trip tests: LeRobotV3Writer output read back with LeRobotV3Reader."""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import Episode, Frame, LazyImage
from forge.formats.lerobot_v3.reader import LeRobotV3Reader
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _make_episode(episode_idx: int, num_frames: int, task: str) -> Episode:
    def make_image_loader(index: int):
        def loader():
            img = np.full((32, 32, 3), (index * 10) % 256, dtype=np.uint8)
            return img

        return loader

    frames = [
        Frame(
            index=i,
            timestamp=i / 30.0,
            images={
                "camera0": LazyImage(
                    loader=make_image_loader(i), height=32, width=32, channels=3
                )
            },
            state=np.full(3, episode_idx, dtype=np.float32),
            action=np.full(3, episode_idx + 0.5, dtype=np.float32),
        )
        for i in range(num_frames)
    ]
    return Episode(
        episode_id=f"ep_{episode_idx:03d}",
        metadata={"num_frames": num_frames},
        language_instruction=task,
        robot_type="test_robot",
        cameras=["camera0"],
        state_dim=3,
        action_dim=3,
        fps=30.0,
        _frames_cache=frames,
    )


class TestLeRobotV3RoundTrip:
    """Writer output must read back losslessly through the reader."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_language_instruction_survives_round_trip(self, tmp_path: Path):
        """read_episodes restores tasks from v3.0 parquet metadata.

        v3.0 datasets store tasks in meta/tasks.parquet and per-episode
        metadata in meta/episodes/*.parquet (not the v2 jsonl files), so a
        reader that only checks the jsonl paths yields every episode with
        language_instruction=None and round-trips lose the language labels.
        """
        output_dir = tmp_path / "dataset"
        episodes = [
            _make_episode(0, 3, "pick up the red cube"),
            _make_episode(1, 4, "place the cube in the bin"),
        ]
        writer = LeRobotV3Writer(
            LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        )
        writer.write_dataset(iter(episodes), output_dir)

        read_back = list(LeRobotV3Reader().read_episodes(output_dir))

        assert len(read_back) == 2
        assert read_back[0].language_instruction == "pick up the red cube"
        assert read_back[1].language_instruction == "place the cube in the bin"
        assert len(list(read_back[0].frames())) == 3
        assert len(list(read_back[1].frames())) == 4
