"""Regression test: LeRobot v3 video round-trip across multiple episodes.

forge's v3 writer flushes one chunk per episode (each episode's video is its own
chunk-XXX/file-000.mp4) but stamps info.json with the multi-episode
"{file_index}" template. The reader then seeks by GLOBAL frame index and, for
every episode after the first, walked off the end of the per-episode file and
returned black frames. This test writes distinct content per episode, reads it
back through the reader, and asserts each episode decodes to its own content.
"""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v3.reader import LeRobotV3Reader
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig

H = W = 64
T = 12
FPS = 30.0
N_EPISODES = 3
BASE = [40, 100, 160]   # distinct per-episode luma base


def _expected_mean(ep_idx: int, frame_idx: int) -> float:
    """Solid-frame value used by the generator: base + 3*frame_idx."""
    return BASE[ep_idx] + 3 * frame_idx


def _make_episode(ep_idx: int) -> Episode:
    def frame_loader():
        for t in range(T):
            val = _expected_mean(ep_idx, t)

            def loader(v=val):
                return np.full((H, W, 3), v, dtype=np.uint8)

            yield Frame(
                index=t,
                timestamp=t / FPS,
                images={"cam": LazyImage(loader=loader, height=H, width=W, channels=3)},
                state=np.full(7, 0.1 * ep_idx, dtype=np.float32),
                action=np.full(7, 0.01 * ep_idx, dtype=np.float32),
            )

    return Episode(
        episode_id=f"episode_{ep_idx:06d}",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
        fps=FPS,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def written_dataset(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    episodes = [_make_episode(i) for i in range(N_EPISODES)]
    info = DatasetInfo(
        path=out,
        format="lerobot-v3",
        num_episodes=N_EPISODES,
        total_frames=N_EPISODES * T,
        inferred_fps=FPS,
        inferred_robot_type="test",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
    )
    LeRobotV3Writer(LeRobotV3WriterConfig(fps=FPS, robot_type="test")).write_dataset(
        iter(episodes), out, dataset_info=info
    )
    return out


def test_each_episode_decodes_its_own_video(written_dataset: Path):
    """Every episode — not just the first — must read back its real frames."""
    reader = LeRobotV3Reader()
    episodes = list(reader.read_episodes(written_dataset))
    assert len(episodes) == N_EPISODES

    per_episode_frame8_means = []
    for ep_idx, ep in enumerate(episodes):
        frames = list(ep.frames())
        assert len(frames) == T
        img = frames[8].images["cam"].load()
        mean = float(img.mean())
        per_episode_frame8_means.append(mean)
        # Correct episode AND correct frame offset within it (regression guard:
        # the bug returned ~0 / black for episodes 1+).
        assert mean == pytest.approx(_expected_mean(ep_idx, 8), abs=10), (
            f"episode {ep_idx} frame 8 mean {mean:.1f} != "
            f"expected {_expected_mean(ep_idx, 8)}"
        )

    # Episodes must be distinct from one another (not all collapsed to black).
    assert max(per_episode_frame8_means) - min(per_episode_frame8_means) > 50


def test_frame_offset_is_correct_within_episode(written_dataset: Path):
    """Frames are read in order, not all pinned to the same frame."""
    reader = LeRobotV3Reader()
    episodes = list(reader.read_episodes(written_dataset))
    last_ep = episodes[-1]  # the episode most affected by the global-index bug
    frames = list(last_ep.frames())
    means = [float(frames[t].images["cam"].load().mean()) for t in (0, 5, 11)]
    for t, m in zip((0, 5, 11), means):
        assert m == pytest.approx(_expected_mean(N_EPISODES - 1, t), abs=10)
    # Strictly increasing content over time.
    assert means[0] < means[1] < means[2]
