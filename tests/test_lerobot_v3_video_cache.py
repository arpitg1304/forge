"""Regression tests for the LeRobot v3 VideoFrameCache.

Two defects this guards against:
- The cache used to track the frame index by counting up from a *guessed* seek
  origin, so the same index returned different pixels depending on access order
  (and sequential vs random reads disagreed).
- The end-of-stream path returned a black frame, so the last frame of a clip
  decoded blank.

Both are exercised via a real writer -> reader round-trip.
"""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
import forge.formats.lerobot_v3.reader as reader_mod
from forge.formats.lerobot_v3.reader import LeRobotV3Reader
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig

H = W = 64
T = 24
FPS = 30.0


def _expected_mean(t: int) -> float:
    """Per-frame solid value encoding the frame index."""
    return 20 + 4 * t


def _make_episode(ep_idx: int) -> Episode:
    def frame_loader():
        for t in range(T):
            val = _expected_mean(t)

            def loader(v=val):
                return np.full((H, W, 3), v, dtype=np.uint8)

            yield Frame(
                index=t,
                timestamp=t / FPS,
                images={"cam": LazyImage(loader=loader, height=H, width=W, channels=3)},
                state=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
            )

    return Episode(
        episode_id=f"episode_{ep_idx:06d}",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
        fps=FPS,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    episodes = [_make_episode(i) for i in range(2)]
    info = DatasetInfo(
        path=out, format="lerobot-v3", num_episodes=2, total_frames=2 * T,
        inferred_fps=FPS, inferred_robot_type="test",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
    )
    LeRobotV3Writer(LeRobotV3WriterConfig(fps=FPS, robot_type="test")).write_dataset(
        iter(episodes), out, dataset_info=info
    )
    return out


def _episode_frames(dataset: Path, ep_index: int) -> list[Frame]:
    reader_mod._video_cache.clear()  # isolate cache state per scenario
    ep = list(LeRobotV3Reader().read_episodes(dataset))[ep_index]
    return list(ep.frames())


def test_sequential_decode_matches_content(dataset):
    frames = _episode_frames(dataset, 0)
    assert len(frames) == T
    for t in range(T):
        mean = float(frames[t].images["cam"].load().mean())
        assert mean == pytest.approx(_expected_mean(t), abs=8), f"frame {t}"


def test_last_frame_is_not_black(dataset):
    frames = _episode_frames(dataset, 0)
    last = frames[T - 1].images["cam"].load()
    assert last.std() > 1.0 or last.mean() > 1.0
    assert float(last.mean()) == pytest.approx(_expected_mean(T - 1), abs=8)


def test_random_access_matches_sequential(dataset):
    # Sequential reference.
    seq = {t: _episode_frames(dataset, 0)[t].images["cam"].load().copy() for t in range(T)}

    # Scrambled access order (forces backward seeks) must return identical pixels.
    frames = _episode_frames(dataset, 0)
    for t in [T - 1, 0, T // 2, 3, T - 1, 1, T - 2, 5]:
        got = frames[t].images["cam"].load()
        assert np.array_equal(got, seq[t]), f"frame {t} differs under random access"


def test_second_episode_unaffected_by_first(dataset):
    """Decoding episode 0 first must not change episode 1's frames."""
    ep1_isolated = _episode_frames(dataset, 1)
    iso = {t: ep1_isolated[t].images["cam"].load().copy() for t in (0, T // 2, T - 1)}

    # Now read both episodes sequentially through a shared cache.
    reader_mod._video_cache.clear()
    eps = list(LeRobotV3Reader().read_episodes(dataset))
    [f.images["cam"].load() for f in eps[0].frames()]  # warm with episode 0
    ep1_frames = list(eps[1].frames())
    for t in (0, T // 2, T - 1):
        assert np.array_equal(ep1_frames[t].images["cam"].load(), iso[t]), f"frame {t}"
