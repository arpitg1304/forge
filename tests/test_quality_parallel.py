"""Tests for parallel quality analysis (forge quality --workers).

The parallel path uses a ``spawn`` ProcessPoolExecutor, so workers re-import
fresh and only see formats registered at import time — hence the integration
test runs against a real written LeRobot v3 dataset rather than a mock reader.
"""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig
from forge.formats.registry import FormatRegistry
from forge.quality.analyzer import QualityAnalyzer
from forge.quality.config import QualityConfig
from forge.quality.parallel import _chunk_bounds, analyze_parallel

H = W = 16
T = 16
FPS = 30.0
N = 6


# ── chunk bounds (pure, no subprocess) ───────────────────────────


def test_chunk_bounds_even():
    assert _chunk_bounds(10, 2) == [(0, 5), (5, 10)]


def test_chunk_bounds_uneven_covers_all():
    bounds = _chunk_bounds(10, 3)
    assert bounds[0][0] == 0 and bounds[-1][1] == 10
    for (a, b), (c, d) in zip(bounds, bounds[1:]):
        assert b == c  # contiguous
    assert sum(e - s for s, e in bounds) == 10


def test_chunk_bounds_more_workers_than_items():
    bounds = _chunk_bounds(3, 8)
    assert sum(e - s for s, e in bounds) == 3
    assert all(e > s for s, e in bounds)  # no empty chunks


# ── real-dataset parallel == sequential ──────────────────────────


def _make_episode(idx: int) -> Episode:
    rng = np.random.default_rng(idx)
    traj = np.cumsum(rng.normal(0, 0.02, (T, 7)), axis=0).astype(np.float32)

    def frame_loader():
        for t in range(T):
            yield Frame(
                index=t,
                timestamp=t / FPS,
                images={
                    "cam": LazyImage(
                        loader=lambda: np.full((H, W, 3), 120, dtype=np.uint8),
                        height=H, width=W, channels=3,
                    )
                },
                state=traj[t],
                action=traj[t],
            )

    return Episode(
        episode_id=f"episode_{idx:06d}",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
        fps=FPS,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    info = DatasetInfo(
        path=out, format="lerobot-v3", num_episodes=N, total_frames=N * T,
        inferred_fps=FPS, inferred_robot_type="test",
        cameras={"cam": CameraInfo(name="cam", height=H, width=W)},
    )
    LeRobotV3Writer(LeRobotV3WriterConfig(fps=FPS, robot_type="test")).write_dataset(
        (_make_episode(i) for i in range(N)), out, dataset_info=info
    )
    return out


def test_parallel_matches_sequential(dataset):
    config = QualityConfig(fps=FPS)
    reader = FormatRegistry.get_reader("lerobot-v3")
    analyzer = QualityAnalyzer(config=config)
    seq = {
        eq.episode_id: eq
        for eq in (analyzer.analyze_episode(ep) for ep in reader.read_episodes(dataset))
    }

    episodes, errors = analyze_parallel(dataset, "lerobot-v3", N, config, None, num_workers=3)

    assert errors == []
    assert [e.episode_id for e in episodes] == sorted(seq)  # ordered, complete
    for eq in episodes:
        ref = seq[eq.episode_id]
        assert eq.overall_score == pytest.approx(ref.overall_score, rel=1e-9)
        assert eq.flags == ref.flags
        assert eq.num_frames == ref.num_frames


def test_parallel_progress_counts_all(dataset):
    seen: list[tuple[int, int]] = []
    analyze_parallel(
        dataset, "lerobot-v3", N, QualityConfig(fps=FPS), None,
        num_workers=3, progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1][0] == N
    assert all(t == N for _, t in seen)
