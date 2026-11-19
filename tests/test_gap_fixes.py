"""Regression tests for the bugs fixed from FORGE_GAPS.md (B2, B4, B7, B8, B20)."""

import json
from pathlib import Path

import pytest

from forge.filter.engine import FilterConfig, FilterEngine
from forge.quality.models import EpisodeQuality


def test_b2_min_quality_rejects_unscored_episode():
    engine = FilterEngine(FilterConfig(min_quality=6.0))
    eq = EpisodeQuality(episode_id="ep_000", overall_score=None, flags=[])
    keep, reasons = engine._evaluate_episode("ep_000", eq)
    assert not keep
    assert any("no quality score" in r for r in reasons)


def test_b2_unscored_episode_kept_without_quality_gate():
    engine = FilterEngine(FilterConfig(exclude_flags=["jerky"]))
    eq = EpisodeQuality(episode_id="ep_000", overall_score=None, flags=[])
    keep, _ = engine._evaluate_episode("ep_000", eq)
    assert keep


def test_b7_chatter_recommendation_uses_existing_flag():
    import inspect

    from forge.quality import analyzer

    src = inspect.getsource(analyzer)
    assert "--max-gripper-chatter" not in src
    assert "--exclude-flags gripper_chatter" in src


def test_b20_version_matches_pyproject():
    import forge

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert f'version = "{forge.__version__}"' in pyproject


def _lerobot_dir(tmp_path: Path, codebase_version: str) -> Path:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": codebase_version, "fps": 30}))
    (root / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "pick"}\n')
    return root


def test_b8_v21_with_tasks_jsonl_is_not_v3(tmp_path: Path):
    from forge.formats.lerobot_v3.reader import LeRobotV3Reader

    assert not LeRobotV3Reader.can_read(_lerobot_dir(tmp_path, "v2.1"))


def test_b8_v3_still_detected(tmp_path: Path):
    from forge.formats.lerobot_v3.reader import LeRobotV3Reader

    assert LeRobotV3Reader.can_read(_lerobot_dir(tmp_path, "v3.0"))


def _writer_deps() -> bool:
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _episodes(fps: float, n: int = 2):
    import numpy as np

    from forge.core.models import CameraInfo, Episode, Frame, LazyImage

    def make(ep: int) -> Episode:
        def frames():
            for i in range(10):
                yield Frame(
                    index=i,
                    timestamp=i / fps,
                    images={"camera0": LazyImage(loader=lambda: np.zeros((32, 32, 3), np.uint8), height=32, width=32, channels=3)},
                    state=np.zeros(3, np.float32),
                    action=np.zeros(3, np.float32),
                )

        return Episode(
            episode_id=f"ep_{ep:03d}",
            language_instruction="pick",
            cameras={"camera0": CameraInfo(name="camera0", height=32, width=32)},
            fps=fps,
            _frame_loader=frames,
        )

    return [make(i) for i in range(n)]


@pytest.mark.skipif(not _writer_deps(), reason="PyAV or PyArrow not installed")
@pytest.mark.parametrize("fps,expected", [(12.5, 12.5), (29.97, 29.97), (30.0, 30)])
def test_b4_v3_writer_keeps_fractional_fps(tmp_path: Path, fps, expected):
    from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig

    sample_episodes = _episodes(fps)

    out = tmp_path / "out"
    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=fps, robot_type="test_robot"))
    writer.write_dataset(iter(sample_episodes), out)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["fps"] == expected
    assert isinstance(info["fps"], type(expected))
