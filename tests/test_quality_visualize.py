"""Quality review must preserve episode identities and frame alignment."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from typer.testing import CliRunner

from forge.cli import app
from forge.core.models import Episode, Frame
from forge.formats.registry import FormatRegistry
from forge.quality import QualityAnalyzer, QualityReport
from forge.visualize.web_viewer import WebBackend


def episode(name, missing=False):
    frames = [
        Frame(
            index=i,
            timestamp=i / 30,
            action=np.array([1.0]),
            state=None if missing and i == 7 else np.array([1.0]),
        )
        for i in range(40)
    ]
    return Episode(episode_id=name, fps=30, _frames_cache=frames)


def reader_fixture(monkeypatch, episodes):
    reader = SimpleNamespace(
        read_episodes=lambda path: iter(episodes),
        inspect=lambda path: SimpleNamespace(inferred_fps=30),
    )
    monkeypatch.setattr(FormatRegistry, "detect_format", lambda path: "test")
    monkeypatch.setattr(FormatRegistry, "get_reader", lambda name: reader)


def test_backend_joins_report_by_id_and_preserves_missing_frames(monkeypatch, tmp_path):
    selected = episode("chosen", missing=True)
    reader_fixture(monkeypatch, [episode("skip"), selected])
    eq = QualityAnalyzer().analyze_episode(selected)
    assert "nan_in_states" in eq.flags
    report = QualityReport(str(tmp_path), num_episodes=1, per_episode=[eq])
    backend = WebBackend(tmp_path, max_episodes=1, quality_report=report)
    assert backend.get_info_dict()["episode_ids"] == ["chosen"]
    payload = backend.get_episode_dict(0)
    assert payload["quality"]["episode_id"] == "chosen"
    assert len(payload["states"]) == 40
    assert payload["states"][7] == [None]
    assert payload["states"][8] == [1.0]
    json.dumps(payload, allow_nan=False)


def test_cli_visualize_passes_analyzed_report_and_exports_events(monkeypatch, tmp_path):
    reader_fixture(monkeypatch, [episode("chosen"), episode("skip")])
    viewer = Mock()
    monkeypatch.setattr("forge.visualize.web_viewer.web_visualize", viewer)
    export = tmp_path / "report.json"
    result = CliRunner().invoke(
        app, ["quality", str(tmp_path), "--sample", "1", "--visualize", "--export", str(export)]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    viewer.assert_called_once()
    report = viewer.call_args.kwargs["quality_report"]
    assert [q.episode_id for q in report.per_episode] == ["chosen"]
    assert viewer.call_args.kwargs["max_episodes"] == 1
    saved = json.loads(export.read_text())
    assert saved["per_episode"][0]["events"]
    assert report.per_episode[0].grippers == {}


def test_cli_rejects_unmapped_tracking_semantics(tmp_path):
    result = CliRunner().invoke(app, ["quality", str(tmp_path), "--tracking-pairs", "0:0"])
    assert result.exit_code != 0
    assert "position" in result.output


def test_lerobot_reader_preserves_metadata_fps_for_quality(tmp_path):
    import pandas as pd

    from forge.formats.lerobot_v3.reader import LeRobotV3Reader

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 15, "features": {}}))
    t = np.arange(60) / 15
    df = pd.DataFrame(
        {
            "timestamp": t,
            "index": np.arange(60),
            "observation.state": [np.array([x]) for x in t],
            "action": [np.array([x]) for x in t],
        }
    )
    ep = LeRobotV3Reader()._build_episode(
        tmp_path, tmp_path / "data.parquet", df, 0, "episode_000000", None
    )
    assert ep.fps == 15
    eq = QualityAnalyzer().analyze_episode(ep)
    assert eq.provenance["fps"] == 15
    assert eq.overall_score is not None
    assert "timestamp_gaps" not in eq.flags
    assert "sample_rate_mismatch" not in eq.flags


def test_explorer_metadata_and_all_channels_are_preserved(monkeypatch, tmp_path):
    from forge.core.models import LazyImage

    first = episode("first")
    second = episode("second")
    second.language_instruction = "Place the cup on the tray"
    for f in second._frames_cache:
        f.action = np.arange(14, dtype=float)
        f.images = {
            "wrist": LazyImage(
                loader=lambda: np.zeros((8, 8, 3), dtype=np.uint8), height=8, width=8
            )
        }
    reader_fixture(monkeypatch, [first, second])
    backend = WebBackend(tmp_path)
    info = backend.get_info_dict()
    assert info["cameras"] == ["wrist"]
    assert info["episodes"][1]["task"] == "Place the cup on the tray"
    assert info["episodes"][1]["num_frames"] == 40
    assert info["quality_mode"] is False
    assert backend.get_episode_dict(0)["cameras"] == []
    payload = backend.get_episode_dict(1)
    assert payload["cameras"] == ["wrist"]
    assert payload["quality"] is None
    assert len(payload["actions"][0]) == 14
    assert payload["timestamps"][7] == 7 / 30
    json.dumps(payload, allow_nan=False)


def test_episode_overview_counts_review_events_not_reversals(monkeypatch, tmp_path):
    from forge.quality import EpisodeQuality, QualityEvent

    ep = episode("review")
    reader_fixture(monkeypatch, [ep])
    eq = EpisodeQuality(
        "review",
        num_frames=40,
        overall_score=7.5,
        flags=["dead_actions"],
        events=[
            QualityEvent("direction_reversal", 1, 5, 0.03, 0.13),
            QualityEvent("held_command", 10, 40, 0.33, 1.3),
        ],
    )
    backend = WebBackend(tmp_path, quality_report=QualityReport(str(tmp_path), per_episode=[eq]))
    summary = backend.get_info_dict()["episodes"][0]
    assert summary["event_count"] == 1
    assert summary["score"] == 7.5
    assert summary["flags"] == ["dead_actions"]


def test_cli_passes_video_integrity_options_and_exports_camera_events(monkeypatch, tmp_path):
    reader_fixture(monkeypatch, [episode("chosen")])
    viewer = Mock()
    monkeypatch.setattr("forge.visualize.web_viewer.web_visualize", viewer)
    result = CliRunner().invoke(
        app,
        [
            "quality",
            str(tmp_path),
            "--video",
            "--video-cameras",
            "absent",
            "--video-drop-frames",
            "4",
            "--video-gap-seconds",
            ".4",
            "--video-fps",
            "15",
            "--visualize",
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    video = viewer.call_args.kwargs["quality_report"].per_episode[0].video
    assert video.per_camera["absent"].missing_frames == 40
    assert video.provenance["config"]["significant_drop_frames"] == 4
    assert video.provenance["config"]["expected_fps"] == 15


def test_cli_rejects_invalid_video_sampling(tmp_path):
    result = CliRunner().invoke(app, ["quality", str(tmp_path), "--video", "--video-stride", "0"])
    assert result.exit_code != 0
    assert "sample_stride" in result.output
