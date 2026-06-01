"""Tests for the `forge tokenize` CLI sub-app."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.core.models import Episode, Frame

pytest.importorskip("pyarrow")

runner = CliRunner()


def _write_lerobot_v3_source(tmp_path, num_episodes=3, length=12, dim=7):
    """Write a tiny real LeRobot v3 dataset to disk via the writer."""
    from forge.core.models import DatasetInfo
    from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig

    rng = np.random.default_rng(0)
    out = tmp_path / "src"
    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0))
    for e in range(num_episodes):
        frames = [
            Frame(
                index=j,
                timestamp=j / 30.0,
                action=(rng.standard_normal(dim) * 2.0).astype(np.float32),
            )
            for j in range(length)
        ]
        ep = Episode(episode_id=f"ep_{e}", fps=30.0, _frames_cache=frames)
        writer.write_episode(ep, out, episode_index=e)
    writer.finalize(
        out,
        DatasetInfo(
            path=out,
            format="lerobot-v3",
            num_episodes=num_episodes,
            total_frames=num_episodes * length,
            inferred_fps=30.0,
        ),
    )
    return out


def test_tokenize_list():
    result = runner.invoke(app, ["tokenize", "list"])
    assert result.exit_code == 0
    assert "uniform-bins" in result.output
    assert "openvla-bins" in result.output


def test_tokenize_compare(tmp_path):
    src = _write_lerobot_v3_source(tmp_path)
    export = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["tokenize", "compare", str(src), "--sample", "20", "--export", str(export)],
    )
    assert result.exit_code == 0, result.output
    assert export.exists()
    report = json.loads(export.read_text())
    assert "uniform-bins" in report["stats_by_strategy"]


def test_tokenize_fit(tmp_path):
    src = _write_lerobot_v3_source(tmp_path)
    out = tmp_path / "tok.json"
    result = runner.invoke(
        app,
        ["tokenize", "fit", str(src), "--strategy", "openvla-bins", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    from forge.tokenize import load_tokenizer

    assert load_tokenizer(out).name == "openvla-bins"


def test_tokenize_write(tmp_path):
    src = _write_lerobot_v3_source(tmp_path)
    out = tmp_path / "tokenized"
    result = runner.invoke(
        app,
        ["tokenize", "write", str(src), str(out), "--strategy", "uniform-bins"],
    )
    assert result.exit_code == 0, result.output
    info = json.loads((out / "meta" / "info.json").read_text())
    assert "action_tokens" in info["features"]
    assert (out / "meta" / "action_tokenizer.json").exists()
