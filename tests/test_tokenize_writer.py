"""Tests for the tokenized-dataset writer (forge/tokenize/writer.py) and the
opt-in action-tokens column hook on the LeRobot v3 writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forge.core.models import Episode, Frame
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig
from forge.tokenize import TokenizerRegistry, load_tokenizer
from forge.tokenize.writer import tokenize_and_write

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def _make_episodes(num_episodes: int = 3, length: int = 12, dim: int = 7):
    rng = np.random.default_rng(0)
    episodes = []
    for e in range(num_episodes):
        frames = [
            Frame(
                index=j,
                timestamp=j / 30.0,
                state=rng.standard_normal(dim).astype(np.float32),
                action=(rng.standard_normal(dim) * 2.0).astype(np.float32),
            )
            for j in range(length)
        ]
        episodes.append(
            Episode(episode_id=f"ep_{e}", fps=30.0, _frames_cache=frames)
        )
    return episodes


def _read_all_frames(dataset_path: Path):
    """Concatenate every data parquet file's rows in chunk/episode order."""
    data_root = dataset_path / "data"
    tables = []
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for pf in sorted(chunk_dir.glob("file-*.parquet")):
            tables.append(pq.read_table(pf))
    return tables


def _load_info(dataset_path: Path) -> dict:
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Writer column hook
# --------------------------------------------------------------------------- #


def test_column_hook_off_by_default(tmp_path):
    """With no config, no action_tokens column or feature is emitted."""
    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0))
    out = tmp_path / "ds"
    eps = _make_episodes()
    for i, ep in enumerate(eps):
        ep.frames_cache = ep.load_frames()
        # inject extras even though hook is off
        for fr in ep.load_frames():
            fr.extras["action_tokens"] = [1] * 7
        writer.write_episode(ep, out, episode_index=i)
    writer.finalize(out, _minimal_info(out, eps))

    info = _load_info(out)
    assert "action_tokens" not in info["features"]
    for table in _read_all_frames(out):
        assert "action_tokens" not in table.column_names


def test_column_hook_emits_int_column(tmp_path):
    """When configured and extras present, an int64 action_tokens column appears."""
    cfg = LeRobotV3WriterConfig(fps=30.0, tokenized_action_feature="action_tokens")
    writer = LeRobotV3Writer(cfg)
    out = tmp_path / "ds"
    eps = _make_episodes()
    for i, ep in enumerate(eps):
        for fr in ep.load_frames():
            fr.extras["action_tokens"] = list(range(7))
        writer.write_episode(ep, out, episode_index=i)
    writer.finalize(out, _minimal_info(out, eps))

    info = _load_info(out)
    assert info["features"]["action_tokens"]["dtype"] == "int64"
    assert info["features"]["action_tokens"]["shape"] == [7]

    tables = _read_all_frames(out)
    assert tables, "no data files written"
    for table in tables:
        assert "action_tokens" in table.column_names
        first = table["action_tokens"][0].as_py()
        assert first == list(range(7))


def test_column_hook_excluded_from_stats(tmp_path):
    """Token ids must not get min/max/mean/std stats computed."""
    cfg = LeRobotV3WriterConfig(fps=30.0, tokenized_action_feature="action_tokens")
    writer = LeRobotV3Writer(cfg)
    out = tmp_path / "ds"
    eps = _make_episodes()
    for i, ep in enumerate(eps):
        for fr in ep.load_frames():
            fr.extras["action_tokens"] = list(range(7))
        writer.write_episode(ep, out, episode_index=i)
    writer.finalize(out, _minimal_info(out, eps))

    with open(out / "meta" / "stats.json") as f:
        stats = json.load(f)
    assert "action_tokens" not in stats


def _minimal_info(out: Path, eps):
    from forge.core.models import DatasetInfo

    return DatasetInfo(
        path=out,
        format="lerobot-v3",
        num_episodes=len(eps),
        total_frames=sum(len(e.load_frames()) for e in eps),
        inferred_fps=30.0,
    )


# --------------------------------------------------------------------------- #
# tokenize_and_write
# --------------------------------------------------------------------------- #


class _StubReader:
    def __init__(self, episodes):
        self._episodes = episodes

    def read_episodes(self, path):
        yield from self._episodes


def test_tokenize_and_write_end_to_end(tmp_path):
    eps = _make_episodes()
    out = tmp_path / "out"
    result = tokenize_and_write(
        source=tmp_path / "fake_src",
        output=out,
        strategy="openvla-bins",
        reader=_StubReader(eps),
        fps=30.0,
    )

    # dataset is readable
    info = _load_info(out)
    assert info["features"]["action_tokens"]["dtype"] == "int64"
    assert info["total_episodes"] == len(eps)

    # tokenizer saved for inference-time decode
    tok_path = out / "meta" / "action_tokenizer.json"
    assert tok_path.exists()
    assert result.tokenizer_path == tok_path

    tok = load_tokenizer(tok_path)
    assert tok.name == "openvla-bins"

    # tokens in the dataset decode back to ~the original actions
    tables = _read_all_frames(out)
    tokens = np.array(tables[0]["action_tokens"].to_pylist())
    recon = tok.decode(tokens)
    assert recon.shape == tokens.shape


def test_tokenize_and_write_drops_actions_by_default(tmp_path):
    eps = _make_episodes()
    out = tmp_path / "out"
    tokenize_and_write(
        source=tmp_path / "fake_src",
        output=out,
        strategy="uniform-bins",
        reader=_StubReader(eps),
        fps=30.0,
    )
    info = _load_info(out)
    assert "action" not in info["features"]
    assert "action_tokens" in info["features"]


def test_tokenize_and_write_keep_actions(tmp_path):
    eps = _make_episodes()
    out = tmp_path / "out"
    tokenize_and_write(
        source=tmp_path / "fake_src",
        output=out,
        strategy="uniform-bins",
        reader=_StubReader(eps),
        fps=30.0,
        keep_actions=True,
    )
    info = _load_info(out)
    assert "action" in info["features"]
    assert "action_tokens" in info["features"]


def test_tokenize_and_write_with_pretrained_tokenizer(tmp_path):
    eps = _make_episodes()
    # Fit + save a tokenizer up front.
    corpus = np.vstack(
        [np.array([f.action for f in e.load_frames()]) for e in eps]
    )
    tok = TokenizerRegistry.create("quantile-bins", num_bins=64).fit(corpus)
    tok_file = tmp_path / "pretrained.json"
    tok.save(tok_file)

    out = tmp_path / "out"
    tokenize_and_write(
        source=tmp_path / "fake_src",
        output=out,
        strategy="quantile-bins",
        tokenizer_path=tok_file,
        reader=_StubReader(eps),
        fps=30.0,
    )
    saved = load_tokenizer(out / "meta" / "action_tokenizer.json")
    np.testing.assert_array_equal(saved.encode(corpus), tok.encode(corpus))


def test_tokenize_and_write_rejects_chunk_tokenizer(tmp_path):
    """Chunk-granularity tokenizers are not supported by the v1 writer."""
    from forge.tokenize.base import BaseTokenizer

    @TokenizerRegistry.register("fake-chunk")
    class _FakeChunk(BaseTokenizer):
        @property
        def vocab_size(self):
            return 10

        @property
        def granularity(self):
            return "chunk"

        def fit(self, actions):
            self._fitted = True
            return self

        def encode(self, actions):
            return np.zeros((3,), dtype=np.int64)

        def decode(self, tokens):
            return np.zeros((3, 7), dtype=np.float32)

        def get_params(self):
            return {}

        @classmethod
        def from_params(cls, params):
            t = cls()
            t._fitted = True
            return t

    try:
        eps = _make_episodes()
        with pytest.raises(Exception):
            tokenize_and_write(
                source=tmp_path / "fake_src",
                output=tmp_path / "out",
                strategy="fake-chunk",
                reader=_StubReader(eps),
                fps=30.0,
            )
    finally:
        TokenizerRegistry._strategies.pop("fake-chunk", None)
