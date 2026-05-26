"""Upstream-validity tests: assert Forge-written v3 datasets load with the
real `lerobot` package's LeRobotDataset class.

This is the gold-standard test — anything weaker can lie. If `lerobot`'s
loader can open a Forge-written dataset, it's compatible with the wider
HuggingFace robotics ecosystem.

Skipped if `lerobot` isn't installed (it's a heavy optional dep, ~1.5GB
including torch). To enable:

    pip install lerobot
    pytest tests/test_lerobot_v3_upstream_validity.py -v

Each test isolates one structural requirement (info.json schema, episodes
parquet columns, tasks.parquet index, etc.) so a failure points at exactly
which v3 invariant is violated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Heavy optional dep — skip the whole module if not installed.
lerobot = pytest.importorskip(
    "lerobot",
    reason="`lerobot` not installed — `pip install lerobot` to run upstream validity checks",
)

import pyarrow.parquet as pq  # noqa: E402

from forge.core.models import CameraInfo, Episode, Frame, LazyImage  # noqa: E402
from forge.formats.lerobot_v3.writer import (  # noqa: E402
    LeRobotV3Writer,
    LeRobotV3WriterConfig,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------


def _synth_episode(episode_id: str, n_frames: int = 30, state_dim: int = 7) -> Episode:
    rng = np.random.default_rng(seed=hash(episode_id) % (2**31))
    state_seq = rng.standard_normal((n_frames, state_dim)).astype(np.float32)
    action_seq = rng.standard_normal((n_frames, state_dim)).astype(np.float32)
    img_seq = rng.integers(0, 255, (n_frames, 16, 16, 3), dtype=np.uint8)

    def gen():
        for i in range(n_frames):
            yield Frame(
                index=i,
                timestamp=float(i) / 30.0,
                state=state_seq[i],
                action=action_seq[i],
                images={
                    "top": LazyImage(
                        loader=lambda im=img_seq[i]: im,
                        height=16, width=16, channels=3,
                    )
                },
                is_first=(i == 0),
                is_last=(i == n_frames - 1),
            )

    return Episode(
        episode_id=episode_id,
        cameras={"top": CameraInfo(name="top", height=16, width=16, channels=3)},
        state_dim=state_dim, action_dim=state_dim, fps=30.0,
        language_instruction=f"task for {episode_id}",
        _frame_loader=gen,
    )


@pytest.fixture
def written_v3(tmp_path: Path) -> Path:
    """Write a 3-episode v3 dataset and return its path."""
    out = tmp_path / "synth_v3"
    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, robot_type="synth"))
    episodes = (_synth_episode(f"ep_{i:03d}") for i in range(3))
    writer.write_dataset(episodes, out)
    return out


# ---------------------------------------------------------------------------
# Constants — sourced from upstream lerobot==0.5.1
# ---------------------------------------------------------------------------


from lerobot.datasets.utils import (  # noqa: E402
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    STATS_PATH,
)


# ---------------------------------------------------------------------------
# 1. info.json schema
# ---------------------------------------------------------------------------


class TestInfoJson:
    def test_codebase_version_exact(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        assert info["codebase_version"] == "v3.0", \
            f"upstream loader hard-rejects anything other than 'v3.0' at lerobot/datasets/utils.py:check_version_compatibility"

    def test_required_top_level_keys(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        required = {
            "codebase_version", "robot_type", "total_episodes", "total_frames",
            "total_tasks", "chunks_size", "fps", "splits",
            "data_path", "video_path", "features",
            # v3-specific (added during v2.1 → v3.0 conversion):
            "data_files_size_in_mb", "video_files_size_in_mb",
        }
        missing = required - set(info.keys())
        assert not missing, f"info.json missing required v3 keys: {sorted(missing)}"

    def test_data_path_template_matches_upstream(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        assert info["data_path"] == DEFAULT_DATA_PATH

    def test_video_path_template_matches_upstream(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        assert info["video_path"] == DEFAULT_VIDEO_PATH

    def test_per_feature_fps_present_on_non_video(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        for fname, fdef in info["features"].items():
            if fdef.get("dtype") == "video":
                continue
            assert "fps" in fdef, \
                f"feature {fname!r} missing required `fps` (v3 requires fps on every non-video feature)"

    def test_size_fields_are_ints(self, written_v3: Path):
        info = json.loads((written_v3 / INFO_PATH).read_text())
        assert isinstance(info.get("data_files_size_in_mb"), int)
        assert isinstance(info.get("video_files_size_in_mb"), int)


# ---------------------------------------------------------------------------
# 2. tasks.parquet schema
# ---------------------------------------------------------------------------


class TestTasksParquet:
    def test_tasks_parquet_exists(self, written_v3: Path):
        assert (written_v3 / DEFAULT_TASKS_PATH).exists()

    def test_tasks_index_is_named_task(self, written_v3: Path):
        df = pq.read_table(written_v3 / DEFAULT_TASKS_PATH).to_pandas()
        # Upstream sets `tasks.index.name = "task"` then writes; pandas
        # preserves index in the parquet metadata.
        # Forge currently writes flat (task_index, task) columns — this fails.
        assert df.index.name == "task", (
            f"upstream io_utils.write_tasks sets `tasks.index.name = 'task'`. "
            f"Forge wrote {df.index.name!r} with columns={list(df.columns)}"
        )

    def test_tasks_has_task_index_column(self, written_v3: Path):
        df = pq.read_table(written_v3 / DEFAULT_TASKS_PATH).to_pandas()
        assert "task_index" in df.columns


# ---------------------------------------------------------------------------
# 3. episodes parquet schema (the big one)
# ---------------------------------------------------------------------------


class TestEpisodesParquet:
    def _load_first(self, written_v3: Path):
        episodes_dir = written_v3 / "meta" / "episodes"
        files = sorted(episodes_dir.rglob("*.parquet"))
        assert files, "no episodes parquet files written"
        return pq.read_table(files[0]).to_pandas()

    def test_has_episode_index_and_length(self, written_v3: Path):
        df = self._load_first(written_v3)
        assert {"episode_index", "length"}.issubset(df.columns)

    def test_has_tasks_list_column(self, written_v3: Path):
        df = self._load_first(written_v3)
        assert "tasks" in df.columns, (
            "v3 episodes.parquet must include `tasks` (list[str]) — Forge currently only "
            "writes `task_index`. Without `tasks` upstream loaders can't resolve language."
        )

    def test_has_data_chunk_pointers(self, written_v3: Path):
        df = self._load_first(written_v3)
        for col in ("data/chunk_index", "data/file_index"):
            assert col in df.columns, f"missing v3 episode pointer column: {col!r}"

    def test_has_dataset_row_range(self, written_v3: Path):
        df = self._load_first(written_v3)
        for col in ("dataset_from_index", "dataset_to_index"):
            assert col in df.columns, (
                f"missing {col!r} — without these the upstream reader can't slice "
                "episodes out of a multi-episode parquet chunk file"
            )

    def test_dataset_row_range_is_consistent(self, written_v3: Path):
        df = self._load_first(written_v3)
        if {"dataset_from_index", "dataset_to_index", "length"}.issubset(df.columns):
            for _, row in df.iterrows():
                spanned = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
                assert spanned == int(row["length"]), \
                    f"row range {spanned} != length {row['length']} for episode {row.get('episode_index')}"

    def test_has_per_camera_video_pointers(self, written_v3: Path):
        df = self._load_first(written_v3)
        # Find at least one video-key prefixed set of pointers.
        video_cols = [c for c in df.columns if c.startswith("videos/")]
        assert video_cols, "no `videos/<key>/...` columns in episodes.parquet"
        # For each unique video key, check the 4 required sub-columns.
        from collections import defaultdict

        per_key: dict[str, set[str]] = defaultdict(set)
        for col in video_cols:
            _, key, sub = col.split("/", 2)
            per_key[key].add(sub)
        required = {"chunk_index", "file_index", "from_timestamp", "to_timestamp"}
        for key, subs in per_key.items():
            missing = required - subs
            assert not missing, f"video key {key!r} missing sub-columns: {sorted(missing)}"

    def test_has_meta_episodes_pointers(self, written_v3: Path):
        df = self._load_first(written_v3)
        for col in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
            assert col in df.columns, f"missing {col!r}"

    def test_has_flattened_stats(self, written_v3: Path):
        df = self._load_first(written_v3)
        stats_cols = [c for c in df.columns if c.startswith("stats/")]
        assert stats_cols, (
            "no `stats/{feature}/{stat}` columns in episodes.parquet — v3 stores "
            "per-episode stats here (replacing v2.1's episodes_stats.jsonl)"
        )
        # For at least observation.state we expect min/max/mean/std/count.
        for stat in ("min", "max", "mean", "std", "count"):
            matching = [c for c in stats_cols if c.endswith(f"/{stat}")]
            assert matching, f"no `stats/.../{stat}` columns found"


# ---------------------------------------------------------------------------
# 4. meta/stats.json (aggregate)
# ---------------------------------------------------------------------------


class TestAggregateStats:
    def test_stats_json_exists(self, written_v3: Path):
        assert (written_v3 / STATS_PATH).exists(), (
            f"upstream v3 requires aggregate stats at {STATS_PATH} — Forge skips this entirely"
        )

    def test_stats_has_features(self, written_v3: Path):
        stats = json.loads((written_v3 / STATS_PATH).read_text())
        assert isinstance(stats, dict) and stats, "stats.json should be a non-empty dict"


# ---------------------------------------------------------------------------
# 5. THE GOLD-STANDARD TEST — does upstream's loader actually open it?
# ---------------------------------------------------------------------------


class TestUpstreamLoader:
    """If this passes, Forge-written v3 datasets are usable by anyone in the
    HuggingFace robotics ecosystem. If it fails, the failure message tells
    us exactly which invariant is violated."""

    def test_metadata_loads(self, written_v3: Path):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        # Local dataset — repo_id is a fiction here, root path overrides it.
        meta = LeRobotDatasetMetadata(
            repo_id="forge/synth_v3",
            root=written_v3,
        )
        assert meta.total_episodes == 3
        assert meta.fps == 30
        assert "observation.state" in meta.features
        assert "action" in meta.features

    def test_dataset_loads_and_iterates(self, written_v3: Path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(
            repo_id="forge/synth_v3",
            root=written_v3,
        )
        assert len(ds) == 3 * 30  # 3 episodes × 30 frames
        # First sample should have state + action tensors.
        sample = ds[0]
        assert "observation.state" in sample
        assert "action" in sample
