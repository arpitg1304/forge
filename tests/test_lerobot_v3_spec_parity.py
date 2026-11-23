"""Parity with LeRobot's own writer: the same episodes written by
``lerobot.datasets.LeRobotDataset.create`` and by Forge must produce the same
artifacts.

Loading is a weak test (``test_lerobot_v3_upstream_validity.py`` covers it):
LeRobot opens datasets that are missing joint names, quantile stats or image
stats, and the gaps only show up later, as a ValueError when pi0.5 starts
training, as unnormalized images in ACT, or as a refused merge. This module
compares artifact by artifact instead:

- info.json features, compared with LeRobot's own merge rule
- meta/tasks.parquet
- meta/episodes rows: every column, every value except layout pointers
- meta/stats.json and per-episode stats, including quantiles and image stats
- data parquet columns, Arrow types and values

and then checks the two downstream uses that break on a mismatch:
``aggregate_datasets`` with a LeRobot-recorded dataset, and pi0.5's
quantile normalizer.

Known, deliberate difference: Forge writes one data/video chunk per episode;
LeRobot packs episodes into shared files. Pointer columns are excluded below.

Skipped if ``lerobot`` isn't installed. Checked with lerobot 0.5.1 and 0.7.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="`lerobot` not installed")

from importlib.metadata import version as _pkg_version  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from packaging.version import Version  # noqa: E402

from forge.core.models import CameraInfo, Episode, Frame, LazyImage  # noqa: E402
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig  # noqa: E402

# lerobot < 0.7 computed different stats: image std overflowed in uint8
# (always 0) and dataset quantiles were count-weighted means. Forge follows
# 0.7's corrected algorithm, so stats are compared only from 0.7 on.
_STATS_COMPARABLE = Version(_pkg_version("lerobot")) >= Version("0.7.0")
needs_lerobot_07_stats = pytest.mark.skipif(
    not _STATS_COMPARABLE, reason="lerobot < 0.7 stats differ by design (uint8 overflow fix, quantile envelope)"
)

N_EP, N_FR, D, H, W, FPS = 3, 30, 6, 32, 32, 30
NAMES = [f"joint_{i}" for i in range(D)]
TASKS = ["pick the cube", "place the cube", "pick the cube"]
VIDEO = "observation.images.top"

# Episode columns that point into the file layout, which differs on purpose.
_LAYOUT_COLUMNS = {
    "data/chunk_index",
    "data/file_index",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
    f"videos/{VIDEO}/chunk_index",
    f"videos/{VIDEO}/file_index",
    f"videos/{VIDEO}/from_timestamp",
    f"videos/{VIDEO}/to_timestamp",
}


def _image(ep: int, i: int) -> np.ndarray:
    ramp = np.add.outer(np.arange(H), np.arange(W))[..., None] * 4
    return np.clip(ramp + np.array([0, 60, 120]) + 3 * i + 40 * ep, 0, 255).astype(np.uint8)


def _episodes_data() -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "state": rng.standard_normal((N_FR, D)).astype(np.float32),
            "action": rng.standard_normal((N_FR, D)).astype(np.float32),
            "img": np.stack([_image(e, i) for i in range(N_FR)]),
            "task": TASKS[e],
        }
        for e in range(N_EP)
    ]


def _write_upstream(root: Path, data: list[dict], repo_id: str = "parity/upstream") -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (D,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (D,), "names": NAMES},
        VIDEO: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(repo_id, fps=FPS, features=features, root=root, robot_type="synth")
    for ep in data:
        for i in range(N_FR):
            ds.add_frame(
                {
                    "observation.state": ep["state"][i],
                    "action": ep["action"][i],
                    VIDEO: ep["img"][i],
                    "task": ep["task"],
                }
            )
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    return root


def _forge_episode(e: int, ep: dict) -> Episode:
    def gen():
        for i in range(N_FR):
            yield Frame(
                index=i,
                timestamp=i / FPS,
                state=ep["state"][i],
                action=ep["action"][i],
                images={"top": LazyImage(loader=lambda im=ep["img"][i]: im, height=H, width=W, channels=3)},
                is_first=i == 0,
                is_last=i == N_FR - 1,
            )

    return Episode(
        episode_id=f"ep{e}",
        cameras={"top": CameraInfo("top", H, W, 3)},
        state_dim=D,
        action_dim=D,
        fps=FPS,
        language_instruction=ep["task"],
        state_names=NAMES,
        action_names=NAMES,
        _frame_loader=gen,
    )


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("parity")
    data = _episodes_data()
    up = _write_upstream(base / "upstream", data)
    fg = base / "forge"
    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=FPS, robot_type="synth"))
    writer.write_dataset((_forge_episode(e, ep) for e, ep in enumerate(data)), fg)
    return up, fg


def _read_dir(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _assert_close(a, b, what: str, rtol=1e-5, atol=1e-6) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{what}: shape {a.shape} (upstream) != {b.shape} (forge)"
    np.testing.assert_allclose(b, a, rtol=rtol, atol=atol, err_msg=what)


class TestInfo:
    def test_features_mergeable_by_lerobot_rule(self, pair):
        """LeRobot's aggregate_datasets refuses to merge unless this holds."""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        try:
            from lerobot.datasets.feature_utils import features_equal_for_merge
        except ImportError:  # lerobot < 0.7 compares with ==, minus video info
            features_equal_for_merge = None

        up, fg = pair
        fu = LeRobotDatasetMetadata("parity/upstream", root=up).features
        ff = LeRobotDatasetMetadata("parity/forge", root=fg).features
        if features_equal_for_merge is not None:
            assert features_equal_for_merge(fu, ff), f"\nupstream={fu}\nforge={ff}"
        else:
            strip = lambda f: {k: {kk: vv for kk, vv in v.items() if kk != "info"} for k, v in f.items()}  # noqa: E731
            assert strip(fu) == strip(ff)

    def test_joint_names_written(self, pair):
        _, fg = pair
        features = json.loads((fg / "meta/info.json").read_text())["features"]
        assert features["observation.state"]["names"] == NAMES
        assert features["action"]["names"] == NAMES

    def test_top_level_keys_match(self, pair):
        up, fg = pair
        iu = json.loads((up / "meta/info.json").read_text())
        ifg = json.loads((fg / "meta/info.json").read_text())
        assert set(iu) == set(ifg)
        for key in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks",
                    "chunks_size", "fps", "splits", "data_path", "video_path"):
            assert ifg[key] == iu[key], key


class TestTasks:
    def test_tasks_parquet_equal(self, pair):
        up, fg = pair
        pd.testing.assert_frame_equal(
            pd.read_parquet(fg / "meta/tasks.parquet"), pd.read_parquet(up / "meta/tasks.parquet")
        )


class TestEpisodes:
    def test_same_columns(self, pair):
        up, fg = pair
        eu, ef = _read_dir(up / "meta/episodes"), _read_dir(fg / "meta/episodes")
        assert sorted(ef.columns) == sorted(eu.columns)

    def test_same_values(self, pair):
        up, fg = pair
        eu, ef = _read_dir(up / "meta/episodes"), _read_dir(fg / "meta/episodes")
        assert len(eu) == len(ef) == N_EP
        columns = set(eu.columns) - _LAYOUT_COLUMNS
        if not _STATS_COMPARABLE:
            columns = {c for c in columns if not c.startswith("stats/")}
        for col in sorted(columns):
            for r in range(N_EP):
                a, b = eu[col].iloc[r], ef[col].iloc[r]
                if col == "tasks":
                    assert list(b) == list(a)
                    continue
                # Image stats come from lossless source frames on both sides.
                _assert_close(np.stack(a) if np.asarray(a).dtype == object else a,
                              np.stack(b) if np.asarray(b).dtype == object else b,
                              f"episode {r} {col}", rtol=1e-4, atol=1e-5)

    def test_video_spans_cover_episode(self, pair):
        _, fg = pair
        ef = _read_dir(fg / "meta/episodes")
        for _, row in ef.iterrows():
            span = row[f"videos/{VIDEO}/to_timestamp"] - row[f"videos/{VIDEO}/from_timestamp"]
            assert span == pytest.approx(row["length"] / FPS)


class TestStats:
    @needs_lerobot_07_stats
    def test_aggregate_stats_match(self, pair):
        up, fg = pair
        su = json.loads((up / "meta/stats.json").read_text())
        sf = json.loads((fg / "meta/stats.json").read_text())
        assert sorted(sf) == sorted(su)
        for feat in su:
            assert sorted(sf[feat]) == sorted(su[feat]), feat
            for stat in su[feat]:
                _assert_close(su[feat][stat], sf[feat][stat], f"{feat}/{stat}", rtol=1e-4, atol=1e-5)


class TestData:
    def test_arrow_types_match(self, pair):
        up, fg = pair
        su = pq.read_schema(next((up / "data").rglob("*.parquet")))
        sf = pq.read_schema(next((fg / "data").rglob("*.parquet")))
        assert sorted(sf.names) == sorted(su.names)
        for name in su.names:
            assert sf.field(name).type == su.field(name).type, name

    def test_values_match(self, pair):
        up, fg = pair
        du, df = _read_dir(up / "data"), _read_dir(fg / "data")
        for col in du.columns:
            _assert_close(np.stack(du[col].map(np.asarray)), np.stack(df[col].map(np.asarray)), col)


class TestDownstream:
    def test_aggregates_with_lerobot_recording(self, pair, tmp_path):
        """Merging a Forge conversion with a natively recorded dataset of the
        same robot is the everyday reason feature dicts must match."""
        from lerobot.datasets.aggregate import aggregate_datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        up, fg = pair
        out = tmp_path / "merged"
        aggregate_datasets(
            repo_ids=["parity/upstream", "parity/forge"],
            aggr_repo_id="parity/merged",
            roots=[up, fg],
            aggr_root=out,
        )
        merged = LeRobotDataset("parity/merged", root=out)
        assert merged.meta.total_episodes == 2 * N_EP
        assert len(merged) == 2 * N_EP * N_FR
        assert merged[len(merged) - 1]["task"] == TASKS[-1]

    def test_pi05_quantile_normalizer_builds(self, pair):
        """pi0.5 normalizes state/action by q01/q99 and raises without them."""
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        try:
            from lerobot.utils.feature_utils import dataset_to_policy_features
        except ImportError:
            from lerobot.datasets.feature_utils import dataset_to_policy_features

        _, fg = pair
        ds = LeRobotDataset("parity/forge", root=fg)
        step = NormalizerProcessorStep.from_lerobot_dataset(
            ds,
            dataset_to_policy_features(ds.meta.features),
            {
                FeatureType.VISUAL: NormalizationMode.MEAN_STD,
                FeatureType.STATE: NormalizationMode.QUANTILES,
                FeatureType.ACTION: NormalizationMode.QUANTILES,
            },
        )
        sample = ds[0]
        out = step._apply_transform(sample["observation.state"], "observation.state", FeatureType.STATE)
        assert torch.isfinite(out).all()
        # Without image stats the normalizer silently skips the camera.
        assert VIDEO in step._tensor_stats
        assert tuple(step._tensor_stats[VIDEO]["mean"].shape) == (3, 1, 1)
