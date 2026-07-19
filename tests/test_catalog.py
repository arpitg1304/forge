"""Tests for the Forge catalog (Phase 1).

Covers schema validation, the manifest-last commit protocol and crash recovery,
hash-based idempotent re-ingest, ``v_latest_quality`` version selection, an
end-to-end ingest→query→stats integration on a synthetic dataset, and catalog
read/write over fsspec's ``memory://`` filesystem (the cloud stand-in).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# The catalog is an optional extra; skip the whole module if it's not installed.
pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("xxhash")

from forge.catalog import Catalog, CatalogError  # noqa: E402
from forge.catalog.schema import (  # noqa: E402
    EPISODES_SCHEMA,
    QUALITY_SCORES_SCHEMA,
    SCHEMA_VERSION,
    SCORER_VERSION,
    CatalogSchemaError,
    validate_rows,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ep_row(**over) -> dict:
    row = {
        "episode_id": "ep-1",
        "content_hash": "hash-1",
        "source_uri": "/data/ds",
        "source_format": "lerobot-v3",
        "robot": "franka",
        "task": "pick",
        "language_instruction": "pick up the cube",
        "operator_id": None,
        "scene_id": None,
        "num_frames": 100,
        "duration_s": 6.6,
        "fps": 15.0,
        "cameras": [{"name": "wrist", "width": 320, "height": 180, "fps": 15.0}],
        "state_dim": 7,
        "action_dim": 7,
        "ingest_batch_id": "batch-1",
        "ingested_at": _now(),
        "extra": {"source_episode_id": "0"},
    }
    row.update(over)
    return row


def _q_row(episode_id="ep-1", scorer_version=SCORER_VERSION, overall_score=8.0, **over) -> dict:
    row = {
        "episode_id": episode_id,
        "scorer_version": scorer_version,
        "overall_score": overall_score,
        "ldlj": 0.6,
        "sparc": -13.4,
        "dead_fraction": 0.02,
        "gripper_chatter_rate": 0.0,
        "joint_path_length": 1.2,
        "overall_saturation": 0.82,
        "mean_entropy": 0.69,
        "psd_high_fraction": 0.003,
        "state_action_consistency": 0.003,
        "static_fraction": 0.0,
        "jitter_ratio": 0.01,
        "flags": [],
        "computed_at": _now(),
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_valid_rows_build_table(self):
        tbl = validate_rows([_ep_row()], EPISODES_SCHEMA, table="episodes")
        assert tbl.num_rows == 1
        assert tbl.schema.equals(EPISODES_SCHEMA)

    def test_rejects_unexpected_column(self):
        with pytest.raises(CatalogSchemaError, match="unexpected column"):
            validate_rows([_ep_row(bogus="x")], EPISODES_SCHEMA, table="episodes")

    def test_rejects_missing_required(self):
        row = _ep_row()
        del row["source_uri"]
        with pytest.raises(CatalogSchemaError, match="missing required"):
            validate_rows([row], EPISODES_SCHEMA, table="episodes")

    def test_rejects_null_in_required(self):
        with pytest.raises(CatalogSchemaError, match="missing required"):
            validate_rows([_ep_row(num_frames=None)], EPISODES_SCHEMA, table="episodes")

    def test_rejects_bad_type(self):
        with pytest.raises(CatalogSchemaError):
            validate_rows(
                [_ep_row(num_frames="not-an-int")], EPISODES_SCHEMA, table="episodes"
            )

    def test_quality_schema_mirrors_metric_names(self):
        # Guardrail: the schema columns must match EpisodeQuality's flat metrics.
        names = set(QUALITY_SCORES_SCHEMA.names)
        for metric in ("ldlj", "sparc", "dead_fraction", "static_fraction", "jitter_ratio"):
            assert metric in names


# ---------------------------------------------------------------------------
# Lifecycle, commit & read
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_init_creates_catalog_json(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "cat"))
        assert cat.schema_version == SCHEMA_VERSION
        assert (tmp_path / "cat" / "catalog.json").exists()

    def test_open_missing_raises(self, tmp_path: Path):
        with pytest.raises(CatalogError, match="No catalog"):
            Catalog.open(str(tmp_path / "nope"))

    def test_init_refuses_existing(self, tmp_path: Path):
        root = str(tmp_path / "cat")
        Catalog.init(root)
        with pytest.raises(CatalogError, match="already exists"):
            Catalog.init(root)
        # exist_ok reuses it
        assert Catalog.init(root, exist_ok=True).schema_version == SCHEMA_VERSION

    def test_refuses_newer_schema_version(self, tmp_path: Path):
        import json

        root = tmp_path / "cat"
        root.mkdir()
        (root / "catalog.json").write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(CatalogError, match="schema_version 999"):
            Catalog.open(str(root))

    def test_empty_catalog_queries_and_stats(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "cat"))
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 0
        assert cat.stats()["episodes"] == 0
        # views exist and are joinable even when empty
        assert cat.sql("SELECT count(*) n FROM v_latest_quality").to_pylist()[0]["n"] == 0


class TestCommitAndRead:
    def test_commit_and_query(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "cat"))
        cat.commit_batch(episodes=[_ep_row()], quality_scores=[_q_row()], batch_id="b1")

        rows = cat.sql(
            "SELECT e.robot, e.task, q.overall_score "
            "FROM episodes e JOIN v_latest_quality q USING(episode_id)"
        ).to_pylist()
        assert rows == [{"robot": "franka", "task": "pick", "overall_score": pytest.approx(8.0)}]

    def test_v_latest_quality_picks_newest_version(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "cat"))
        cat.commit_batch(
            episodes=[_ep_row()],
            quality_scores=[_q_row(scorer_version="q-v1", overall_score=6.0)],
            batch_id="b1",
        )
        # Re-score with a newer version: both rows persist, latest wins.
        cat.add_quality_scores(
            [_q_row(scorer_version="q-v2", overall_score=9.0)], batch_id="b2"
        )
        assert cat.sql("SELECT count(*) n FROM quality_scores").to_pylist()[0]["n"] == 2
        latest = cat.sql(
            "SELECT scorer_version, overall_score FROM v_latest_quality"
        ).to_pylist()
        assert latest == [{"scorer_version": "q-v2", "overall_score": pytest.approx(9.0)}]

    def test_stats(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "cat"))
        cat.commit_batch(
            episodes=[
                _ep_row(episode_id="a", content_hash="h-a", task="pick", robot="franka", num_frames=100),
                _ep_row(episode_id="b", content_hash="h-b", task="place", robot="franka", num_frames=200),
            ],
            quality_scores=[
                _q_row(episode_id="a", overall_score=6.0),
                _q_row(episode_id="b", overall_score=8.0),
            ],
            batch_id="b1",
        )
        s = cat.stats()
        assert s["episodes"] == 2
        assert s["total_frames"] == 300
        assert {r["task"] for r in s["per_task"]} == {"pick", "place"}
        assert s["overall_score"]["mean"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Commit protocol & crash recovery
# ---------------------------------------------------------------------------


class TestCommitProtocol:
    def test_crash_before_manifest_leaves_no_visible_rows(self, tmp_path: Path, monkeypatch):
        cat = Catalog.init(str(tmp_path / "cat"))
        writer = cat._writer
        real = writer._write_json_atomic

        def boom(path, payload):
            if "manifest-" in path:
                raise RuntimeError("simulated crash before manifest")
            return real(path, payload)

        monkeypatch.setattr(writer, "_write_json_atomic", boom)
        with pytest.raises(RuntimeError):
            cat.commit_batch(
                episodes=[_ep_row()], quality_scores=[_q_row()], batch_id="crashed"
            )

        # Part-files exist on disk but no manifest committed them.
        assert cat.orphan_part_files(), "expected orphan part-files from the crash"
        # ...so they are invisible to queries (atomic: no partial rows).
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 0

    def test_recovers_after_crash(self, tmp_path: Path, monkeypatch):
        cat = Catalog.init(str(tmp_path / "cat"))
        writer = cat._writer
        real = writer._write_json_atomic

        def boom(path, payload):
            if "manifest-" in path:
                raise RuntimeError("crash")
            return real(path, payload)

        monkeypatch.setattr(writer, "_write_json_atomic", boom)
        with pytest.raises(RuntimeError):
            cat.commit_batch(episodes=[_ep_row()], quality_scores=[_q_row()], batch_id="crashed")

        # Restore and re-commit properly — the retry succeeds and is visible.
        monkeypatch.setattr(writer, "_write_json_atomic", real)
        cat.commit_batch(episodes=[_ep_row()], quality_scores=[_q_row()], batch_id="retry")
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 1


# ---------------------------------------------------------------------------
# Ingest integration (synthetic zarr dataset)
# ---------------------------------------------------------------------------


def _make_zarr_dataset(path: Path, *, episodes: int = 2, frames: int = 20) -> Path:
    import numpy as np

    zarr = pytest.importorskip("zarr")
    root = zarr.open(str(path), mode="w")
    root.attrs["fps"] = 10
    total = episodes * frames

    data = root.create_group("data")
    dmk = getattr(data, "create_array", None) or data.create_dataset
    action = dmk("action", shape=(total, 7), dtype="float32")
    action[:] = np.random.RandomState(0).randn(total, 7).astype("float32")
    state = dmk("robot_state", shape=(total, 7), dtype="float32")
    state[:] = np.random.RandomState(1).randn(total, 7).astype("float32")

    meta = root.create_group("meta")
    mmk = getattr(meta, "create_array", None) or meta.create_dataset
    ends = mmk("episode_ends", shape=(episodes,), dtype="int64")
    ends[:] = np.array([(i + 1) * frames for i in range(episodes)], dtype="int64")
    return path


class TestIngest:
    def test_ingest_registers_and_scores(self, tmp_path: Path):
        from forge.catalog.ingest import ingest

        ds = _make_zarr_dataset(tmp_path / "ds.zarr", episodes=3, frames=15)
        cat = Catalog.init(str(tmp_path / "cat"))
        stats = ingest([str(ds)], cat)

        assert stats.ingested == 3
        assert stats.skipped == 0
        assert stats.frames == 45
        # Every episode got a quality score row.
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 3
        assert cat.sql("SELECT count(*) n FROM v_latest_quality").to_pylist()[0]["n"] == 3
        scored = cat.sql(
            "SELECT count(*) n FROM v_latest_quality WHERE overall_score IS NOT NULL"
        ).to_pylist()[0]["n"]
        assert scored == 3
        # source_format was detected and recorded.
        fmts = cat.sql("SELECT DISTINCT source_format FROM episodes").to_pylist()
        assert fmts == [{"source_format": "zarr"}]

    def test_idempotent_reingest(self, tmp_path: Path):
        from forge.catalog.ingest import ingest

        ds = _make_zarr_dataset(tmp_path / "ds.zarr", episodes=2, frames=10)
        cat = Catalog.init(str(tmp_path / "cat"))

        first = ingest([str(ds)], cat)
        assert first.ingested == 2

        second = ingest([str(ds)], cat)
        assert second.ingested == 0
        assert second.skipped == 2
        # No duplicate rows created.
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 2


# ---------------------------------------------------------------------------
# Cloud (fsspec memory://)
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_memory_fs():
    import fsspec

    fs = fsspec.filesystem("memory")
    try:
        for p in fs.find("/"):
            fs.rm(p)
    except FileNotFoundError:
        pass
    yield fs


class TestCloudMemory:
    def test_catalog_over_memory(self, clean_memory_fs):
        cat = Catalog.init("memory://labcat")
        cat.commit_batch(episodes=[_ep_row()], quality_scores=[_q_row()], batch_id="b1")

        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 1
        # Reopen from a fresh handle — state is durable on the (memory) fs.
        reopened = Catalog.open("memory://labcat")
        assert reopened.stats()["episodes"] == 1

    def test_content_hash_dedup_over_memory(self, clean_memory_fs):
        cat = Catalog.init("memory://labcat2")
        cat.commit_batch(episodes=[_ep_row(content_hash="dup")], batch_id="b1")
        assert cat.episode_hashes() == {"dup"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCatalogCLI:
    def test_init_ingest_query_stats(self, tmp_path: Path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        ds = _make_zarr_dataset(tmp_path / "ds.zarr", episodes=2, frames=12)
        cat_root = str(tmp_path / "cat")

        r = runner.invoke(app, ["catalog", "init", cat_root])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["ingest", str(ds), "--catalog", cat_root])
        assert r.exit_code == 0, r.output
        assert "2 ingested" in r.output

        r = runner.invoke(
            app,
            ["query", "SELECT count(*) n FROM episodes", "-c", cat_root, "-f", "json"],
        )
        assert r.exit_code == 0, r.output
        assert '"n": 2' in r.output

        r = runner.invoke(app, ["catalog", "stats", "--catalog", cat_root])
        assert r.exit_code == 0, r.output
        assert "Episodes: 2" in r.output

    def test_query_missing_catalog_errors_cleanly(self, tmp_path: Path):
        from typer.testing import CliRunner

        from forge.cli import app

        r = CliRunner().invoke(
            app, ["query", "SELECT 1", "-c", str(tmp_path / "nope")]
        )
        assert r.exit_code == 1
        assert "No catalog" in r.output
