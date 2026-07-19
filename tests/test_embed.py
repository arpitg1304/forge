"""Tests for embeddings + semantic search (Phase 2).

The whole pipeline is exercised with a deterministic **fake** embedding model
(no torch, no network, no GPU), so these run anywhere. A single guarded test at
the end exercises the real SigLIP model when torch/transformers are installed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("xxhash")

from forge.catalog import Catalog, CatalogError  # noqa: E402
from forge.catalog.schema import EMBEDDINGS_SCHEMA, CatalogSchemaError, validate_rows  # noqa: E402
from forge.embed.base import EmbeddingModel, l2_normalize  # noqa: E402
from forge.embed.registry import get_model, register_model  # noqa: E402
from forge.embed.sampling import pool, sample_frame_indices  # noqa: E402

# --------------------------------------------------------------------------
# A deterministic, torch-free model registered for the test session.
# --------------------------------------------------------------------------


class FakeModel(EmbeddingModel):
    name = "fake"
    supports_text = True

    @property
    def dim(self) -> int:
        return 8

    @property
    def checkpoint_hash(self) -> str:
        return "test"

    def _hv(self, s: str) -> np.ndarray:
        return np.frombuffer(
            hashlib.sha256(s.encode()).digest()[:32], dtype=np.uint32
        ).astype(np.float32)

    def embed_images(self, images):
        return l2_normalize(
            np.stack([self._hv(f"img{int(np.asarray(im).sum())}") for im in images])
        )

    def embed_text(self, texts):
        return l2_normalize(np.stack([self._hv(t) for t in texts]))


register_model("fake", lambda **kwargs: FakeModel())
_FAKE_ID = FakeModel().model_id  # "fake@test"


def _now():
    return datetime.now(timezone.utc)


def _ep_row(i: int, instr: str | None = None, **over) -> dict:
    row = {
        "episode_id": f"e{i}",
        "content_hash": f"h{i}",
        "source_uri": "/ds",
        "source_format": "zarr",
        "robot": "r",
        "task": None,
        "language_instruction": instr,
        "operator_id": None,
        "scene_id": None,
        "num_frames": 10,
        "duration_s": 1.0,
        "fps": 10.0,
        "cameras": [],
        "state_dim": 7,
        "action_dim": 7,
        "ingest_batch_id": "b",
        "ingested_at": _now(),
        "extra": {},
    }
    row.update(over)
    return row


def _emb_row(episode_id, vector, *, model_id=_FAKE_ID, level="instruction", camera=None, **over):
    row = {
        "episode_id": episode_id,
        "model_id": model_id,
        "level": level,
        "camera": camera,
        "pooling": "none" if level == "instruction" else "mean",
        "dim": len(vector),
        "vector": list(vector),
        "num_frames_sampled": None,
        "computed_at": _now(),
    }
    row.update(over)
    return row


# --------------------------------------------------------------------------
# Engine units
# --------------------------------------------------------------------------


class TestEngine:
    def test_registry_and_model_id(self):
        m = get_model("fake")
        assert m.model_id == "fake@test"
        assert m.dim == 8 and m.supports_text

    def test_unknown_model_raises(self):
        from forge.embed.base import ModelError

        with pytest.raises(ModelError, match="unknown embedding model"):
            get_model("does-not-exist")

    def test_l2_normalize(self):
        v = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.isclose(np.linalg.norm(v[0]), 1.0)

    def test_sampling_stride(self):
        # 100 frames @ 10 fps, 1 Hz -> every 10th frame
        assert sample_frame_indices(100, 10.0, target_hz=1.0)[:3] == [0, 10, 20]
        # unknown fps -> falls back but stays capped
        assert 0 < len(sample_frame_indices(500, None)) <= 32
        assert sample_frame_indices(0, 10.0) == []

    def test_pooling(self):
        v = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
        assert np.allclose(pool(v, "mean"), [1.0, 2.0])
        assert np.allclose(pool(v, "first-mid-last"), [1.0, 2.0])
        with pytest.raises(ValueError):
            pool(v, "bogus")


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


class TestEmbeddingsSchema:
    def test_valid_row(self):
        tbl = validate_rows(
            [_emb_row("e0", [0.1] * 8)], EMBEDDINGS_SCHEMA, table="embeddings"
        )
        assert tbl.num_rows == 1

    def test_rejects_missing_required(self):
        row = _emb_row("e0", [0.1] * 8)
        del row["model_id"]
        with pytest.raises(CatalogSchemaError, match="missing required"):
            validate_rows([row], EMBEDDINGS_SCHEMA, table="embeddings")


# --------------------------------------------------------------------------
# Search (add_embeddings + Catalog.search)
# --------------------------------------------------------------------------


@pytest.fixture
def instr_catalog(tmp_path: Path) -> Catalog:
    """A catalog with 3 episodes + their instruction embeddings (fake model)."""
    cat = Catalog.init(str(tmp_path / "cat"))
    tasks = ["pick up the red cup", "open the drawer", "fold the towel"]
    cat.commit_batch(
        episodes=[_ep_row(i, t) for i, t in enumerate(tasks)], batch_id="b1"
    )
    m = FakeModel()
    cat.add_embeddings(
        [_emb_row(f"e{i}", m.embed_text([t])[0].tolist()) for i, t in enumerate(tasks)],
        batch_id="b2",
    )
    return cat


class TestSearch:
    def test_text_search_ranks_exact_match_first(self, instr_catalog):
        res = instr_catalog.search("open the drawer", level="instruction").to_pylist()
        assert res[0]["episode_id"] == "e1"
        assert res[0]["score"] > 0.999

    def test_like_search(self, instr_catalog):
        res = instr_catalog.search(like="e0", level="instruction", top=2).to_pylist()
        assert res[0]["episode_id"] == "e0"

    def test_top_k_limits(self, instr_catalog):
        assert len(instr_catalog.search("x", level="instruction", top=2).to_pylist()) == 2

    def test_requires_query_or_like(self, instr_catalog):
        with pytest.raises(ValueError):
            instr_catalog.search()
        with pytest.raises(ValueError):
            instr_catalog.search("x", like="e0")

    def test_no_embeddings_errors(self, tmp_path: Path):
        cat = Catalog.init(str(tmp_path / "empty"))
        with pytest.raises(CatalogError, match="No embeddings"):
            cat.search(like="e0")

    def test_single_model_enforced(self, instr_catalog):
        instr_catalog.add_embeddings(
            [_emb_row("e0", [0.1] * 8, model_id="other@x")], batch_id="b3"
        )
        with pytest.raises(CatalogError, match="Multiple embedding models"):
            instr_catalog.search("x", level="instruction")
        # explicit model_id resolves the ambiguity
        res = instr_catalog.search(
            "open the drawer", level="instruction", model_id=_FAKE_ID
        ).to_pylist()
        assert res[0]["episode_id"] == "e1"

    def test_embedded_episode_ids(self, instr_catalog):
        assert instr_catalog.embedded_episode_ids(_FAKE_ID) == {"e0", "e1", "e2"}
        assert instr_catalog.embedding_model_ids() == [_FAKE_ID]


# --------------------------------------------------------------------------
# Ingest -> embed -> search integration (synthetic zarr with a camera)
# --------------------------------------------------------------------------


def _make_zarr_with_camera(path: Path, *, episodes=3, frames=10) -> Path:
    zarr = pytest.importorskip("zarr")
    root = zarr.open(str(path), mode="w")
    root.attrs["fps"] = 10
    total = episodes * frames
    data = root.create_group("data")
    dmk = getattr(data, "create_array", None) or data.create_dataset
    cam = dmk("camera0_rgb", shape=(total, 16, 16, 3), dtype="uint8")
    cam[:] = np.random.RandomState(0).randint(0, 255, (total, 16, 16, 3), dtype="uint8")
    act = dmk("action", shape=(total, 7), dtype="float32")
    act[:] = np.random.RandomState(1).randn(total, 7).astype("float32")
    st = dmk("robot_state", shape=(total, 7), dtype="float32")
    st[:] = np.random.RandomState(2).randn(total, 7).astype("float32")
    meta = root.create_group("meta")
    mmk = getattr(meta, "create_array", None) or meta.create_dataset
    ends = mmk("episode_ends", shape=(episodes,), dtype="int64")
    ends[:] = np.array([(i + 1) * frames for i in range(episodes)], dtype="int64")
    return path


class TestEmbedPipeline:
    def test_ingest_embed_search(self, tmp_path: Path):
        from forge.catalog.embed import embed_catalog
        from forge.catalog.ingest import ingest

        ds = _make_zarr_with_camera(tmp_path / "ds.zarr", episodes=3, frames=10)
        cat = Catalog.init(str(tmp_path / "cat"))
        ingest([str(ds)], cat)

        stats = embed_catalog(cat, model_name="fake")
        assert stats.embedded == 3
        assert stats.vision_rows == 3  # one camera per episode
        assert cat.sql(
            "SELECT count(*) n FROM embeddings WHERE level='episode'"
        ).to_pylist()[0]["n"] == 3

        # idempotent re-embed
        again = embed_catalog(cat, model_name="fake")
        assert again.embedded == 0 and again.skipped == 3

        # --like search returns the queried episode first
        eid = cat.sql("SELECT episode_id FROM episodes LIMIT 1").to_pylist()[0]["episode_id"]
        res = cat.search(like=eid, level="episode", top=3).to_pylist()
        assert res[0]["episode_id"] == eid and res[0]["score"] > 0.999


# --------------------------------------------------------------------------
# Cloud (memory://) and CLI
# --------------------------------------------------------------------------


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


class TestCloudAndCLI:
    def test_embeddings_over_memory(self, clean_memory_fs):
        cat = Catalog.init("memory://ecat")
        cat.commit_batch(episodes=[_ep_row(0, "open the drawer")], batch_id="b1")
        cat.add_embeddings(
            [_emb_row("e0", FakeModel().embed_text(["open the drawer"])[0].tolist())],
            batch_id="b2",
        )
        res = Catalog.open("memory://ecat").search(
            "open the drawer", level="instruction"
        ).to_pylist()
        assert res[0]["episode_id"] == "e0"

    def test_cli_embed_and_search(self, tmp_path: Path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        ds = _make_zarr_with_camera(tmp_path / "ds.zarr", episodes=2, frames=10)
        cat_root = str(tmp_path / "cat")

        assert runner.invoke(app, ["catalog", "init", cat_root]).exit_code == 0
        assert runner.invoke(app, ["ingest", str(ds), "-c", cat_root]).exit_code == 0

        r = runner.invoke(app, ["embed", "-c", cat_root, "--model", "fake"])
        assert r.exit_code == 0, r.output
        assert "embedded" in r.output

        eid = Catalog.open(cat_root).sql(
            "SELECT episode_id FROM episodes LIMIT 1"
        ).to_pylist()[0]["episode_id"]
        r = runner.invoke(
            app, ["search", "-c", cat_root, "--like", eid, "--level", "episode", "-f", "json"]
        )
        assert r.exit_code == 0, r.output
        assert eid in r.output


# --------------------------------------------------------------------------
# Real SigLIP smoke test — skipped unless torch+transformers are installed.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_real_siglip_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import os

    if os.environ.get("FORGE_RUN_SIGLIP") != "1":
        pytest.skip("set FORGE_RUN_SIGLIP=1 to run the ~1.5GB SigLIP download test")

    model = get_model("siglip-so400m", device="cpu")
    assert model.dim == 1152
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    v = model.embed_images([img])
    assert v.shape == (1, 1152) and np.isfinite(v).all()
    t = model.embed_text(["a robot picks up a cup"])
    assert t.shape == (1, 1152) and np.isfinite(t).all()
