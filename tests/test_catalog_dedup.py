"""Tests for catalog dedup + curation + studio (Phase 3).

Uses hand-built embedding vectors (no torch) so two episodes are near-identical
and reliably form a near-dup pair.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from forge.catalog import Catalog  # noqa: E402
from forge.catalog.dedup import compute_dedup_edges, curate  # noqa: E402
from forge.catalog.schema import (  # noqa: E402
    CURATION_LABELS_SCHEMA,
    DEDUP_EDGES_SCHEMA,
    CatalogSchemaError,
    validate_rows,
)

_MODEL = "fake@t"


def _now():
    return datetime.now(timezone.utc)


def _unit(vec) -> list:
    v = np.asarray(vec, dtype=np.float32)
    return (v / (np.linalg.norm(v) or 1.0)).tolist()


def _ep(i, score, **over) -> dict:
    row = {
        "episode_id": f"e{i}", "content_hash": f"h{i}", "source_uri": "/ds",
        "source_format": "zarr", "robot": "franka", "task": "pick",
        "language_instruction": f"task {i}", "operator_id": None, "scene_id": None,
        "num_frames": 100 + i, "duration_s": 5.0, "fps": 10.0, "cameras": [],
        "state_dim": 7, "action_dim": 7, "ingest_batch_id": "b", "ingested_at": _now(),
        "extra": {},
    }
    row.update(over)
    return {"ep": row, "score": score}


def _emb(eid, vec, camera="cam0") -> dict:
    return {
        "episode_id": eid, "model_id": _MODEL, "level": "episode", "camera": camera,
        "pooling": "mean", "dim": len(vec), "vector": list(vec),
        "num_frames_sampled": 4, "computed_at": _now(),
    }


@pytest.fixture
def dup_catalog(tmp_path: Path) -> Catalog:
    """Catalog where e0 and e1 are near-identical (dup); e2 is distinct."""
    cat = Catalog.init(str(tmp_path / "cat"))
    specs = [_ep(0, 8.0), _ep(1, 6.0), _ep(2, 7.0)]
    cat.commit_batch(
        episodes=[s["ep"] for s in specs],
        quality_scores=[
            {"episode_id": s["ep"]["episode_id"], "scorer_version": "q-v1",
             "overall_score": s["score"], "flags": [], "computed_at": _now()}
            for s in specs
        ],
        batch_id="b1",
    )
    # e0 ~ e1 (cosine ~1); e2 orthogonal-ish
    cat.add_embeddings([
        _emb("e0", _unit([1.0, 0.0, 0.0, 0.02])),
        _emb("e1", _unit([1.0, 0.01, 0.0, 0.0])),
        _emb("e2", _unit([0.0, 0.0, 1.0, 0.0])),
    ], batch_id="b2")
    return cat


class TestSchemas:
    def test_dedup_edge_valid_and_required(self):
        row = {
            "episode_a": "a", "episode_b": "b", "similarity": 0.98,
            "model_id": _MODEL, "method": "ann-cosine", "computed_at": _now(),
        }
        assert validate_rows([row], DEDUP_EDGES_SCHEMA, table="dedup_edges").num_rows == 1
        del row["similarity"]
        with pytest.raises(CatalogSchemaError, match="missing required"):
            validate_rows([row], DEDUP_EDGES_SCHEMA, table="dedup_edges")

    def test_curation_label_valid(self):
        row = {"episode_id": "a", "label": "approved", "reason": None,
               "labeled_by": "user:x", "labeled_at": _now()}
        assert validate_rows([row], CURATION_LABELS_SCHEMA, table="curation_labels").num_rows == 1


class TestDedup:
    def test_finds_pair_canonical_order(self, dup_catalog):
        st = compute_dedup_edges(dup_catalog, threshold=0.97)
        assert st.pairs_found == 1 and st.pairs_added == 1
        edge = dup_catalog.sql("SELECT episode_a, episode_b FROM dedup_edges").to_pylist()[0]
        assert (edge["episode_a"], edge["episode_b"]) == ("e0", "e1")  # a < b

    def test_threshold_excludes_distinct(self, dup_catalog):
        # High threshold: only the true dup; e2 never pairs.
        compute_dedup_edges(dup_catalog, threshold=0.97)
        eps = dup_catalog.sql(
            "SELECT DISTINCT episode_a FROM dedup_edges "
            "UNION SELECT DISTINCT episode_b FROM dedup_edges"
        ).to_pylist()
        ids = {r["episode_a"] for r in eps}
        assert "e2" not in ids

    def test_idempotent(self, dup_catalog):
        compute_dedup_edges(dup_catalog, threshold=0.97)
        st2 = compute_dedup_edges(dup_catalog, threshold=0.97)
        assert st2.pairs_added == 0
        assert dup_catalog.sql("SELECT count(*) n FROM dedup_edges").to_pylist()[0]["n"] == 1

    def test_dup_losers_keeps_higher_quality(self, dup_catalog):
        compute_dedup_edges(dup_catalog, threshold=0.97)
        # e0 (q=8) vs e1 (q=6) -> e1 loses
        losers = dup_catalog.sql(
            "SELECT episode_id FROM v_dup_losers(0.97, 'keep-higher-quality')"
        ).to_pylist()
        assert [r["episode_id"] for r in losers] == ["e1"]

    def test_dup_losers_keep_longer(self, dup_catalog):
        compute_dedup_edges(dup_catalog, threshold=0.97)
        # e1 has more frames (101) than e0 (100) -> e0 loses under keep-longer
        losers = {r["episode_id"] for r in dup_catalog.sql(
            "SELECT episode_id FROM v_dup_losers(0.97, 'keep-longer')").to_pylist()}
        assert losers == {"e0"}


class TestCurate:
    def test_approves_and_rejects_by_policy(self, dup_catalog):
        compute_dedup_edges(dup_catalog, threshold=0.97)
        st = curate(dup_catalog, dedup_threshold=0.97, dedup_policy="keep-higher-quality",
                    label="approved")
        assert st.selected == 3
        assert st.rejected == 1  # e1 dropped
        assert st.approved == 2  # e0, e2
        labels = {r["episode_id"]: r["label"] for r in dup_catalog.sql(
            "SELECT episode_id, label FROM v_curation").to_pylist()}
        assert labels == {"e0": "approved", "e1": "rejected", "e2": "approved"}

    def test_where_filter(self, dup_catalog):
        st = curate(dup_catalog, where="overall_score >= 7.0", label="approved")
        assert st.selected == 2  # e0(8), e2(7)
        assert st.approved == 2

    def test_latest_label_wins(self, dup_catalog):
        curate(dup_catalog, where="episode_id = 'e0'", label="held")
        curate(dup_catalog, where="episode_id = 'e0'", label="approved")
        rows = dup_catalog.sql(
            "SELECT label FROM v_curation WHERE episode_id='e0'").to_pylist()
        assert len(rows) == 1 and rows[0]["label"] == "approved"
        # history preserved
        assert dup_catalog.sql(
            "SELECT count(*) n FROM curation_labels WHERE episode_id='e0'"
        ).to_pylist()[0]["n"] == 2

    def test_unknown_policy_raises(self, dup_catalog):
        with pytest.raises(ValueError, match="unknown dedup policy"):
            curate(dup_catalog, dedup_threshold=0.97, dedup_policy="bogus")


class TestStudio:
    def test_generate_html(self, dup_catalog):
        from forge.catalog.studio import generate_studio

        compute_dedup_edges(dup_catalog, threshold=0.97)
        out = str(Path(dup_catalog._root) / ".." / "studio.html")
        st = generate_studio(dup_catalog, out, threshold=0.97, max_thumbnails=0)
        assert st.episodes == 3 and st.pairs == 1
        h = Path(out).read_text()
        # embedded data is valid JSON and carries the real rows
        data = json.loads(re.search(r"const FORGE_DATA = (\{.*?\});\nconst THUMBS", h, re.S).group(1))
        assert len(data["episodes"]) == 3
        assert len(data["pairs"]) == 1
        assert "Dedup review" in h and "FORGE_DATA" in h


class TestCLI:
    def test_dedup_curate_studio(self, tmp_path: Path, dup_catalog):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        root = dup_catalog.root_uri

        r = runner.invoke(app, ["catalog", "dedup", "-c", root, "-t", "0.97"])
        assert r.exit_code == 0, r.output
        assert "pairs found" in r.output

        r = runner.invoke(app, ["curate", "-c", root, "--dedup", "0.97", "--label", "approved"])
        assert r.exit_code == 0, r.output
        assert "rejected" in r.output

        out = str(tmp_path / "s.html")
        r = runner.invoke(app, ["studio", "-c", root, "-o", out, "--max-thumbnails", "0"])
        assert r.exit_code == 0, r.output
        assert Path(out).exists()
