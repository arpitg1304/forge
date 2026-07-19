"""The Catalog — the single entry point for a Forge catalog.

Division of labor (enforced by convention):

* **Writes** go exclusively through :class:`~forge.catalog.writer.CatalogWriter`
  (pyarrow → Parquet part-files + manifests).
* **Reads** go exclusively through :meth:`Catalog.sql` (DuckDB over the Parquet
  tables). DuckDB never writes to the catalog.

DuckDB reads the catalog's Parquet through the *same* fsspec filesystem the
writer uses (registered via ``register_filesystem``), so a catalog on local
disk, ``s3://``, ``gs://``, or ``memory://`` is queried identically — including
against a local MinIO, since the S3 credentials/endpoint come from the shared
fsspec client rather than DuckDB's own httpfs config.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from forge.catalog.schema import (
    SCHEMA_VERSION,
    TABLES,
    CatalogSchemaError,
    validate_rows,
)
from forge.catalog.writer import CatalogWriter, _normalize_protocol
from forge.core.exceptions import ForgeError
from forge.io.paths import get_filesystem

if TYPE_CHECKING:
    import pyarrow as pa


def _forge_version() -> str:
    try:
        from importlib.metadata import version

        return version("forge-robotics")
    except Exception:
        return "unknown"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CatalogError(ForgeError):
    """Raised for catalog-level problems (missing, incompatible version, …)."""


class Catalog:
    """A Forge catalog rooted at a local path or cloud URI."""

    def __init__(self, root: str):
        self.root_uri = str(root)
        self._fs, self._root = get_filesystem(self.root_uri)
        self._is_local = bool(
            _normalize_protocol(self._fs.protocol) & {"file", "local"}
        )
        self._writer = CatalogWriter(self.root_uri)
        self._meta = self._load_catalog_json()

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def init(cls, root: str, *, exist_ok: bool = False) -> Catalog:
        """Create a new catalog at ``root`` (writes ``catalog.json``)."""
        fs, base = get_filesystem(str(root))
        catalog_json = "/".join([base.rstrip("/"), "catalog.json"])
        if fs.exists(catalog_json) and not exist_ok:
            raise CatalogError(
                f"Catalog already exists at {root} (catalog.json present). "
                "Pass exist_ok=True to reuse it."
            )
        if not fs.exists(catalog_json):
            writer = CatalogWriter(str(root))
            writer.write_catalog_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "forge_version": _forge_version(),
                    "created_at": _utc_now().isoformat(),
                    "tables": sorted(TABLES),
                }
            )
        return cls(str(root))

    @classmethod
    def open(cls, root: str) -> Catalog:
        """Open an existing catalog, refusing a newer schema version."""
        return cls(str(root))

    def _catalog_json_path(self) -> str:
        return "/".join([self._root.rstrip("/"), "catalog.json"])

    def _load_catalog_json(self) -> dict:
        path = self._catalog_json_path()
        if not self._fs.exists(path):
            raise CatalogError(
                f"No catalog at {self.root_uri} (catalog.json not found). "
                "Run `forge catalog init` first."
            )
        import json

        with self._fs.open(path, "rb") as f:
            meta = json.loads(f.read().decode("utf-8"))
        version = int(meta.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise CatalogError(
                f"Catalog at {self.root_uri} has schema_version {version}, but "
                f"this Forge understands up to {SCHEMA_VERSION}. Upgrade Forge."
            )
        return meta

    @property
    def schema_version(self) -> int:
        return int(self._meta.get("schema_version", SCHEMA_VERSION))

    # -- writes -------------------------------------------------------------

    def commit_batch(
        self,
        *,
        episodes: list[dict] | None = None,
        quality_scores: list[dict] | None = None,
        embeddings: list[dict] | None = None,
        dedup_edges: list[dict] | None = None,
        curation_labels: list[dict] | None = None,
        batch_id: str | None = None,
        ingest_date: str | None = None,
    ) -> str:
        """Validate and atomically commit a batch across tables. Returns batch_id.

        Rows for the tables passed here commit together (manifest written last),
        so a batch is either fully present or not at all.
        """
        import uuid

        batch_id = batch_id or uuid.uuid4().hex
        ingest_date = ingest_date or _utc_now().date().isoformat()

        staged = {
            "episodes": episodes,
            "quality_scores": quality_scores,
            "embeddings": embeddings,
            "dedup_edges": dedup_edges,
            "curation_labels": curation_labels,
        }
        tables: dict[str, pa.Table] = {}
        for name, rows in staged.items():
            if rows:
                schema, _ = TABLES[name]
                tables[name] = validate_rows(rows, schema, table=name)
        if tables:
            self._writer.commit_batch(batch_id, tables, ingest_date=ingest_date)
        return batch_id

    def register_episodes(self, rows: list[dict], *, batch_id: str | None = None) -> str:
        """Commit a batch of ``episodes`` rows."""
        return self.commit_batch(episodes=rows, batch_id=batch_id)

    def add_quality_scores(
        self, rows: list[dict], *, batch_id: str | None = None
    ) -> str:
        """Commit a batch of ``quality_scores`` rows (e.g. a re-score backfill)."""
        return self.commit_batch(quality_scores=rows, batch_id=batch_id)

    def add_embeddings(self, rows: list[dict], *, batch_id: str | None = None) -> str:
        """Commit a batch of ``embeddings`` rows (from an embed backfill)."""
        return self.commit_batch(embeddings=rows, batch_id=batch_id)

    def add_dedup_edges(self, rows: list[dict], *, batch_id: str | None = None) -> str:
        """Commit a batch of ``dedup_edges`` rows (near-duplicate pairs)."""
        return self.commit_batch(dedup_edges=rows, batch_id=batch_id)

    def add_curation_labels(
        self, rows: list[dict], *, batch_id: str | None = None
    ) -> str:
        """Commit a batch of ``curation_labels`` rows (append-log decisions)."""
        return self.commit_batch(curation_labels=rows, batch_id=batch_id)

    # -- reads --------------------------------------------------------------

    def _read_base(self) -> str:
        # DuckDB reads local files by path; cloud/memory via a registered fs by
        # URI (scheme preserved so it routes to the right filesystem).
        return self._root if self._is_local else self.root_uri

    def _manifests(self) -> list[dict]:
        """Load every committed batch manifest under ``_batches/``."""
        import json

        pattern = "/".join([self._root.rstrip("/"), "_batches", "*", "manifest-*.json"])
        try:
            paths = self._fs.glob(pattern)
        except FileNotFoundError:
            return []
        out: list[dict] = []
        for m in paths:
            try:
                with self._fs.open(m, "rb") as fh:
                    out.append(json.loads(fh.read().decode("utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def _committed_files(self, table: str) -> list[str]:
        """Read-URIs of the part-files a manifest has committed for ``table``.

        This is the heart of atomicity: only files listed in a committed
        ``manifest-*.json`` are ever read, so part-files left by a crash before
        its manifest landed are invisible to every query.
        """
        base = self._read_base().rstrip("/")
        files: list[str] = []
        for manifest in self._manifests():
            for entry in manifest.get("files", []):
                if entry.get("table") == table and entry.get("path"):
                    files.append(f"{base}/{entry['path']}")
        return files

    def orphan_part_files(self) -> list[str]:
        """Part-files on disk not referenced by any manifest (crash residue).

        The catalog never reads or deletes these — the manifest convention just
        makes a future ``gc`` possible. Exposed mainly for tests.
        """
        committed = set()
        for manifest in self._manifests():
            for entry in manifest.get("files", []):
                if entry.get("path"):
                    committed.add(entry["path"])
        base = self._root.rstrip("/")
        orphans: list[str] = []
        for table in TABLES:
            pattern = "/".join([base, table, "*", "part-*.parquet"])
            try:
                found = self._fs.glob(pattern)
            except FileNotFoundError:
                found = []
            for p in found:
                rel = p[len(base) + 1:] if p.startswith(base) else p
                if rel not in committed:
                    orphans.append(rel)
        return orphans

    def _connect(self):
        import duckdb

        con = duckdb.connect()
        if not self._is_local:
            # Route cloud/memory reads through the same fsspec client the writer
            # uses (shared creds/endpoint — works with real S3 and MinIO alike).
            con.register_filesystem(self._fs)

        for table, (schema, _partition) in TABLES.items():
            files = self._committed_files(table)
            if files:
                literal = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
                con.execute(
                    f"CREATE VIEW {table} AS "
                    f"SELECT * FROM read_parquet([{literal}], union_by_name=true)"
                )
            else:
                # Empty catalog table — expose the columns so queries still work.
                con.register(f"_{table}_empty", schema.empty_table())
                con.execute(f"CREATE VIEW {table} AS SELECT * FROM _{table}_empty")

        # Latest quality row per episode (max scorer_version, then most recent).
        con.execute(
            """
            CREATE VIEW v_latest_quality AS
            SELECT * FROM quality_scores
            QUALIFY row_number() OVER (
                PARTITION BY episode_id
                ORDER BY scorer_version DESC, computed_at DESC
            ) = 1
            """
        )
        # Latest curation label per episode (append-log; newest wins).
        con.execute(
            """
            CREATE VIEW v_curation AS
            SELECT * FROM curation_labels
            QUALIFY row_number() OVER (
                PARTITION BY episode_id ORDER BY labeled_at DESC
            ) = 1
            """
        )
        # Episodes that LOSE a near-dup pairing under a policy, at a threshold.
        # A table macro so it takes parameters: SELECT * FROM v_dup_losers(0.97,
        # 'keep-higher-quality'). Loser = the weaker side per policy; ties break
        # on the greater episode_id (deterministic). An episode losing any
        # pairing is a loser (union).
        con.execute(
            """
            CREATE MACRO v_dup_losers(thr, policy) AS TABLE (
                WITH e AS (SELECT * FROM dedup_edges WHERE similarity >= thr),
                q AS (SELECT episode_id, overall_score FROM v_latest_quality),
                ep AS (SELECT episode_id, num_frames, ingested_at FROM episodes)
                SELECT DISTINCT CASE
                    WHEN policy = 'keep-longer' THEN
                        CASE WHEN coalesce(pa.num_frames,0) < coalesce(pb.num_frames,0) THEN e.episode_a
                             WHEN coalesce(pa.num_frames,0) > coalesce(pb.num_frames,0) THEN e.episode_b
                             ELSE greatest(e.episode_a, e.episode_b) END
                    WHEN policy = 'keep-first' THEN
                        CASE WHEN pa.ingested_at < pb.ingested_at THEN e.episode_b
                             WHEN pa.ingested_at > pb.ingested_at THEN e.episode_a
                             ELSE greatest(e.episode_a, e.episode_b) END
                    ELSE  -- keep-higher-quality (default)
                        CASE WHEN coalesce(qa.overall_score,-1) < coalesce(qb.overall_score,-1) THEN e.episode_a
                             WHEN coalesce(qa.overall_score,-1) > coalesce(qb.overall_score,-1) THEN e.episode_b
                             ELSE greatest(e.episode_a, e.episode_b) END
                END AS episode_id
                FROM e
                LEFT JOIN q qa ON qa.episode_id = e.episode_a
                LEFT JOIN q qb ON qb.episode_id = e.episode_b
                LEFT JOIN ep pa ON pa.episode_id = e.episode_a
                LEFT JOIN ep pb ON pb.episode_id = e.episode_b
            )
            """
        )
        return con

    def sql(self, query: str) -> pa.Table:
        """Run a read-only SQL query and return a ``pyarrow.Table``.

        Views available: ``episodes``, ``quality_scores``, ``v_latest_quality``.
        Call ``.to_pandas()`` on the result for a DataFrame.
        """
        con = self._connect()
        try:
            result = con.execute(query)
            # Fully materialize before closing the connection (unlike .arrow(),
            # which can return a lazy reader). to_arrow_table() is the current
            # name; fetch_arrow_table() is the pre-1.3 spelling.
            to_arrow = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
            return to_arrow()
        finally:
            con.close()

    # -- convenience --------------------------------------------------------

    def episode_hashes(self) -> set[str]:
        """Return the set of ``content_hash`` values already in the catalog.

        Used by ingest for exact-duplicate skipping. Cheap on an empty or small
        catalog; reads only the ``content_hash`` column.
        """
        tbl = self.sql("SELECT content_hash FROM episodes")
        return set(tbl.column("content_hash").to_pylist())

    # -- embeddings & search -----------------------------------------------

    def embedding_model_ids(self) -> list[str]:
        """Distinct ``model_id``s present in the embeddings table."""
        tbl = self.sql("SELECT DISTINCT model_id FROM embeddings ORDER BY model_id")
        return tbl.column("model_id").to_pylist()

    def embedded_episode_ids(self, model_id: str) -> set[str]:
        """Episode ids already embedded for ``model_id`` (for idempotent embed)."""
        tbl = self.sql(
            "SELECT DISTINCT episode_id FROM embeddings WHERE model_id = '"
            + model_id.replace("'", "''")
            + "'"
        )
        return set(tbl.column("episode_id").to_pylist())

    def _resolve_model_id(self, model_id: str | None) -> str:
        ids = self.embedding_model_ids()
        if not ids:
            raise CatalogError(
                "No embeddings in this catalog. Run `forge embed` first."
            )
        if model_id is None:
            if len(ids) == 1:
                return ids[0]
            raise CatalogError(
                "Multiple embedding models present; pass model_id explicitly. "
                f"Available: {ids}"
            )
        if model_id not in ids:
            raise CatalogError(f"model_id {model_id!r} not in catalog. Available: {ids}")
        return model_id

    def search(
        self,
        query: str | None = None,
        *,
        like: str | None = None,
        model_id: str | None = None,
        level: str = "episode",
        camera: str | None = None,
        top: int = 20,
        device: str = "auto",
    ) -> pa.Table:
        """Semantic search over episode embeddings. Returns a ranked pyarrow Table.

        Provide either ``query`` (text → embedded with the model's text tower)
        or ``like`` (an episode_id → uses that episode's vector). Exactly one
        ``model_id`` is used per search; if the catalog has several and none is
        given, this raises.
        """
        import numpy as np

        if (query is None) == (like is None):
            raise ValueError("Provide exactly one of `query` (text) or `like` (episode_id).")

        model_id = self._resolve_model_id(model_id)
        esc = lambda s: s.replace("'", "''")  # noqa: E731
        dim_rows = self.sql(
            f"SELECT DISTINCT dim FROM embeddings WHERE model_id = '{esc(model_id)}'"
        ).to_pylist()
        dim = int(dim_rows[0]["dim"])

        if like is not None:
            where = f"episode_id = '{esc(like)}' AND model_id = '{esc(model_id)}' AND level = '{esc(level)}'"
            if camera:
                where += f" AND camera = '{esc(camera)}'"
            vecs = self.sql(f"SELECT vector FROM embeddings WHERE {where}").to_pylist()
            if not vecs:
                raise CatalogError(
                    f"episode {like!r} has no {level} embeddings for {model_id}"
                )
            qv = np.asarray([v["vector"] for v in vecs], dtype=np.float32).mean(axis=0)
        else:
            from forge.embed import get_model

            name = model_id.split("@", 1)[0]
            model = get_model(name, device=device)
            qv = np.asarray(model.embed_text([query])[0], dtype=np.float32)

        norm = float(np.linalg.norm(qv)) or 1.0
        qv = qv / norm
        qlit = "[" + ",".join(f"{float(x):.8g}" for x in qv) + "]"

        where = f"emb.model_id = '{esc(model_id)}' AND emb.level = '{esc(level)}'"
        if camera:
            where += f" AND emb.camera = '{esc(camera)}'"

        # One row per episode (best-matching camera), ranked by cosine.
        return self.sql(
            f"""
            SELECT episode_id, score, camera, language_instruction, source_uri
            FROM (
                SELECT emb.episode_id, emb.camera,
                       array_cosine_similarity(
                           emb.vector::FLOAT[{dim}], {qlit}::FLOAT[{dim}]
                       ) AS score,
                       ep.language_instruction, ep.source_uri
                FROM embeddings emb
                JOIN episodes ep USING(episode_id)
                WHERE {where}
                QUALIFY row_number() OVER (
                    PARTITION BY emb.episode_id ORDER BY score DESC
                ) = 1
            )
            ORDER BY score DESC
            LIMIT {int(top)}
            """
        )

    def stats(self) -> dict[str, Any]:
        """Return summary statistics as canned SQL through :meth:`sql`."""
        totals = self.sql(
            "SELECT count(*) AS episodes, "
            "coalesce(sum(num_frames), 0) AS frames, "
            "coalesce(sum(duration_s), 0.0) AS seconds "
            "FROM episodes"
        ).to_pylist()[0]

        per_task = self.sql(
            "SELECT coalesce(task, '(none)') AS task, count(*) AS n "
            "FROM episodes GROUP BY 1 ORDER BY n DESC"
        ).to_pylist()
        per_robot = self.sql(
            "SELECT coalesce(robot, '(none)') AS robot, count(*) AS n "
            "FROM episodes GROUP BY 1 ORDER BY n DESC"
        ).to_pylist()

        score = self.sql(
            "SELECT count(*) AS scored, min(overall_score) AS min, "
            "avg(overall_score) AS mean, median(overall_score) AS median, "
            "max(overall_score) AS max FROM v_latest_quality"
        ).to_pylist()[0]

        return {
            "episodes": int(totals["episodes"]),
            "total_frames": int(totals["frames"]),
            "total_hours": round(float(totals["seconds"]) / 3600.0, 3),
            "per_task": per_task,
            "per_robot": per_robot,
            "overall_score": score,
        }


__all__ = ["Catalog", "CatalogError", "CatalogSchemaError"]
