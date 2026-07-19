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
        batch_id: str | None = None,
        ingest_date: str | None = None,
    ) -> str:
        """Validate and atomically commit a batch across tables. Returns batch_id.

        Episodes and their quality scores commit together (manifest written
        last), so an episode is either fully present across both tables or not
        at all.
        """
        import uuid

        batch_id = batch_id or uuid.uuid4().hex
        ingest_date = ingest_date or _utc_now().date().isoformat()

        tables: dict[str, pa.Table] = {}
        if episodes:
            schema, _ = TABLES["episodes"]
            tables["episodes"] = validate_rows(episodes, schema, table="episodes")
        if quality_scores:
            schema, _ = TABLES["quality_scores"]
            tables["quality_scores"] = validate_rows(
                quality_scores, schema, table="quality_scores"
            )
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
