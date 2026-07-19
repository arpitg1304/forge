"""The only module that writes files into a catalog directory.

Enforces the catalog's write invariants:

* **Append-only.** Every commit creates new ``part-<uuid>.parquet`` files; no
  existing file is ever rewritten.
* **Manifest-last commit protocol.** All part-files for a batch are written
  first; the batch ``_manifest.json`` is written LAST and is the commit marker.
  A crash before the manifest leaves orphan part-files and no manifest — a
  re-run recovers because episodes are skipped by content hash.
* **Atomic file appearance.** On local filesystems each file is written to a
  temp name and renamed into place, so a reader never sees a half-written file.
  On object stores the object PUT is itself atomic.

Works over any fsspec filesystem (local, ``s3://``, ``gs://``, ``memory://``)
via :func:`forge.io.paths.get_filesystem`, so catalogs live locally or in the
cloud with identical semantics.
"""

from __future__ import annotations

import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from forge.catalog.schema import SCHEMA_VERSION
from forge.io.paths import get_filesystem


def _normalize_protocol(protocol) -> set[str]:
    if isinstance(protocol, str):
        return {protocol}
    return set(protocol)


class CatalogWriter:
    """Writes part-files, manifests, and ``catalog.json`` for one catalog root."""

    def __init__(self, root: str):
        self.root_uri = str(root)
        self._fs, self._root = get_filesystem(self.root_uri)
        self._is_local = bool(
            _normalize_protocol(self._fs.protocol) & {"file", "local"}
        )

    # -- path helpers -------------------------------------------------------

    def _join(self, *parts: str) -> str:
        return "/".join([self._root.rstrip("/"), *[p.strip("/") for p in parts]])

    # -- atomic primitives --------------------------------------------------

    def _write_bytes_atomic(self, path: str, write_fn) -> None:
        """Write via ``write_fn(file_obj)``, atomically on local filesystems.

        ``write_fn`` receives an open binary file handle. On local filesystems
        the write goes to a temp sibling and is renamed into place; elsewhere
        it's written directly (object PUTs appear atomically).
        """
        parent = path.rsplit("/", 1)[0]
        try:
            self._fs.makedirs(parent, exist_ok=True)
        except (FileExistsError, NotImplementedError):
            pass

        if self._is_local:
            tmp = f"{path}.tmp-{uuid.uuid4().hex}"
            try:
                with self._fs.open(tmp, "wb") as f:
                    write_fn(f)
                self._fs.mv(tmp, path)
            except BaseException:
                if self._fs.exists(tmp):
                    self._fs.rm(tmp)
                raise
        else:
            with self._fs.open(path, "wb") as f:
                write_fn(f)

    def _write_json_atomic(self, path: str, payload: dict) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._write_bytes_atomic(path, lambda f: f.write(data))

    # -- public API ---------------------------------------------------------

    def write_catalog_json(self, meta: dict) -> None:
        """Write the root ``catalog.json`` (schema version, config, timestamps)."""
        self._write_json_atomic(self._join("catalog.json"), meta)

    def write_checkpoint(self, batch_id: str, payload: dict) -> None:
        """Write/overwrite a batch checkpoint (progress marker, not table data).

        The only intentionally-mutable file in a catalog: updated after each
        flush so a crashed ingest's progress is visible. Correctness of resume
        still comes from content-hash skipping, not this file.
        """
        self._write_json_atomic(
            self._join("_batches", batch_id, "_checkpoint.json"), payload
        )

    def commit_batch(
        self,
        batch_id: str,
        tables: dict[str, pa.Table],
        *,
        ingest_date: str,
    ) -> dict:
        """Commit one batch: write all part-files, then the manifest LAST.

        Args:
            batch_id: Groups this ingest run's files.
            tables: Validated ``pyarrow.Table`` per logical table name. Empty
                tables are skipped.
            ingest_date: UTC ``YYYY-MM-DD`` partition value for this batch.

        Returns:
            The manifest dict that was committed.
        """
        produced: list[dict] = []
        for table_name, table in tables.items():
            if table.num_rows == 0:
                continue
            rel = f"{table_name}/ingest_date={ingest_date}/part-{uuid.uuid4().hex}.parquet"
            self._write_bytes_atomic(
                self._join(rel), lambda f, t=table: pq.write_table(t, f)
            )
            produced.append({"table": table_name, "path": rel, "rows": table.num_rows})

        manifest = {
            "batch_id": batch_id,
            "schema_version": SCHEMA_VERSION,
            "ingest_date": ingest_date,
            "files": produced,
        }
        # The manifest is the commit marker for THIS flush — written strictly
        # last, under a unique name so multiple flushes in one batch each get
        # their own atomic commit marker (readers trust only manifested files).
        manifest_name = f"manifest-{uuid.uuid4().hex}.json"
        self._write_json_atomic(
            self._join("_batches", batch_id, manifest_name), manifest
        )
        return manifest
