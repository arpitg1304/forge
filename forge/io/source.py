"""A dataset root that may be local or remote, for streaming metadata reads.

Format readers use :class:`DataSource` to inspect a dataset without downloading
it: for ``s3://`` / ``gs://`` sources the small metadata files (``info.json``,
Parquet footers) are fetched with **range requests**, not a full copy. Local
paths go through the same interface (fsspec's local filesystem), so a reader can
share one code path — but readers keep their existing local logic and only reach
for this on the remote/streaming path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge.io.paths import get_filesystem, is_remote_uri

if TYPE_CHECKING:
    import pyarrow as pa


class DataSource:
    """A dataset root (local or fsspec-remote) with metadata-read helpers."""

    def __init__(self, uri: str):
        self.uri = str(uri)
        self.is_remote = is_remote_uri(self.uri)
        self.fs, self.root = get_filesystem(self.uri)

    def path(self, *parts: str) -> str:
        """Join ``parts`` onto the root as an fs-native path."""
        return "/".join([self.root.rstrip("/"), *[p.strip("/") for p in parts]])

    def exists(self, *parts: str) -> bool:
        try:
            return self.fs.exists(self.path(*parts))
        except FileNotFoundError:
            return False

    def open(self, *parts: str):
        """Open a file (binary) — a range-capable handle for remote sources."""
        return self.fs.open(self.path(*parts), "rb")

    def read_json(self, *parts: str) -> dict:
        with self.open(*parts) as f:
            return json.loads(f.read())

    def read_text(self, *parts: str) -> str:
        with self.open(*parts) as f:
            return f.read().decode("utf-8")

    def glob(self, *parts: str) -> list[str]:
        try:
            return self.fs.glob(self.path(*parts))
        except FileNotFoundError:
            return []

    def parquet_schema(self, full_path: str) -> pa.Schema:
        """Read a Parquet file's schema from its footer (range read; no full copy)."""
        import pyarrow.parquet as pq

        with self.fs.open(full_path, "rb") as f:
            return pq.read_schema(f)

    def read_parquet(self, full_path: str, **kwargs) -> pa.Table:
        """Read a Parquet file (optionally ``columns=…``) via range reads."""
        import pyarrow.parquet as pq

        with self.fs.open(full_path, "rb") as f:
            return pq.read_table(f, **kwargs)
