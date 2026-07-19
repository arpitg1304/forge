"""The Forge catalog — an append-only, queryable registry of episodes.

A catalog is a set of Parquet tables (``episodes``, ``quality_scores``) in a
local directory or cloud bucket, written with pyarrow and queried with DuckDB.
It turns Forge from a per-dataset tool into a system of record: every episode a
lab ingests is registered once and annotated with quality, then queryable via
SQL.

Entry point:

    from forge.catalog import Catalog
    from forge.catalog.ingest import ingest

    cat = Catalog.init("s3://lab-bucket/forge-catalog")
    ingest(["s3://lab-bucket/raw/2026-07-18/"], cat)
    cat.sql("SELECT task, count(*) FROM episodes GROUP BY task")

Package-level imports are lazy (PEP 562) so merely importing :mod:`forge` — or
running ``forge --help`` on a base install — never pulls in the optional
``pyarrow`` / ``duckdb`` dependencies; they load only when a catalog symbol is
used. (The ``ingest`` pipeline lives at :mod:`forge.catalog.ingest` to avoid a
name clash between the submodule and the function.)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "Catalog",
    "CatalogError",
    "CatalogSchemaError",
    "SCHEMA_VERSION",
    "SCORER_VERSION",
]

# name -> submodule that defines it. importlib.import_module bypasses this
# package's __getattr__, so there's no re-entrancy.
_LAZY = {
    "Catalog": "forge.catalog.catalog",
    "CatalogError": "forge.catalog.catalog",
    "CatalogSchemaError": "forge.catalog.schema",
    "SCHEMA_VERSION": "forge.catalog.schema",
    "SCORER_VERSION": "forge.catalog.schema",
}

if TYPE_CHECKING:
    from forge.catalog.catalog import Catalog, CatalogError
    from forge.catalog.schema import SCHEMA_VERSION, SCORER_VERSION, CatalogSchemaError


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
