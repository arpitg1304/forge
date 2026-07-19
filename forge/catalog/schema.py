"""Table schemas for the Forge catalog.

The catalog is a set of append-only Parquet tables. This module is the single
source of truth for their pyarrow schemas, the catalog schema version, and the
current quality scorer version. Every write goes through :func:`validate_rows`,
which builds a validated ``pyarrow.Table`` and fails loudly on any mismatch.

Phase 1 defines exactly two tables: ``episodes`` (the registry) and
``quality_scores`` (versioned derived metrics). Later phases (embeddings, dedup,
curation, snapshots) are intentionally absent.
"""

from __future__ import annotations

import pyarrow as pa

from forge.core.exceptions import ForgeError

# Bump when the catalog table schemas change in a breaking way. catalog.json
# records this; Forge refuses to open a catalog written by a newer version.
SCHEMA_VERSION = 1

# Version tag stamped on every quality_scores row. The quality module has no
# version of its own today, so the catalog owns this constant. Bump it when the
# scoring math changes so old scores remain distinguishable and reproducible.
SCORER_VERSION = "q-v1"

# Names of the per-episode quality metrics, mirrored EXACTLY from the flat dict
# produced by forge.quality.models.EpisodeQuality.to_dict(). Kept as a list so
# schema, validation, and the ingest mapping never drift apart.
QUALITY_METRIC_COLUMNS = [
    "ldlj",
    "sparc",
    "dead_fraction",
    "gripper_chatter_rate",
    "joint_path_length",
    "overall_saturation",
    "mean_entropy",
    "psd_high_fraction",
    "state_action_consistency",
    "static_fraction",
    "jitter_ratio",
]

_CAMERA_STRUCT = pa.struct(
    [
        pa.field("name", pa.string(), nullable=False),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("fps", pa.float32()),
    ]
)

# One row per ingested episode. Facts only — nothing derived, nothing mutable.
EPISODES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("source_format", pa.string(), nullable=False),
        pa.field("robot", pa.string()),
        pa.field("task", pa.string()),
        pa.field("language_instruction", pa.string()),
        pa.field("operator_id", pa.string()),
        pa.field("scene_id", pa.string()),
        pa.field("num_frames", pa.int32(), nullable=False),
        pa.field("duration_s", pa.float32(), nullable=False),
        pa.field("fps", pa.float32(), nullable=False),
        pa.field("cameras", pa.list_(_CAMERA_STRUCT), nullable=False),
        pa.field("state_dim", pa.int16()),
        pa.field("action_dim", pa.int16()),
        pa.field("ingest_batch_id", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("extra", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)

# One row per (episode_id, scorer_version).
QUALITY_SCORES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("scorer_version", pa.string(), nullable=False),
        pa.field("overall_score", pa.float32()),
        *[pa.field(name, pa.float32()) for name in QUALITY_METRIC_COLUMNS],
        pa.field("flags", pa.list_(pa.string()), nullable=False),
        pa.field("computed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Logical table name -> (pyarrow schema, partition column).
TABLES: dict[str, tuple[pa.Schema, str]] = {
    "episodes": (EPISODES_SCHEMA, "ingest_date"),
    "quality_scores": (QUALITY_SCORES_SCHEMA, "ingest_date"),
}


class CatalogSchemaError(ForgeError):
    """Raised when rows don't conform to a catalog table schema."""


def validate_rows(rows: list[dict], schema: pa.Schema, *, table: str) -> pa.Table:
    """Build a validated ``pyarrow.Table`` from ``rows`` for ``table``.

    Fails loudly (``CatalogSchemaError``) on any mismatch: unexpected columns,
    a required (non-nullable) field that is missing or null, or a value that
    can't be coerced to the declared type. This is the only sanctioned way to
    turn Python dicts into catalog Parquet data.
    """
    allowed = set(schema.names)
    required = {f.name for f in schema if not f.nullable}

    for i, row in enumerate(rows):
        extra_keys = set(row) - allowed
        if extra_keys:
            raise CatalogSchemaError(
                f"{table}: row {i} has unexpected column(s) "
                f"{sorted(extra_keys)}; allowed: {sorted(allowed)}"
            )
        missing = {k for k in required if row.get(k) is None}
        if missing:
            raise CatalogSchemaError(
                f"{table}: row {i} missing required value(s) {sorted(missing)}"
            )

    try:
        tbl = pa.Table.from_pylist(rows, schema=schema)
        tbl.validate(full=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise CatalogSchemaError(f"{table}: {exc}") from exc
    return tbl
