"""Table schemas for the Forge catalog.

The catalog is a set of append-only Parquet tables. This module is the single
source of truth for their pyarrow schemas, the catalog schema version, and the
current quality scorer version. Every write goes through :func:`validate_rows`,
which builds a validated ``pyarrow.Table`` and fails loudly on any mismatch.

Tables by phase: ``episodes`` + ``quality_scores`` (1), ``embeddings`` (2),
``dedup_edges`` + ``curation_labels`` (3). Snapshots (4) are JSON manifests, not
a table.
"""

from __future__ import annotations

import pyarrow as pa

from forge.core.exceptions import ForgeError

# Bump when the catalog table schemas change in a breaking way. catalog.json
# records this; Forge refuses to open a catalog written by a newer version. Each
# bump so far is purely additive (new tables), so older catalogs open unchanged
# and gain the new tables on first use. v2 added `embeddings`; v3 adds
# `dedup_edges` and `curation_labels`.
SCHEMA_VERSION = 3

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

# One row per (episode_id, model_id, level, camera). Vectors are stored as a
# variable-length list<float32> (+ an explicit `dim`) rather than a fixed-size
# list, because one table holds multiple model_ids with different dimensions.
# Search filters to a single model_id, then casts to FLOAT[dim] for cosine.
EMBEDDINGS_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("level", pa.string(), nullable=False),  # episode | instruction
        pa.field("camera", pa.string()),  # null for instruction level
        pa.field("pooling", pa.string(), nullable=False),
        pa.field("dim", pa.int32(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),
        pa.field("num_frames_sampled", pa.int32()),
        pa.field("computed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# One row per detected near-duplicate pair. Edges record *similarity* — which
# episode "wins" is decided at curation time by policy, never baked in here
# (that would freeze the policy). Canonical order: episode_a < episode_b.
DEDUP_EDGES_SCHEMA = pa.schema(
    [
        pa.field("episode_a", pa.string(), nullable=False),
        pa.field("episode_b", pa.string(), nullable=False),
        pa.field("similarity", pa.float32(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("method", pa.string(), nullable=False),  # ann-cosine | exact-hash
        pa.field("computed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Append-log of human/policy decisions. Latest row wins per episode (resolved
# with a window function at query time); appending never deletes history.
CURATION_LABELS_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),  # approved | rejected | held
        pa.field("reason", pa.string()),
        pa.field("labeled_by", pa.string(), nullable=False),  # user or policy:<name>
        pa.field("labeled_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Logical table name -> (pyarrow schema, partition column).
TABLES: dict[str, tuple[pa.Schema, str]] = {
    "episodes": (EPISODES_SCHEMA, "ingest_date"),
    "quality_scores": (QUALITY_SCORES_SCHEMA, "ingest_date"),
    "embeddings": (EMBEDDINGS_SCHEMA, "ingest_date"),
    "dedup_edges": (DEDUP_EDGES_SCHEMA, "ingest_date"),
    "curation_labels": (CURATION_LABELS_SCHEMA, "ingest_date"),
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
