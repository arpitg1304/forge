# `forge.catalog`

An append-only, queryable registry of episodes — the system-of-record layer for
Forge. This is **Phase 1** of the [Forge Data Engine design](../../docs/forge_data_engine_design.md):
the `episodes` and `quality_scores` tables plus ingest, query, and stats.

## Why

Forge on its own operates one dataset at a time. Labs that collect thousands of
episodes a day need a single growing corpus where every episode is registered
once, annotated with derived signals (quality now; embeddings/dedup later), and
queryable for curation. The catalog is that corpus.

Design commitments (Phase 1 slice of them):

- **Append-only & immutable.** Raw data is never touched; derived facts are
  versioned annotations. Every write is a new part-file; nothing is rewritten.
- **Zero-server.** Parquet files in a directory or bucket, queried by embedded
  DuckDB. No database to run.
- **Open & portable.** Plain partitioned Parquet — readable by pandas / Polars /
  Spark without Forge.
- **Local or cloud, identical.** Local dir, `s3://`, `gs://`, or `memory://`
  behave the same (fsspec under the hood).

## Architecture

```
                    Catalog  (single entry point)
                   /        \
        WRITES (pyarrow)     READS (DuckDB / SQL)
              |                     |
   writer.py  |                     |  catalog.sql(), views
  (part-files + manifests)   (reads only manifested files)
              |                     |
              └─────  Parquet tables in <root>/ ─────┘
```

Strict division of labor:

- **`schema.py`** — the pyarrow schemas, `SCHEMA_VERSION`, `SCORER_VERSION`, and
  `validate_rows()` (every write is validated, failing loudly on mismatch).
- **`writer.py`** — the *only* module that writes into a catalog directory.
- **`catalog.py`** — the `Catalog` class: `init`/`open`, `sql()`, `commit_batch`,
  `stats()`, and the DuckDB views. Reads only.
- **`ingest.py`** — the ingest pipeline (discover → hash → skip → register →
  score → stage), reusing existing Forge readers and the quality scorer.
- **`cli.py`** — `forge catalog init/stats`, `forge ingest`, `forge query`.

> **Invariant:** nothing outside `writer.py` writes catalog files; nothing reads
> catalog data except through `Catalog.sql()`.

## Storage layout

```
<catalog-root>/
├── catalog.json                              # schema_version, forge version, created_at
├── episodes/ingest_date=YYYY-MM-DD/part-<uuid>.parquet
├── quality_scores/ingest_date=YYYY-MM-DD/part-<uuid>.parquet
└── _batches/<batch_id>/
    ├── manifest-<uuid>.json                  # commit marker (one per flush)
    └── _checkpoint.json                      # progress marker (mutable)
```

## Commit protocol (atomicity)

1. A flush writes all its part-files first.
2. It then writes a uniquely-named `manifest-<uuid>.json` **last** — the commit
   marker listing exactly the files it produced.
3. **Reads only ever load files named in a committed manifest.** Part-files left
   by a crash before its manifest landed are invisible to every query — so a
   query never sees a partially-committed batch.

On local filesystems each file is written to a temp name and renamed into place
(atomic appearance); on object stores the PUT is itself atomic.

Recovery is automatic: a crashed ingest leaves orphan part-files and no manifest;
re-running skips already-committed episodes by **content hash** and re-does only
the lost ones. Orphan detection is exposed (`Catalog.orphan_part_files()`) so a
future `gc` can clean them — Phase 1 leaves the convention in place but does not
delete.

## Schemas

`episodes` — one row per ingested episode, facts only (id, `content_hash`,
source, robot/task/language, frame/time/fps, cameras, dims, ingest provenance,
and an `extra` map escape hatch).

`quality_scores` — one row per `(episode_id, scorer_version)`: `overall_score`
plus one column per Forge quality metric (`ldlj`, `sparc`, `dead_fraction`,
`gripper_chatter_rate`, `joint_path_length`, `overall_saturation`,
`mean_entropy`, `psd_high_fraction`, `state_action_consistency`,
`static_fraction`, `jitter_ratio`), `flags`, and `computed_at`. Re-scoring the
corpus appends new rows under a new `scorer_version`; old rows remain for
reproducibility.

`catalog.json` records `schema_version`; Forge refuses to open a catalog written
by a newer version than it understands.

## Content hashing

`content_hash` is an xxh3 of the episode's **proprio stream** (state/action/
timestamp, streamed frame-by-frame) plus a shape signature. It's deterministic
and per-episode, which is what makes re-ingest idempotent. Forge readers expose
logical episodes rather than per-episode byte ranges (multiple episodes often
share one parquet/video file), so hashing content is the portable choice;
images are excluded (expensive to decode, and proprio is a strong fingerprint
for teleop data).

## Query surface

Views registered on every connection:

- `episodes`, `quality_scores` — the raw tables.
- `v_latest_quality` — the quality row for the newest `scorer_version` per
  episode (then most recent `computed_at`).

```python
cat.sql("SELECT task, count(*) FROM episodes GROUP BY task")   # -> pyarrow.Table
```

## Example

[`catalog_example_droid_100/`](catalog_example_droid_100/) is a small, ready-to-query
catalog built from the `lerobot/droid_100` dataset (100 episodes, ~44 KB). Its
[README](catalog_example_droid_100/README.md) has example queries and the exact
commands to reproduce it (from a local MinIO `s3://` source, or straight from
`hf://` with no MinIO).

```bash
forge catalog stats -c forge/catalog/catalog_example_droid_100
```

## Not in Phase 1

Embeddings, dedup edges, curation labels, and snapshots (Phases 2–4);
distributed execution, locking, compaction, and gc. The manifest convention is
in place so gc becomes possible later, but it is not implemented.
