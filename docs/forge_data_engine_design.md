# Forge Data Engine — Design Doc

**Status:** Draft / RFC
**Scope:** End-to-end lifecycle of robot demonstration data in Forge: ingestion → registration → quality scoring → embedding → dedup → query/curation → snapshots → training-set export.
**Non-goals (for now):** Distributed execution (Ray), a hosted service, per-frame vector search at scale, real-time streaming ingest.

---

## 1. Motivation

Forge today is a *tool*: it operates on one dataset at a time (`forge convert`, `forge quality`, `forge filter`). Labs that collect data continuously — thousands of episodes per day across operators, robots, and tasks — need a *system of record*: a single growing corpus where every episode is registered once, annotated with derived signals (quality, embeddings, dedup), and queryable for curation.

This doc defines that system. Design goals:

1. **Append-only and immutable.** Raw episode data lands once and never changes. All derived facts are versioned annotations, never in-place mutations.
2. **Zero-server.** The store is Parquet files in a directory or object store bucket. The query engine is embedded (DuckDB). No database to operate, no migrations to run against a live service.
3. **Open and portable.** Everything is readable by pandas / Polars / Daft / Spark without Forge. Forge adds convenience, not lock-in.
4. **Reproducible.** Any training set ever produced can be reconstructed exactly from a snapshot ID.
5. **Incremental.** Daily ingest of N new episodes costs O(N) work (plus one ANN pass for dedup), regardless of corpus size.

## 2. Architecture Overview

```
                        ┌────────────────────────────────────────────┐
                        │              FORGE DATA ENGINE             │
                        │                                            │
 teleop rigs  ──┐       │  ┌─────────┐   ┌──────────────────────┐    │
 sim rollouts ──┼──────►│  │ INGEST  │──►│  episodes (registry)  │    │
 hf:// / s3://──┘       │  └────┬────┘   └──────────────────────┘    │
                        │       │ per-episode, checkpointed          │
                        │       ▼                                    │
                        │  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
                        │  │ quality  │  │  embed   │  │  dedup  │  │
                        │  │ _scores  │  │ dings    │  │ _edges  │  │
                        │  └──────────┘  └──────────┘  └─────────┘  │
                        │       │             │             │        │
                        │       └──────┬──────┴─────────────┘        │
                        │              ▼                             │
                        │       ┌────────────┐    ┌──────────────┐   │
                        │       │ QUERY (SQL │───►│  snapshots    │   │
                        │       │  / DuckDB) │    │  + export     │   │
                        │       └────────────┘    └──────────────┘   │
                        └────────────────────────────────────────────┘
```

Two layers:

- **Data plane:** raw episode files (LeRobot/RLDS/HDF5/…) stored wherever the lab keeps them (`s3://`, `gs://`, local). Forge references them by URI; it does not copy or own them.
- **Catalog plane:** a set of append-only Parquet tables (the "forgeдb", working name `catalog/`) that describe and annotate the corpus. This doc is mostly about the catalog plane.

## 3. Storage Layout

```
s3://lab-bucket/
├── raw/                                  # data plane — lab-owned, any format
│   ├── 2026-07-18/run_0413/...
│   └── ...
└── forge-catalog/                        # catalog plane — owned by Forge
    ├── catalog.json                      # version, config, table locations
    ├── episodes/
    │   └── ingest_date=2026-07-18/part-*.parquet
    ├── quality_scores/
    │   └── ingest_date=2026-07-18/part-*.parquet
    ├── embeddings/
    │   └── model_id=siglip-so400m@a1b2/ingest_date=.../part-*.parquet
    ├── dedup_edges/
    │   └── computed_date=2026-07-18/part-*.parquet
    ├── curation_labels/
    │   └── part-*.parquet                # small; unpartitioned append log
    └── snapshots/
        ├── snap_2026-07-18_pickplace_v1.json
        └── ...
```

Rules:

- **Writes are new files only.** A daily ingest appends `part-*.parquet` files under the day's partition. Nothing rewrites existing files. Concurrent readers are always safe.
- **Partitioning** is by `ingest_date` (and `model_id` for embeddings) — matches the write pattern and prunes well for "recent data" queries.
- **Compaction** is an optional maintenance job (`forge catalog compact`) that merges small files within a closed partition. It replaces files atomically per-partition and is the only rewrite in the system.
- `catalog.json` carries a schema version so future migrations are explicit and detectable.

## 4. Schemas

Types below are Arrow/Parquet types. `PK` is logical (Parquet has no constraints; uniqueness is enforced at ingest).

### 4.1 `episodes` — the registry

One row per ingested episode. Facts only — nothing derived, nothing that changes.

| column | type | notes |
|---|---|---|
| `episode_id` | `string` (uuid7) | PK. Time-ordered UUID so sorting by ID ≈ ingest order |
| `content_hash` | `string` | xxh3 of raw episode bytes; exact-dup rejection at ingest |
| `source_uri` | `string` | `s3://…/run_0413`, `hf://…` — where raw data lives |
| `source_format` | `string` | `lerobot-v3`, `rlds`, `hdf5`, `rosbag`, … |
| `robot` | `string` | embodiment id, e.g. `franka`, `aloha` |
| `task` | `string` | task id/name; nullable |
| `language_instruction` | `string` | nullable |
| `operator_id` | `string` | who collected it; nullable |
| `scene_id` | `string` | nullable |
| `num_frames` | `int32` | |
| `duration_s` | `float32` | |
| `fps` | `float32` | nominal control/camera rate |
| `cameras` | `list<struct<name, width, height, fps>>` | |
| `state_dim` / `action_dim` | `int16` | |
| `ingest_batch_id` | `string` | groups one ingest run |
| `ingested_at` | `timestamp[us, UTC]` | |
| `extra` | `map<string,string>` | escape hatch for lab-specific metadata |

### 4.2 `quality_scores` — versioned derived metrics

One row per `(episode_id, scorer_version)`. Re-scoring the corpus with an improved scorer appends new rows; old rows remain for reproducibility.

| column | type | notes |
|---|---|---|
| `episode_id` | `string` | FK → episodes |
| `scorer_version` | `string` | e.g. `q-v3`; maps to a config committed in the repo |
| `overall_score` | `float32` | 0–10 |
| `ldlj` | `float32` | smoothness |
| `dead_action_ratio` | `float32` | |
| `gripper_chatter` | `float32` | |
| `static_ratio` | `float32` | |
| `timestamp_jitter` | `float32` | |
| `saturation_ratio` | `float32` | |
| `action_entropy` | `float32` | |
| `path_length_norm` | `float32` | |
| `flags` | `list<string>` | `jerky`, `mostly_static`, … |
| `computed_at` | `timestamp[us, UTC]` | |

New metrics = new columns (Parquet schemas can evolve additively) or a `metrics: map<string,float32>` overflow column — start with explicit columns for the 8 shipped metrics, add the map when third-party metrics arrive.

### 4.3 `embeddings` — versioned per model

One row per `(episode_id, model_id, level, camera)`.

| column | type | notes |
|---|---|---|
| `episode_id` | `string` | |
| `model_id` | `string` | `siglip-so400m@<ckpt-hash>` — name **and** checkpoint hash |
| `level` | `string` | `episode` \| `instruction` (per-frame: see §9) |
| `camera` | `string` | which view was embedded; null for `instruction` |
| `pooling` | `string` | `mean`, `first-mid-last`, … |
| `vector` | `fixed_size_list<float32>[D]` | D fixed per model_id |
| `computed_at` | `timestamp[us, UTC]` | |

Mixing vectors across `model_id`s in one similarity computation is an error; Forge enforces this in every query path.

### 4.4 `dedup_edges` — similarity facts, not verdicts

One row per detected near-duplicate pair. **Edges record similarity; which episode "wins" is decided at curation time by policy** (e.g. keep the higher quality score). Baking verdicts into edges would freeze the policy.

| column | type | notes |
|---|---|---|
| `episode_a` / `episode_b` | `string` | canonical order: a < b |
| `similarity` | `float32` | cosine |
| `model_id` | `string` | embeddings used |
| `method` | `string` | `exact-hash` \| `ann-cosine` |
| `computed_at` | `timestamp[us, UTC]` | |

### 4.5 `curation_labels` — human/policy decisions (append log)

| column | type | notes |
|---|---|---|
| `episode_id` | `string` | |
| `label` | `string` | `approved` \| `rejected` \| `held` |
| `reason` | `string` | free text or rule id |
| `labeled_by` | `string` | user or `policy:<name>` |
| `labeled_at` | `timestamp[us, UTC]` | |

Latest-row-wins per episode (resolved with a window function at query time). Appending a new label supersedes without deleting history.

### 4.6 `snapshots` — frozen training sets

One JSON manifest per snapshot (not Parquet — they're small and self-describing):

```json
{
  "snapshot_id": "snap_2026-07-18_pickplace_v1",
  "created_at": "2026-07-18T21:04:00Z",
  "created_by": "arpit",
  "query": "SELECT ... WHERE q.overall_score > 6 AND ...",
  "scorer_version": "q-v3",
  "embed_model_id": "siglip-so400m@a1b2",
  "episode_ids": ["...", "..."],
  "counts": {"episodes": 4812, "frames": 1_912_330},
  "catalog_schema_version": 1
}
```

The manifest records the query **and** the resolved episode list, so it is reproducible even after the catalog grows.

## 5. Ingestion Pipeline

`forge ingest <uri...> --catalog s3://lab-bucket/forge-catalog [--batch-id ...]`

Per-episode stages, each idempotent and checkpointed:

```
discover → hash → dedup-exact → register → score → embed → dedup-ann → commit
```

1. **Discover.** Expand the URI to candidate episodes using existing Forge readers (any supported format). Cloud URIs via fsspec.
2. **Hash.** `content_hash = xxh3(raw bytes)` streamed, no full-episode load.
3. **Exact-dup check.** If `content_hash` exists in `episodes`, skip (log as `already_ingested`). This makes ingest safely re-runnable — re-pointing at yesterday's folder is a no-op.
4. **Register.** Extract metadata (reusing `forge inspect` internals) → row in `episodes`.
5. **Score.** Run the current scorer version → row in `quality_scores`.
6. **Embed.** Sample frames (default ≈1 Hz effective), run the configured model, pool → row(s) in `embeddings`. Heaviest stage; GPU if available.
7. **ANN dedup.** Query new episode vectors against the existing corpus index (see §7); similarities ≥ threshold → rows in `dedup_edges`.
8. **Commit.** Stage rows in a local buffer; flush as new Parquet part-files per table at batch end (or every K episodes). A `_manifest.json` per batch lists written files, enabling cleanup of partial batches.

**Failure model:** a crashed ingest resumes from its per-episode checkpoint file; already-committed episodes are skipped via the exact-dup check. No partial rows: an episode either fully commits across all tables in its flush, or its staged rows are discarded.

**Ordering note:** stages 5–7 can also run as separate backfill jobs (`forge score`, `forge embed`, `forge dedup`) decoupled from registration — required anyway for re-scoring the corpus when versions bump.

## 6. Query & Curation Layer

**Engine: embedded DuckDB.** It reads partitioned Parquet directly (local or S3), joins across tables, and adds zero operational burden. Forge ships views so users never hand-write the joins:

- `v_latest_quality` — quality rows for the *current* scorer_version
- `v_curation` — latest label per episode
- `v_dup_losers(threshold, policy)` — episodes that lose a dedup pairing under a policy

CLI surface:

```bash
forge query "SELECT task, count(*) FROM episodes GROUP BY task"     # raw SQL
forge curate --where "quality > 6 AND task = 'pick_place'" \
             --dedup 0.97 --dedup-policy keep-higher-quality \
             --label approved --reason "training mix v2"
forge search "picks up the red cup" --top 20                        # text → vector → ANN
```

`forge curate` = query → resolve dedup policy → append `curation_labels` rows. It never touches raw data.

**Python API** mirrors this: `catalog = forge.Catalog("s3://…"); catalog.sql(...)`, plus a thin dataframe-ish helper for common filters. The SQL escape hatch is the power-user API; we do not build an expression engine.

## 7. Vector Search & Dedup Mechanics

- **≤ ~100K episodes:** brute-force cosine in DuckDB (`array_cosine_similarity`) is sub-second on episode-level vectors. Ship this first.
- **Beyond that, or per-frame vectors:** move the `embeddings` table to **Lance** format (columnar like Parquet, built-in IVF/HNSW ANN, embeds like DuckDB). This is a storage-format swap for one table, not an architecture change — a deliberate later upgrade, not a v1 dependency.
- Ingest-time ANN keeps a small in-memory index of corpus vectors (or the Lance index) so daily dedup is O(new × log corpus), not O(new × corpus).
- Dedup threshold default **0.97**, conservative on purpose: teleop corpora contain many *legitimately* similar episodes (same task, same scene). `forge dedup --dry-run --html` renders candidate pairs with thumbnails for human calibration before any labels are applied.

## 8. Versioning & Reproducibility Rules

1. **Scorer/model changes append, never overwrite.** `scorer_version` and `model_id` are part of every derived row.
2. **Snapshots pin versions.** A snapshot records which scorer_version/model_id its query filtered on — re-running the query later against the same versions yields the same set.
3. **Raw data is immutable.** Corrections (e.g. wrong task label) are handled by appending to `curation_labels` or a future `episode_annotations` table, not by editing `episodes`.
4. **Schema evolution is additive.** New columns are nullable; `catalog.json` bumps `schema_version`; Forge refuses to write to a catalog with a newer schema than it understands.

## 9. Phasing

**Phase 1 — Catalog core.** `episodes` + `quality_scores` tables, `forge ingest` (stages 1–5, 8), `forge query`, DuckDB views. Value on day one: a queryable registry with quality over a growing corpus.

**Phase 2 — Embeddings + search.** `embeddings` table, `forge embed` (backfill + ingest stage 6), `forge search`. Model deps behind `pip install forge-robotics[embed]`.

**Phase 3 — Dedup + curation.** `dedup_edges`, `curation_labels`, `forge dedup`, `forge curate` with policies, HTML pair-review.

**Phase 4 — Snapshots + export.** `forge snapshot create/list/materialize`; `materialize` reuses Forge's converters to write the snapshot as LeRobot-v3/RLDS for training.

**Phase 5 — Scale-outs (as needed).** Lance for ANN; per-frame embeddings (`level='frame'`, unlocks segment-level search using PELT segments); diversity metrics in `forge stats`; Ray for embed backfills.

## 10. Open Questions

- **Catalog concurrency:** single-writer assumption for v1 (one ingest job at a time). Do we need a lightweight lock file (`catalog.lock` with lease) before Phase 3, or is convention enough for lab-scale?
- **Per-frame embeddings storage cost:** at 1 Hz, D=1152, a 60 s episode ≈ 270 KB of vectors — fine; at full frame rate it isn't. Confirm 1 Hz default is acceptable for segment search.
- **Task/robot vocabularies:** free-form strings vs. a controlled vocab table? Free-form for v1; revisit when cross-lab sharing matters.
- **Iceberg/Delta instead of raw Parquet dirs?** They add real transactionality but heavier deps. Current position: raw partitioned Parquet + manifest files is enough at lab scale; the layout is Iceberg-compatible if we ever promote it.

## Appendix A — Example end-to-end session

```bash
# Nightly cron
forge ingest s3://lab-bucket/raw/2026-07-18/ --catalog s3://lab-bucket/forge-catalog

# Researcher, next morning
forge query "SELECT task, avg(overall_score) FROM v_latest_quality \
             JOIN episodes USING (episode_id) GROUP BY task"
forge search "regrasps after a failed pick" --top 20
forge dedup --dry-run --html review.html
forge curate --where "quality > 6 AND task='pick_place'" --dedup 0.97 \
             --dedup-policy keep-higher-quality --label approved

# Training time
forge snapshot create pickplace_v2 --where "label = 'approved'"
forge snapshot materialize pickplace_v2 ./train_pickplace_v2 --format lerobot-v3
```
