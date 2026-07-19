# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Catalog dedup, curation + Forge Studio (Phase 3).** Find near-duplicate
  episodes from the embeddings and curate a clean, labeled training set.

  ```bash
  forge catalog dedup -c ./forge-catalog --threshold 0.97
  forge curate -c ./forge-catalog --where "overall_score > 6" \
      --dedup 0.97 --dedup-policy keep-higher-quality --label approved
  forge studio -c ./forge-catalog -o studio.html
  ```

  - Two new tables (bumps catalog `SCHEMA_VERSION` 2 → 3, additive):
    `dedup_edges` (near-dup pairs as *facts* — similarity, not verdicts) and
    `curation_labels` (an append-log of approve/reject/hold decisions,
    latest-row-wins).
  - `forge catalog dedup` computes near-dup pairs (cosine over episode
    embeddings, max over shared cameras; idempotent). `forge curate` applies a
    WHERE filter + a dedup **policy** (`keep-higher-quality` / `keep-longer` /
    `keep-first`) and labels survivors approved, dedup losers rejected.
  - DuckDB views/macros: `v_curation` (latest label per episode) and
    `v_dup_losers(threshold, policy)`.
  - **`forge studio`** generates a self-contained, themed HTML app (Overview ·
    Corpus · Dedup review · Snapshot) from real catalog data and embedded video
    thumbnails — one shareable file, no server.
  - Reuses the Phase 2 vectors, the readers for thumbnails, and the catalog
    writer/commit machinery. The per-dataset `forge dedup` (perceptual-hash) is
    unchanged; catalog dedup lives under `forge catalog dedup`.
  - See [forge/catalog/README.md](forge/catalog/README.md).

- **Catalog embeddings + semantic search (Phase 2).** Embed the episodes in a
  catalog and search them by natural language.

  ```bash
  pip install "forge-robotics[embed]"
  forge embed --catalog ./forge-catalog
  forge search "picks up the red cup" -c ./forge-catalog --top 10
  forge search --like <episode_id> -c ./forge-catalog
  ```

  - New `embeddings` table (bumps catalog `SCHEMA_VERSION` 1 → 2, additive — v1
    catalogs open unchanged). One row per `(episode_id, model_id, level,
    camera)`: episode-level **vision** vectors per camera + an **instruction**
    (text) vector.
  - New `forge/embed/` engine — an `EmbeddingModel` registry with a **SigLIP**
    (`siglip-so400m`, shared image–text space) implementation, so text queries
    match episode video. Device auto-selects **CUDA → Apple MPS → CPU**.
  - `forge embed` (backfill) and `forge ingest --embed` (opt-in stage); `forge
    search` (text and `--like`) and `Catalog.search(...)`. Brute-force cosine in
    DuckDB — sub-second at lab scale.
  - Vectors are versioned per model (`model_id = <name>@<ckpt-hash>`,
    reproducible across machines); search enforces a single `model_id` (never
    mixes vector spaces).
  - Reuses existing internals — the readers behind `forge inspect` for frames,
    `forge.io` for cloud sources, `[video]` for decode, and the catalog
    writer/commit machinery. New deps behind the `[embed]` extra (torch,
    transformers); the base CLI never imports them.
  - See [forge/embed/README.md](forge/embed/README.md).

- **The catalog (Phase 1) — an append-only, queryable registry of episodes.**
  Turns Forge from a per-dataset tool into a system of record: a set of
  append-only Parquet tables (`episodes`, `quality_scores`) written with pyarrow
  and queried with embedded DuckDB, on a local directory or an `s3://` / `gs://`
  bucket.

  ```bash
  pip install "forge-robotics[catalog]"
  forge catalog init ./forge-catalog
  forge ingest ./my_dataset --catalog ./forge-catalog
  forge query "SELECT task, count(*) FROM episodes GROUP BY task" -c ./forge-catalog
  forge catalog stats --catalog ./forge-catalog
  ```

  - New module `forge/catalog/` with a `Catalog` class as the single entry
    point (`from forge.catalog import Catalog`), plus `forge/catalog/ingest.py`.
  - New CLI commands: `forge catalog init`, `forge catalog stats`,
    `forge ingest`, `forge query` (SQL over `episodes`, `quality_scores`, and a
    `v_latest_quality` view; `--format table|json|csv`).
  - **Ingestion reuses existing internals** — the same format readers behind
    `forge inspect` for metadata and `QualityAnalyzer.analyze_episode` (the
    engine behind `forge quality`) for scoring. No metadata extraction or
    scoring logic was reimplemented, and no existing behavior changed.
  - **Idempotent & crash-safe.** Episodes are keyed by a content hash, so
    re-running an ingest over the same sources is a no-op. Each flush commits
    atomically via a manifest-last protocol; queries read only manifested
    part-files, so a crash never exposes a partial batch.
  - New deps behind the `[catalog]` extra: `duckdb`, `xxhash` (and an explicit
    `pyarrow`). The base CLI never imports them — they load only when a catalog
    command runs.
  - See [forge/catalog/README.md](forge/catalog/README.md) and
    [docs/forge_data_engine_design.md](docs/forge_data_engine_design.md).

- **Cloud storage support (`s3://`, `gs://`).** Every command that accepts a
  dataset path now also accepts Amazon S3 and Google Cloud Storage URIs, in
  addition to local paths and `hf://` URLs. For example:

  ```bash
  forge inspect s3://my-bucket/datasets/run_0413
  forge convert gs://lab-data/rosbags ./out --format lerobot-v3
  forge quality s3://my-bucket/datasets/droid --report report.html
  ```

  - Backed by [fsspec](https://filesystem-spec.readthedocs.io/) (now a core
    dependency), with `s3fs` / `gcsfs` as optional extras:
    `pip install "forge-robotics[s3]"` or `[gcs]`. Passing an `s3://` URI
    without `s3fs` installed fails with the exact `pip install` command to run.
  - All filesystem access is routed through a single utility,
    [`forge.io.paths`](forge/io/paths.py). Remote datasets are downloaded to a
    temporary directory on first access and cleaned up automatically at process
    exit, so every format (including video, HDF5, and rosbag, which need random
    file access) behaves identically to local paths.
  - Authentication uses each provider's default credential chain (AWS env
    vars / profiles / IAM roles; GCP Application Default Credentials). Forge
    never handles credentials itself.

### Changed

- `fsspec` is now a core dependency.

### Known limitations

- **Writing** outputs directly to cloud URIs (e.g. `forge convert … s3://bucket/out`)
  is not supported yet; commands fail fast with a clear message. Write to a
  local directory and upload it afterwards.
- Remote datasets are downloaded in full before processing. True range-read
  streaming (reading parquet/zarr remotely without a full download) is a
  planned follow-up; the fsspec plumbing in `forge.io.paths` is in place for it.
