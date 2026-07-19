# Example catalog — `droid_100`

A small, checked-in example catalog built by ingesting the
[`lerobot/droid_100`](https://huggingface.co/datasets/lerobot/droid_100) dataset
(100 episodes) into the Forge catalog. It shows exactly what a catalog looks like
on disk and gives you real data to run `forge query` / `forge catalog stats`
against without ingesting anything yourself.

See [`../README.md`](../README.md) for the catalog architecture and
[the design doc](../../../docs/forge_data_engine_design.md) for the bigger
picture.

## What's in here

```
catalog_example_droid_100/
├── catalog.json                                          # schema version, forge version, created_at
├── episodes/ingest_date=.../part-*.parquet               # 100 rows — one per episode (facts)
├── quality_scores/ingest_date=.../part-*.parquet         # 100 rows — quality per episode
├── embeddings/ingest_date=.../part-*.parquet             # 346 rows — SigLIP vectors (vision + text)
└── _batches/<batch_id>/
    ├── manifest-*.json                                   # commit marker
    └── _checkpoint.json                                  # ingest progress marker
```

The whole thing is **~2.2 MB** — the catalog stores derived *facts* (metadata,
quality scores, embedding vectors), never the raw ~460 MB of videos. Episodes
reference their raw data by `source_uri`. The `embeddings` table holds 346
SigLIP vectors (D=1152): a per-camera vision vector for each of the 100 episodes
(300) plus an instruction text vector for the 46 episodes that have one.

Summary of this catalog:

| episodes | frames | hours | fps | avg quality | range |
|---:|---:|---:|---:|---:|---:|
| 100 | 32,212 | 0.60 | 15 | 8.26 | 7.40 – 8.96 |

## Query it (no ingest needed)

```bash
forge catalog stats -c forge/catalog/catalog_example_droid_100

# Top-quality episodes and their language instructions
forge query "SELECT e.language_instruction, round(q.overall_score, 2) AS score
             FROM episodes e JOIN v_latest_quality q USING(episode_id)
             ORDER BY score DESC LIMIT 10" \
    -c forge/catalog/catalog_example_droid_100 --format json

# Frame-count distribution
forge query "SELECT min(num_frames), max(num_frames), round(avg(num_frames)) AS avg
             FROM episodes" -c forge/catalog/catalog_example_droid_100
```

### Semantic search (embeddings included)

This example ships with SigLIP embeddings, so search works out of the box:

```bash
# Image-to-image — find episodes visually similar to one you pick. No model needed.
forge search --like <episode_id> -c forge/catalog/catalog_example_droid_100

# Text search — needs the [embed] extra to encode the query text.
pip install "forge-robotics[embed]"
forge search "close the drawer" -c forge/catalog/catalog_example_droid_100 --top 5
```

Because it's plain Parquet, you can also read it with anything else — no Forge:

```bash
duckdb -c "SELECT source_format, count(*) FROM
           'forge/catalog/catalog_example_droid_100/episodes/*/*.parquet' GROUP BY 1"
# or: python -c "import pandas as pd, glob;
#     print(pd.read_parquet(glob.glob('forge/catalog/catalog_example_droid_100/episodes/*/*.parquet')))"
```

## How this example was built

The source `droid_100` was stored in a local MinIO bucket (S3-compatible), so the
ingest read from `s3://` and wrote the catalog locally:

```bash
# Point s3fs/forge at the local MinIO where droid_100 lives (see forge/io/TESTING_MINIO.md)
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://127.0.0.1:9000

forge catalog init forge/catalog/catalog_example_droid_100
forge ingest s3://forge-datasets/droid_100 \
    --catalog forge/catalog/catalog_example_droid_100
forge embed --catalog forge/catalog/catalog_example_droid_100   # SigLIP on MPS/CUDA/CPU
```

### Reproduce it yourself (no MinIO required)

`droid_100` is public on the Hugging Face Hub, so anyone can rebuild an identical
catalog straight from `hf://`:

```bash
pip install "forge-robotics[catalog,lerobot,hub,embed]"

forge catalog init /tmp/droid_100_catalog
forge ingest hf://lerobot/droid_100 --catalog /tmp/droid_100_catalog
forge embed --catalog /tmp/droid_100_catalog
forge catalog stats -c /tmp/droid_100_catalog
forge search "close the drawer" -c /tmp/droid_100_catalog
```

Re-running ingest or embed is a no-op — episodes are skipped by content hash /
model_id.
