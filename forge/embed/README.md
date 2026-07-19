# `forge.embed`

The embedding engine behind catalog **semantic search** (Phase 2). It maps
episode frames and instructions to vectors; the catalog stores and searches
them. This package is model-only — it knows nothing about the catalog, mirroring
how `forge/quality/` is independent of it.

See [`../catalog/README.md`](../catalog/README.md) for how the catalog uses this,
and [the design doc](../../docs/forge_data_engine_design.md) for the roadmap.

## What it enables

```bash
pip install "forge-robotics[embed]"

forge embed  --catalog ./forge-catalog                 # compute vectors for every episode
forge search "picks up the red cup" -c ./forge-catalog # text → nearest episodes
forge search --like <episode_id>    -c ./forge-catalog # image → similar episodes
```

## Model: SigLIP (shared image–text space)

Default: `google/siglip-so400m-patch14-384` (D=1152), registered as
`siglip-so400m`. SigLIP embeds images and text into the *same* space, so a text
query can be compared directly to episode **vision** vectors — that's what makes
text search work.

Each episode gets:
- one **vision** vector **per camera**, pooled over frames sampled at ~1 Hz
  (`mean` by default; `first-mid-last` optional);
- one **instruction** vector from its `language_instruction` (text tower).

### Reproducible `model_id`

Every vector is tagged with `model_id = "<name>@<checkpoint-hash>"` (e.g.
`siglip-so400m@a1b2c3d4e5f6`), where the hash is derived from the resolved HF
revision. This means:
- vectors from two different checkpoints never mix;
- vectors computed on different machines (a Mac and a CUDA box) are
  interchangeable in the same catalog;
- search **enforces a single `model_id`** — mixing vector spaces is an error.

## Device selection

Automatic: **`cuda → mps → cpu`** (override with `--device`).

- **NVIDIA** → CUDA.
- **Apple Silicon** → MPS (Metal). We set `PYTORCH_ENABLE_MPS_FALLBACK=1` so any
  op not implemented on Metal falls back to CPU instead of erroring, and keep
  fp32 for numerical safety.
- **Otherwise** → CPU (works, just slower).

The same command runs everywhere; the Mac is fine for development and lab-scale
backfills, a GPU box is the fast lane for large corpora.

## Pluggable models

Models register a factory under a short name:

```python
from forge.embed import register_model, get_model

register_model("my-encoder", lambda **kw: MyEncoder(**kw))
model = get_model("my-encoder", device="auto")
vecs = model.embed_images([frame_hwc_uint8, ...])   # (N, model.dim) float32, L2-normalized
```

This is how tests use a torch-free deterministic fake model, and how a
vision-only model like DINOv2 (better for dedup) can be added later without
touching the catalog. An `EmbeddingModel` implements `dim`, `checkpoint_hash`,
`embed_images`, and (for shared-space models) `embed_text`.

## Backfill vs. ingest

- **`forge embed`** (primary) reads a catalog, re-opens each episode's source via
  the existing readers (cloud sources localized through `forge.io`, video decoded
  via `[video]`), embeds, and appends rows. Idempotent — episodes already
  embedded for the `model_id` are skipped, so re-runs are no-ops and re-embedding
  with a new model appends.
- **`forge ingest --embed`** runs the same embedding as an opt-in stage right
  after ingest.

## Not in this phase

Frame-level embeddings / segment search (`level='frame'`), Lance/ANN indexes
(only needed past ~100K episodes — brute-force DuckDB cosine is sub-second at lab
scale), and dedup/curation (Phase 3). The schema and registry already
accommodate them.
