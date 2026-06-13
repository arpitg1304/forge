# Forge Dedup

Perceptual **episode** deduplication for robotics datasets. Finds near-duplicate
episodes (exact copies, re-encodes, trimmed near-identical takes) and writes a
deduplicated dataset in the same source format. **Tier 0: numpy only, CPU, no
model, no new dependencies** — it reuses the video-quality grayscale path for
decoding/downscaling.

## How it works

1. **Per-frame perceptual hash** on a downscaled grayscale frame:
   - **pHash** (default) — DCT low-frequency block; robust to scaling, blur, compression.
   - **dHash** — horizontal gradient; fast, robust to brightness.
   - **aHash** — pixel-vs-mean; cheapest.
2. **Episode signature** — resample each episode to K evenly-spaced keyframes
   (`--keyframes`, default 16) **per camera** and hash them. Resampling to a fixed
   K normalizes across different episode lengths.
3. **Episode distance** — mean per-keyframe normalized Hamming distance within a
   camera, then the **max across shared cameras** (an episode is a duplicate only
   if *every* shared camera matches). No shared camera → never a duplicate.
4. **Clustering** — union-find over all pairs within `--threshold`, so
   transitively-similar episodes collapse into one cluster.
5. **Keep policy** — one representative per cluster (earliest in dataset order);
   the rest are dropped. Writing is delegated to the filter engine (exclude the
   dropped IDs), so there is a single code path that writes the source format.

## Usage

```bash
# Dry-run: report duplicate clusters
forge dedup ./dataset

# Write the deduplicated dataset
forge dedup ./dataset ./deduped --threshold 0.05

# Faster/cheaper hashing
forge dedup ./dataset ./deduped --method dhash --keyframes 8

# Restrict to specific cameras
forge dedup ./dataset --cameras observation.images.top
```

`--threshold` is normalized Hamming distance in [0, 1]: `0.0` = byte-near-identical
only, `~0.10` (default) = near-duplicates including re-encodes, higher = looser.

### Library

```python
from forge.dedup import DedupEngine, DedupConfig

result = DedupEngine(DedupConfig(method="phash", threshold=0.10)).analyze("./ds")
for cluster in result.clusters:
    print("keep", cluster.representative, "drop", cluster.duplicates)
print(result.num_unique, "unique /", result.total_episodes, "total")
```

## Notes & roadmap

- Clustering is O(N²) on compact signatures (fast for thousands of episodes). For
  much larger corpora, an LSH / BK-tree prefilter is the next optimization.
- **Semantic dedup** (`--method clip`, CLIP-cosine near-duplicates) arrives with
  Tier 2. The CLI rejects `--method clip` until then.

See `VIDEO_QUALITY_PLAN.md` at the repo root for the full design.
