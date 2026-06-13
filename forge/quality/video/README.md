# Forge Video Quality (Tier 0)

Pixel-statistic quality metrics for the **video modality**, computed on a
downscaled grayscale frame. This is **Tier 0** of the video extension: pure numpy,
no model, no GPU, and no new dependencies beyond whatever the dataset reader
already uses to decode frames. The metrics run in a single decode pass, so cost is
dominated by the decode, not the math.

`forge quality` stays **proprio-only by default** — Tier 0 is opt-in behind
`--video`.

## Metrics

| Metric | What it catches | Flag |
|---|---|---|
| **Sharpness** (variance of Laplacian) | motion blur, defocus, compression mush | `blurry` |
| **Exposure** (per-frame luma + pixel clipping) | over/under-exposed clips | `over_exposed` / `under_exposed` |
| **Frozen frame** (consecutive-frame MAE) | encoder stalls, stuck streams, dead-static video | `frozen_frames` |
| **Colorfulness** (Hasler–Süsstrunk) | scene-diversity input (not flagged) | — |

Metrics are computed **per camera**, then rolled up to worst-case episode-level
scalars for one-line filtering.

## Usage

### CLI

```bash
# Proprio + Tier 0 video
forge quality ./my_dataset --video

# Subsample frames for speed (every 4th image-bearing frame)
forge quality ./my_dataset --video --video-stride 4

# Tune the analysis frame size and cap frames per camera
forge quality ./my_dataset --video --video-downscale 96 --video-max-frames 200

# Restrict to specific cameras
forge quality ./my_dataset --video --video-cameras observation.images.top

# Export — video fields land in the same per-episode report rows
forge quality ./my_dataset --video --export report.json
```

### Library

```python
from forge.quality.video import VideoQualityAnalyzer, VideoQualityConfig

analyzer = VideoQualityAnalyzer(VideoQualityConfig(downscale=128, sample_stride=2))
video = analyzer.analyze_episode(episode)   # VideoQuality | None
print(video.min_sharpness, video.flags)
for name, cam in video.per_camera.items():
    print(name, cam.blurry_fraction, cam.frozen_fraction)
```

The `QualityAnalyzer` runs Tier 0 in the **same frame loop** as the proprio pass
(one decode per episode) when given a `video_config`:

```python
from forge.quality import QualityAnalyzer
from forge.quality.video import VideoQualityConfig

eq = QualityAnalyzer(video_config=VideoQualityConfig()).analyze_episode(episode)
eq.video      # VideoQuality, attached to the EpisodeQuality
eq.flags      # video flags are merged in alongside proprio flags
```

## Report integration

`VideoQuality` attaches to `EpisodeQuality.video`. Headline scalars are flattened
into the per-episode report row (`video_min_sharpness`, `video_blurry_fraction`,
`video_frozen_fraction`, …), and video flags merge into `flags` — so they compose
with `forge filter --exclude-flags blurry,frozen_frames` for free. Numeric video
filter criteria (`--max-blur`, `--min-motion`) arrive in a later PR.

## Roadmap

- **Tier 1 (motion)** — optical-flow magnitude/smoothness, camera-vs-scene split,
  shot/cut detection. `--video-level motion`.
- **Tier 2 (semantic)** — CLIP/SigLIP embeddings, semantic temporal coherence,
  text–image alignment against the language instruction. `--video-level semantic`.

See `VIDEO_QUALITY_PLAN.md` at the repo root for the full design.
