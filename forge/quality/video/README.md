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
with `forge filter` for free, both as flags and as numeric thresholds:

```bash
# Flags
forge filter ./ds ./clean --exclude-flags blurry,frozen_frames

# Numeric Tier 0 thresholds (trigger live video analysis automatically)
forge filter ./ds ./clean --min-sharpness 80 --max-frozen 0.5
forge filter ./ds ./clean --max-overexposed 0.25 --max-underexposed 0.25

# Or read video fields straight from a report
forge quality ./ds --video --export report.json
forge filter ./ds ./clean --from-report report.json --min-sharpness 80
```

`--min-motion` is a Tier 1 metric (see below) and auto-triggers optical-flow analysis.

## Tier 1 — motion (`--video-level motion`)

Classical optical flow (Farnebäck dense flow, opencv) on a small frame
(`motion_downscale`, default 64). Requires the `[video]` extra; without opencv the
metrics raise a helpful install error rather than a bare `ImportError`.

| Metric | What it catches | Flag |
|---|---|---|
| **Motion magnitude** (mean flow, px/frame) | low-signal episodes where nothing moves | `no_motion` |
| **Motion smoothness** (pixel-space LDLJ of the global-motion path) | shaky / jittery video, frame-drop-then-jump | `shaky` |
| **Camera-vs-scene split** (global-affine fit; residual = object motion) | a moving wrist-cam vs an active scene | — |
| **Shot/cut detection** (histogram discontinuity **and** flow spike) | corrupted / concatenated episodes (a robot episode has no cuts) | `cut_detected` |

`no_motion` is decided at the **episode** level (the most-active camera must be
below threshold), so one moving camera keeps the episode "in motion". A cut needs
**both** a luma-histogram drop and a flow spike relative to the episode's median,
so gradual content change isn't mistaken for a cut. Optical flow against a
blank/uniform frame is skipped (meaningless, and dodges occasional decoder black
frames).

```bash
forge quality ./ds --video --video-level motion
forge filter  ./ds ./clean --min-motion 0.1          # drop low-signal episodes
forge filter  ./ds ./clean --exclude-flags cut_detected,shaky
```

## Roadmap

- **Tier 2 (semantic)** — CLIP/SigLIP embeddings, semantic temporal coherence,
  text–image alignment against the language instruction. `--video-level semantic`.

See `VIDEO_QUALITY_PLAN.md` at the repo root for the full design.
