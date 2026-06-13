# Video Modality for Forge Quality + Dedup

## Context

`forge quality` today is **proprio-only by design** — it crunches `Frame.action`,
`Frame.state`, and `Frame.timestamp` into numpy arrays and scores smoothness (LDLJ/SPARC),
dead actions, gripper chatter, static detection, timestamp regularity, saturation, and
entropy ([forge/quality/analyzer.py:185-218](forge/quality/analyzer.py#L185-L218)). The
command's own docstring promises "No video processing — pure numpy number crunching"
([forge/cli.py:1840](forge/cli.py#L1840)). Yet every supported dataset already carries
camera streams — `Frame.images` is a dict of `LazyImage`
([forge/core/models.py:135-173](forge/core/models.py#L135-L173)) and the LeRobot readers
decode per-camera MP4s — and the single most-named web-scale curation operation, perceptual
**dedup**, has no home in Forge at all.

**Goal:** extend the existing `forge quality` verb (not a parallel command) to score the
**video modality**, and add a first-class `forge dedup` verb. Both must compose with the
existing report → `forge filter` contract for free. The proprio-only path stays the
**default fast lane**; video is strictly opt-in behind a `[video]` extra so the no-dependency
promise holds.

Design is **tiered by cost-per-frame**, because web-scale curation lives and dies on decode
throughput. Each tier is independently shippable; tier 0 delivers most of the value at
near-zero marginal cost over the decode itself.

```
forge quality ./ds                                          # proprio only (unchanged default)
forge quality ./ds --video --video-level pixel              # tier 0
forge quality ./ds --video --video-level motion             # tier 0 + 1
forge quality ./ds --video --video-level semantic           # tier 0 + 1 + 2 (GPU, [video] extra)
forge dedup   ./ds ./deduped --method phash                 # exact-ish
forge dedup   ./ds ./deduped --method clip --threshold 0.95 # semantic near-dupes
forge filter  ./ds ./clean --max-blur 80 --min-motion 0.5 \
              --exclude-flags cut_detected,frozen_frames    # video fields drive filter for free
```

The module copies the layout of `forge/quality/` (`config.py` / `models.py` / `metrics.py` /
`analyzer.py` / `__init__.py` / `README.md`) and reuses the existing
`to_dict`/`from_dict`/`to_json`/`from_json` serialization style in
[forge/quality/models.py:90-189](forge/quality/models.py#L90-L189).

---

## The metric menu (organized by cost)

### Tier 0 — pure pixel stats · ~decode speed · CPU · no model
Computed on a downscaled grayscale frame (64–128px) in a **single decode pass**, so they add
almost nothing on top of reading the MP4.

| Metric | What it buys | Curation phrase |
|---|---|---|
| **Sharpness** (variance of Laplacian) | flags motion blur, defocus, compression mush | quality filtering |
| **Exposure / clipping** (saturated-pixel fraction, mean luma, histogram spread) | catches over/underexposed clips; mirrors sibling lighting metric for cross-tool consistency | quality filtering |
| **Frozen-frame / encoder-stall** (per-frame pixel MAE between consecutive frames; near-zero runs) | pixel-space analog of proprio static detection; doubles as a dedup primitive | temporal coherence |
| **Colorfulness + contrast** (Hasler-Süsstrunk) | cheap scene-diversity input | scene diversity |

### Tier 1 — classical motion · fast on downsampled frames · CPU
Optical flow (Farnebäck dense at 64px, or sparse Lucas-Kanade on a grid) is the workhorse;
stays cheap at low resolution.

| Metric | What it buys | Curation phrase |
|---|---|---|
| **Motion magnitude** (mean flow over episode) | an episode where nothing visually moves is low-signal even if proprio looks fine | motion quality |
| **Pixel-space motion smoothness** (temporal derivative of flow field) | image-domain analog of LDLJ jerk; detects shaky/jittery video and frame-drop-then-jump | motion consistency |
| **Camera-vs-scene motion separation** (fit global affine/homography, residual = object motion) | distinguishes "wrist cam moving" from "scene active"; the handheld-shake signal web curation filters on | — |
| **Shot-boundary / cut detection** (sudden flow + histogram discontinuity) | a robot episode should contain no cuts → a detected cut means a corrupted/concatenated episode | temporal coherence, content classification |

### Tier 2 — learned embeddings · GPU · fast with batching · `[video]` extra
Where it overlaps the VLM/CLIP curation line and connects to the sibling embedding tooling.

| Metric | What it buys | Curation phrase |
|---|---|---|
| **CLIP/SigLIP frame embeddings** | reused for embedding dedup, scene-diversity-from-spread, and a tiny LAION-style aesthetic head | deploy VLMs/CLIP at scale for automated filtering |
| **Semantic temporal coherence** (cosine between consecutive embeddings) | sharp drops localize cuts/glitches; smooth drift localizes scene change — more robust than raw pixels | temporal coherence |
| **Text–image alignment** (CLIP score frame vs. episode `language_instruction`) | flags mislabeled/off-task episodes; robot language labels are notoriously noisy | action content |
| **VLM auto-caption / action-verb tagging** (Qwen2-VL class) | annotation; heaviest tier, strictly opt-in | video understanding for annotation |

`Episode.language_instruction` already exists
([forge/core/models.py:283](forge/core/models.py#L283)), so text–image alignment needs no
new plumbing.

### The two that matter most first
- **Perceptual dedup** as a first-class `forge dedup` verb — cheap pHash/dHash for exact-ish
  duplicates, optional CLIP-cosine for semantic near-dupes. Real gap in Forge today.
- **FVD reframed** — true FVD compares a *generated* distribution against a real one via I3D,
  so it doesn't apply to a single dataset and Forge isn't evaluating a generator. Repurpose
  the machinery: compute I3D/VideoMAE features and report **inter-subset Fréchet distance** as
  a domain-gap diagnostic ("how far is robot-A from robot-B," "sim vs real"). FVD-as-divergence,
  not FVD-as-generation-metric. (Tier 3 / future; documented here, not in the first PRs.)

---

## Design

### New module: `forge/quality/video/`
Keeps the proprio quality code untouched and isolates the optional-dependency surface.

```
forge/quality/video/
  __init__.py          # exports VideoQualityAnalyzer, VideoQualityConfig
  decode.py            # single-pass strided frame iterator (decord → PyAV fallback)
  pixel.py             # tier 0: sharpness, exposure, frozen-frame, colorfulness
  motion.py            # tier 1: optical flow, smoothness, cam/scene split, cut detection
  semantic.py          # tier 2: CLIP/SigLIP embeddings, temporal coherence, text-image align
  analyzer.py          # VideoQualityAnalyzer.analyze_episode(episode, level) -> VideoQuality
  config.py            # VideoQualityConfig: thresholds + flags (mirrors quality/config.py)
  models.py            # VideoQuality dataclass + to_dict/from_dict
```

### Decode is the bottleneck, not the metrics
The whole performance story is the decoder. Levers, all in `decode.py`:
- decode at **reduced resolution via the codec** rather than full-res-then-resize;
- **strided** frame sampling (decord excels at this; PyAV fallback);
- **GPU decode (NVDEC)** when present;
- compute **all tier-0 stats in the one decode pass**;
- **batch frames to GPU in fp16** for embeddings;
- parallelize across episodes with the existing `--workers` story.

A `FrameSource` abstraction yields downsampled grayscale frames (for tier 0/1) and, on
demand, full-color batches (for tier 2), so a single decode feeds every tier. Decord is
optional; PyAV (`av>=11.0.0`) is already a declared dependency
([pyproject.toml:63-65](pyproject.toml#L63-L65)) and is the guaranteed fallback — the same
library `forge/video/encoder.py` already uses, with the lazy-import-or-helpful-error pattern.

### `VideoQuality` schema (composes into the existing report)
New optional fields on a `VideoQuality` dataclass, attached to `EpisodeQuality` as a single
nested `video: VideoQuality | None = None`
([forge/quality/models.py:34-84](forge/quality/models.py#L34-L84)) — same pattern as the
existing nested `static: StaticResult` and `timestamps: TimestampResult`. Per camera, since
`Frame.images` is multi-camera:

```python
@dataclass
class VideoQuality:
    per_camera: dict[str, CameraVideoQuality]   # keyed by camera name
    # episode-level rollups (min sharpness across cams, any cut, etc.) for easy filtering
    min_sharpness: float | None = None
    max_frozen_run: int | None = None
    mean_motion: float | None = None
    motion_smoothness: float | None = None      # pixel-space LDLJ analog
    cut_count: int | None = None
    colorfulness: float | None = None
    text_image_alignment: float | None = None   # tier 2 only
    flags: list[str] = field(default_factory=list)
```

`to_dict` flattens the headline scalars (min_sharpness, mean_motion, cut_count, …) alongside
the proprio scalars so the JSON report stays one flat row per episode for filtering.

### New flags (feed `forge filter` for free)
Thresholds live in `VideoQualityConfig` mirroring
[forge/quality/config.py](forge/quality/config.py):

| Flag | Trigger | Config field |
|---|---|---|
| `blurry` | min sharpness < `sharpness_flag` | `sharpness_flag` |
| `over_exposed` / `under_exposed` | clipped fraction > `exposure_flag` | `exposure_flag` |
| `frozen_frames` | longest near-zero run > `frozen_run_flag` | `frozen_run_flag` |
| `no_motion` | mean flow < `motion_low_flag` | `motion_low_flag` |
| `shaky` | motion smoothness < `motion_smoothness_flag` | `motion_smoothness_flag` |
| `cut_detected` | cut_count > 0 | — |
| `mislabeled` | text-image alignment < `alignment_flag` (tier 2) | `alignment_flag` |

### `forge filter` extension
Add video-aware options to [forge/cli.py:2081-2093](forge/cli.py#L2081-L2093) and
`FilterConfig` ([forge/filter/engine.py:21-40](forge/filter/engine.py#L21-L40)):
`--max-blur` (keep if sharpness ≥), `--min-motion`, `--exclude-flags cut_detected,frozen_frames`.
`--exclude-flags` already works on any flag string, so the new flags need **zero filter-engine
changes** beyond making sure they land in the report; `--max-blur`/`--min-motion` are new
numeric criteria in `_evaluate_episode` ([forge/filter/engine.py:199-243](forge/filter/engine.py#L199-L243)).

### `forge dedup` (new verb)
New module `forge/dedup/` (`engine.py` + `hashes.py`), registered like the other verbs in
[forge/cli.py](forge/cli.py). Two methods:
- `--method phash|dhash` (default): perceptual hash on sampled frames, Hamming-distance
  clustering — CPU, no model, no extra deps beyond Pillow/numpy.
- `--method clip`: CLIP-cosine clustering of frame embeddings (reuses `semantic.py`), `[video]`
  extra. `--threshold` sets the cosine/Hamming cutoff.
Writes the de-duplicated dataset in the **same source format** (same pattern as `FilterEngine`),
and emits which episodes were dropped as near-dupes of which kept episode.

### Parallelism
`forge quality`/`filter` are currently single-process
([forge/cli.py:1915-1925](forge/cli.py#L1915-L1925)). Video decode makes per-episode work
heavy enough to justify wiring the existing `ProcessPoolExecutor` pattern from
[forge/convert/converter.py](forge/convert/converter.py) into the quality loop behind a
`--workers` flag. Episodes are embarrassingly parallel; the GPU embedding path batches within
a worker rather than across.

### Dependencies (`pyproject.toml`)
The `[video]` extra exists today with just `av` ([pyproject.toml:63-65](pyproject.toml#L63-L65)).
Extend it (tiers gate themselves at runtime via lazy import + helpful error):
```toml
video = [
    "av>=11.0.0",          # decode fallback + existing encode (tier 0/1)
    "opencv-python-headless>=4.8.0",  # Laplacian, Farnebäck flow, homography (tier 0/1)
    "Pillow>=10.0.0",      # pHash/dHash (dedup)
    "decord>=0.6.0",       # fast strided/GPU decode when available (optional accelerator)
]
video-semantic = [        # tier 2, separate so tier 0/1 stays light
    "torch>=2.0.0",
    "open-clip-torch>=2.20.0",
]
```
Add both to the `all` extra.

---

## Build order (each row is a shippable PR)

| # | PR | Scope | Deps |
|---|---|---|---|
| 1 ✅ | **tier-0 module** | `forge/quality/video/` (`pixel.py` + streaming `analyzer.py`): sharpness, exposure, frozen-frame, colourfulness → `VideoQuality`, wired into `forge quality --video --video-level pixel`, shares the proprio decode pass, fields flattened into the report. **Numpy-only — no new deps** (operates on already-decoded `LazyImage` frames; decode stays the reader's concern). | none |
| 2 ✅ | **filter integration** | Numeric Tier 0 criteria through `forge filter` (`--min-sharpness`, `--max-frozen`, `--max-overexposed`, `--max-underexposed`) + video flags via `--exclude-flags`; both trigger live video analysis automatically or read rollups from `--from-report`. (`--min-sharpness` replaces the plan's contradictory `--max-blur`; `--min-motion` deferred to Tier 1.) | none |
| 3 ✅ | **`forge dedup` (phash)** | new `forge/dedup/` verb: per-camera keyframe pHash/dHash/aHash signatures, union-find clustering, same-format writer (delegated to the filter engine). **Numpy-only — no Pillow** (reuses the Tier 0 grayscale path; DCT via cached cosine basis). `--method clip` deferred to Tier 2. | none |
| 4 | **tier-1 motion** | `motion.py`: optical flow magnitude, pixel-space smoothness, cam/scene split, cut detection; `--video-level motion` | opencv |
| 5 | **`--workers`** | parallelize the quality loop (ProcessPoolExecutor) | — |
| 6 | **tier-2 semantic** | `semantic.py`: CLIP embeddings, semantic temporal coherence, text-image alignment; `forge dedup --method clip`; `--video-level semantic` | torch, open-clip |
| 7 | **(future) FVD-as-divergence** | inter-subset Fréchet distance via I3D/VideoMAE as a domain-gap diagnostic | — |

PR 1 delivers most of the value at near-zero marginal cost and is the cleanest first cut.

---

## Invariants

- **Default stays proprio-only and dependency-free.** `--video` is opt-in; importing the video
  module without the extra raises a helpful install hint, never a bare `ImportError`.
- **One decode pass per episode.** Tiers 0/1 share the same downsampled frame stream; tier 2
  requests full-color batches on top. No metric triggers a second full decode.
- **Video fields are additive in the report.** Existing proprio consumers and the JSON schema
  keep working; `video` is one nested optional block + flattened headline scalars.
- **Scores compose into filtering.** Every new score lands in the per-episode report and is
  reachable from `forge filter` — preserving the "scores compose into the report, report drives
  filtering" contract.
- **Per camera, then rolled up.** Metrics compute per camera (datasets are multi-cam) and
  expose episode-level rollups for one-line filtering.
