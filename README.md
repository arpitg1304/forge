<p align="center">
<pre>
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
</pre>
<h2>⚒ Robotics Data Toolkit ⚒</h2>
<i>Convert, inspect, visualize, score, and discover robotics datasets across every major format.</i>
<br><br>
<a href="https://pypi.org/project/forge-robotics/"><img alt="PyPI" src="https://img.shields.io/pypi/v/forge-robotics?style=flat-square&color=6c9fff"></a>
<a href="https://colab.research.google.com/github/arpitg1304/forge/blob/main/notebooks/forge_quickstart.ipynb"><img alt="Open in Colab" src="https://img.shields.io/badge/Colab-try%20it-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"></a>
<a href="https://arpitg1304.github.io/forge/"><img alt="Website" src="https://img.shields.io/badge/website-live-6c9fff?style=flat-square"></a>
<a href="https://github.com/arpitg1304/forge"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"></a>
<a href="https://github.com/arpitg1304/forge/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
<br><br>
<code>RLDS ═══╗         ╔═══► LeRobot</code><br>
<code>HDF5 ═══╣         ╠═══► MCAP</code><br>
<code>Zarr ═══╬════⚙════╬═══► RoboDM</code><br>
<code>MCAP ═══╝         ╚═══► RLDS</code>
</p>

Convert between robotics dataset formats with one command. Score demonstration quality with research-backed metrics. Lint datasets for hygiene defects before training. Segment episodes into sub-skills with changepoint detection.

| Format | Read | Write | Visualize | Notes |
|--------|:----:|:-----:|:---------:|-------|
| RLDS | ✓ | ✓ | ✓ | Open-X, TensorFlow Datasets |
| LeRobot v2/v3 | ✓ | ✓ | ✓ | HuggingFace, Parquet + MP4 |
| GR00T | ✓ | - | ✓ | NVIDIA Isaac, LeRobot v2 with embodiment metadata |
| RoboDM | ✓ | ✓ | ✓ | Berkeley's .vla format, up to 70x compression* |
| Zarr | ✓ | - | ✓ | Diffusion Policy, UMI |
| HDF5 | ✓ | - | ✓ | robomimic, ACT/ALOHA |
| MCAP | ✓ | ✓ | ✓ | ROS2 CDR + Foxglove Protobuf, no ROS install required |
| Rosbag | ✓ | - | ✓ | ROS1 .bag, ROS2 SQLite3 |

*\*RoboDM requires manual installation from GitHub (see below)*

See [docs/model_formats.md](docs/model_formats.md) for which models (Octo, OpenVLA, ACT, Diffusion Policy, etc.) use which format. See [docs/format_reference.md](docs/format_reference.md) for detailed format specifications.

## Why Forge?

Every robotics lab has their own data format: Open-X uses RLDS, HuggingFace uses LeRobot, Diffusion Policy uses Zarr, robomimic uses HDF5, real-world ROS2 / teleop pipelines use MCAP. Want to train Octo on your ALOHA data? Write a converter. Want to use LeRobot on Open-X datasets? Write another.

Forge uses a hub-and-spoke architecture — one intermediate representation, O(n) format support:

```
Any Reader → Episode/Frame → Any Writer
```

Add a reader, get all writers for free. Add a writer, get all readers for free. No N×M conversion logic. See [docs/architecture.md](docs/architecture.md) for details.

## Try it in 60 seconds (no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arpitg1304/forge/blob/main/notebooks/forge_quickstart.ipynb) &nbsp; **→** Pick a public LeRobot dataset, score every episode on 8 quality metrics, drill into the worst demos. No GPU. No auth. ~60 seconds wall-clock.

## Quick Start

```bash
pip install forge-robotics                  # base CLI + LeRobot v3 read/write
pip install "forge-robotics[mcap]"          # add MCAP read/write
pip install "forge-robotics[rlds,lerobot]"  # pick the formats you need
pip install "forge-robotics[s3]"            # read from Amazon S3 (s3://)
pip install "forge-robotics[gcs]"           # read from Google Cloud Storage (gs://)
pip install "forge-robotics[all]"           # everything
```

That gives you the `forge` CLI:
```bash
forge inspect path/to/dataset
forge convert path/to/dataset ./out --format lerobot-v3
forge visualize path/to/dataset
```

### Develop from source

```bash
git clone https://github.com/arpitg1304/forge.git
cd forge
pip install -e ".[all,dev]"
```

### RoboDM Support (Optional)

RoboDM requires manual installation from GitHub (PyPI version has a codec bug):

```bash
git clone https://github.com/BerkeleyAutomation/robodm.git
pip install -e robodm
```

### Usage

```bash
# See what's in a dataset
forge inspect /path/to/dataset

# Convert it
forge convert /path/to/rlds ./output --format lerobot-v3
forge convert hf://arpitg1304/stack_lego ./stack_lego_rlds --format rlds --workers 4 --visualize
forge convert hf://lerobot/pusht ./pusht_robodm --format robodm
```

Works with HuggingFace Hub too:

```bash
forge inspect hf://lerobot/pusht
forge convert hf://lerobot/pusht ./output --format lerobot-v3
```

### Cloud storage (S3 & GCS)

Every command that takes a dataset path also accepts `s3://` and `gs://` URIs,
in addition to local paths and `hf://` URLs:

```bash
pip install "forge-robotics[s3]"     # or [gcs] for Google Cloud Storage

forge inspect s3://my-bucket/datasets/run_0413
forge convert gs://lab-data/rosbags ./out --format lerobot-v3
forge quality s3://my-bucket/datasets/droid --report report.html
```

Cloud datasets are downloaded to a temporary directory on first access and
cleaned up automatically when the command finishes. This keeps every format
(including video, HDF5, and rosbag, which need random file access) working
exactly as it does locally.

**Authentication** uses each provider's standard credential chain — Forge never
handles credentials itself:

- **S3** — `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars, `~/.aws/config`
  profiles (`AWS_PROFILE`), or the instance/EKS IAM role. See the
  [AWS credentials docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).
- **GCS** — Application Default Credentials: `gcloud auth application-default login`,
  a service-account key via `GOOGLE_APPLICATION_CREDENTIALS`, or the attached
  service account on GCP. See the
  [GCP ADC docs](https://cloud.google.com/docs/authentication/application-default-credentials).

> **Writing** outputs directly to `s3://` / `gs://` is not supported yet — write
> to a local directory and upload it afterwards (`aws s3 cp --recursive`,
> `gcloud storage cp --recursive`).

See [forge/io/README.md](forge/io/README.md) for the Python API and a guide to
diagnosing cloud bucket connectivity issues.

## Common conversions

| You have | You want | One command |
|---|---|---|
| MCAP recording from ROS2 / teleop | LeRobot v3 for HuggingFace | `forge convert teleop.mcap ./out --format lerobot-v3` |
| RLDS from Open-X Embodiment | LeRobot for finetuning | `forge convert hf://openvla/modified_libero_rlds ./out --format lerobot-v3` |
| HDF5 from ALOHA / robomimic | MCAP for Foxglove playback | `forge convert aloha.hdf5 ./out --format mcap` |
| Zarr from Diffusion Policy | LeRobot v3 | `forge convert pusht.zarr ./out --format lerobot-v3` |
| Any supported format | Quality scores per episode | `forge quality ./dataset` |
| Any supported format | Video quality (blur, motion, cuts) | `forge quality ./dataset --video --video-level motion` |
| Any supported format | Lint for hygiene defects | `forge lint ./dataset` |
| Any supported format | Filter out bad demos | `forge filter ./dataset ./clean --min-quality 6.0` |
| Any supported format | Remove near-duplicate episodes | `forge dedup ./dataset ./deduped` |
| Continuous actions | Discrete action tokens for VLA training | `forge tokenize write ./dataset ./tokenized --strategy openvla-bins` |

## Python API

```python
import forge

# Inspect
info = forge.inspect("/path/to/dataset")
print(info.format, info.num_episodes, info.cameras)

# Convert
forge.convert(
    "/path/to/rlds",
    "/path/to/output",
    target_format="lerobot-v3"
)
```

## Quality Metrics

Automated episode-level quality scoring from proprioception data alone — no video processing needed.

```bash
forge quality ./my_dataset
forge quality hf://lerobot/aloha_sim_cube --export report.json
```

Scores each episode 0-10 based on 8 research-backed metrics:

- **Smoothness (LDLJ)** — jerk-based smoothness from motor control literature (Hogan & Sternad, 2009)
- **Dead actions** — zero/constant action detection (Kim et al. "OpenVLA", 2024)
- **Gripper chatter** — rapid open/close transitions (Sakr et al., 2024)
- **Static detection** — idle periods where the robot isn't moving (Liu et al. "SCIZOR", 2025)
- **Timestamp regularity** — dropped frames and frequency jitter
- **Action saturation** — time spent at hardware limits
- **Action entropy** — diversity vs repetitiveness (Belkhale et al. "DemInf", 2025)
- **Path length** — wandering/hesitation in joint space

See [forge/quality/README.md](forge/quality/README.md) for full metric details, paper references, and how to add new metrics.

### Video quality (opt-in)

Proprio scoring is the default fast lane; pass `--video` to also score the camera streams. **Tier 0** (`pixel`) adds sharpness/blur, exposure, frozen-frame, and colorfulness; **Tier 1** (`motion`) adds optical-flow motion magnitude, smoothness, camera-vs-scene split, and shot-cut detection. Add `--workers N` to parallelize.

```bash
forge quality ./my_dataset --video                       # Tier 0 (pixel)
forge quality ./my_dataset --video --video-level motion  # Tier 1 (optical flow)
forge filter ./my_dataset ./clean --min-sharpness 80 --min-motion 0.1 --exclude-flags cut_detected
```

Requires the `[video]` extra (`pip install forge-robotics[video]`). See [forge/quality/video/README.md](forge/quality/video/README.md).

## Episode Filtering

Filter datasets by quality score, flags, or episode IDs. Supports dry-run previews and pre-computed quality reports.

```bash
forge filter ./my_dataset --min-quality 6.0                          # Dry-run preview
forge filter ./my_dataset ./filtered --min-quality 6.0               # Write filtered dataset
forge filter ./my_dataset ./filtered --exclude-flags jerky,mostly_static
forge filter ./my_dataset ./filtered --from-report report.json       # Skip re-analysis
```

See [forge/filter/README.md](forge/filter/README.md) for full details.

## Dataset Linting

Check a dataset against Hugging Face's published [LeRobot recording guidelines](https://huggingface.co/blog/lerobot-datasets) and flag hygiene defects *before* you spend GPU-hours training on it. Where `forge quality` scores trajectory *content* (smoothness, dead actions, chatter), `forge lint` checks *hygiene*: missing or placeholder task strings, ambiguous camera naming, low-resolution or single-view setups, and missing action fields.

```bash
forge lint ./my_dataset
forge lint hf://lerobot/pusht --export lint.json
forge lint ./my_dataset --strict                                     # fail on warnings too, not just errors
```

Runs against the reader's inspected metadata — no video decode, no full episode scan. Exits non-zero on any error (or any warning under `--strict`), so it drops straight into CI.

See [forge/lint/README.md](forge/lint/README.md) for the full check list and thresholds.

## Deduplication

Find and remove near-duplicate episodes (exact copies, re-encodes, near-identical takes) by perceptual hashing of per-camera keyframes — numpy only, no model.

```bash
forge dedup ./my_dataset                                 # Dry-run: report duplicate clusters
forge dedup ./my_dataset ./deduped --threshold 0.05      # Write deduplicated dataset
forge dedup ./my_dataset ./deduped --method dhash        # phash (default) | dhash | ahash
```

See [forge/dedup/README.md](forge/dedup/README.md) for the algorithm and tuning.

## The catalog

Forge is a per-dataset tool by default. The **catalog** turns it into a *system of record*: an append-only set of Parquet tables that registers every episode you ingest and annotates it with quality scores, all queryable with SQL. It's zero-server (just Parquet + embedded DuckDB), works on a local directory or an `s3://` / `gs://` bucket, and is readable by pandas/Polars/Spark without Forge.

```bash
pip install "forge-robotics[catalog]"

# 1. Create a catalog (local dir or cloud bucket)
forge catalog init ./forge-catalog

# 2. Ingest datasets — registers + quality-scores each episode.
#    Re-running is a no-op (episodes are skipped by content hash).
forge ingest ./my_dataset --catalog ./forge-catalog
forge ingest s3://lab-bucket/raw/2026-07-18/ -c ./forge-catalog

# 3. Query with SQL (views: episodes, quality_scores, v_latest_quality)
forge query "SELECT task, count(*) FROM episodes GROUP BY task" -c ./forge-catalog
forge query "SELECT e.language_instruction, q.overall_score
             FROM episodes e JOIN v_latest_quality q USING(episode_id)
             ORDER BY q.overall_score DESC LIMIT 10" -c ./forge-catalog --format json

# 4. Summary stats
forge catalog stats --catalog ./forge-catalog
```

Python API:

```python
from forge.catalog import Catalog
from forge.catalog.ingest import ingest

cat = Catalog.init("s3://lab-bucket/forge-catalog")
ingest(["s3://lab-bucket/raw/2026-07-18/"], cat)
df = cat.sql("SELECT robot, avg(overall_score) FROM episodes "
             "JOIN v_latest_quality USING(episode_id) GROUP BY robot").to_pandas()
```

Ingestion reuses Forge's existing readers (the metadata behind `forge inspect`) and scorer (the engine behind `forge quality`), so the catalog stays consistent with the rest of the toolkit. Writes go through pyarrow; reads through DuckDB; nothing else touches catalog files. See [forge/catalog/README.md](forge/catalog/README.md) for the architecture, storage layout, and commit protocol.

### Semantic search

Embed the episodes in a catalog, then search them by natural language — "find me the regrasp-after-a-failed-pick episodes" instead of scrolling folders. Uses [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) (a shared image–text model), so text queries match episode *video*, not just metadata.

```bash
pip install "forge-robotics[embed]"

# Embed every episode (vision per camera + instruction text). GPU auto-detected
# (CUDA → Apple MPS → CPU); re-running is a no-op.
forge embed --catalog ./forge-catalog

# Search by text …
forge search "picks up the red cup" -c ./forge-catalog --top 10
# … or find visually-similar episodes to one you already like
forge search --like <episode_id> -c ./forge-catalog
```

Vectors are versioned per model (`model_id = siglip-so400m@<ckpt-hash>`) and stored in the same append-only catalog. Brute-force cosine in DuckDB is sub-second at lab scale. See [forge/embed/README.md](forge/embed/README.md) for models, device selection, and reproducibility.

## Dataset Registry

A curated catalog of 23+ prominent robotics datasets — browse, search, and download by name instead of memorizing URIs. **[Browse the registry online](https://arpitg1304.github.io/forge/registry.html)**

```bash
# Browse all datasets
forge registry list

# Open an interactive HTML browser with filtering
forge registry list --html

# Filter by format, embodiment, or tags
forge registry list --format rlds --embodiment franka
forge registry list --tag manipulation --demo

# Get detailed info on a dataset
forge registry info droid

# Search across names, tags, embodiments, and task types
forge registry search "franka manipulation"

# Validate the registry (for contributors)
forge registry validate
```

### Registry ID Resolution

Use dataset IDs directly in any command — no need for full paths or URIs:

```bash
forge inspect droid          # resolves to hf://lerobot/droid
forge quality pusht          # resolves to hf://lerobot/pusht
forge convert droid ./output --format lerobot-v3
```

### Quick Start with `forge demo`

Download a small demo dataset, inspect it, and run quality scoring — all in one command:

```bash
forge demo                   # uses pusht by default
forge demo aloha_sim_cube    # or pick any demo-suitable dataset
```

See [forge/registry/CONTRIBUTING.md](forge/registry/CONTRIBUTING.md) for how to add new datasets to the registry.

## Episode Segmentation

Automatic episode segmentation via PELT changepoint detection on proprioception signals. Splits episodes into contiguous phases (sub-skills, regime changes, idle periods) without video processing.

```bash
forge segment ./my_dataset
forge segment hf://lerobot/droid_100 --export segments.json --plot timeline.png
forge segment ./my_dataset --signal action --penalty bic --cost-model rbf
forge segment ./my_dataset --sample 20
```

Detects where the statistical properties of the proprio signal change abruptly — e.g., transitions between reaching, grasping, and placing phases. Configurable cost models (`rbf`, `l2`, `l1`), penalty methods (`bic`, `aic`, or numeric), and signal selection (`observation.state`, `action`, `qpos`).

See [forge/segment/README.md](forge/segment/README.md) for full details.

## Action Tokenization

Turn continuous action vectors into discrete tokens (and back) for VLA / robot-learning models, which predict discrete action tokens rather than continuous vectors. Proven strategies ship in-box; a comparator benchmarks them on *your* dataset so you don't have to guess.

```bash
forge tokenize list                                                  # registered strategies
forge tokenize compare ./my_dataset --sample 20 --export report.json # benchmark recon error / vocab util
forge tokenize fit ./my_dataset --strategy openvla-bins --out tok.json
forge tokenize write ./my_dataset ./tokenized --strategy openvla-bins  # LeRobot v3 + action_tokens column
```

Built-in strategies: `uniform-bins` (RT-1), `openvla-bins` (OpenVLA), `quantile-bins`, and `mu-law` — all per-step and numpy-only. `write` saves the fitted tokenizer to `meta/action_tokenizer.json` for inference-time detokenization. Add your own strategy with a one-line registry decorator.

See [forge/tokenize/README.md](forge/tokenize/README.md) for full details and the extension API.

## Visualization

Forge ships three visualization backends selectable with `--backend`:

```bash
forge visualize pusht                             # web (default) — browser-based, no install
forge visualize pusht --backend matplotlib        # matplotlib — sliders, comparison mode
forge visualize pusht --backend rerun             # Rerun — cameras + time-series on one timeline
forge visualize pusht --backend rerun --segment   # with PELT phase labels
forge visualize pusht --backend rerun --samples 3 # stream multiple episodes
```

The **Rerun backend** logs each frame's camera images, per-dimension action and state scalars, and segment labels into the [Rerun](https://rerun.io) viewer — all aligned on a shared `frame` timeline.

![Rerun viewer showing camera stream alongside action and state time series](docs/assets/rerun_viz.png)

Install the Rerun extra to use it:

```bash
pip install "forge-robotics[rerun]"
```

## CLI Reference

See [docs/cli.md](docs/cli.md) for the full command reference including:

- `forge inspect` - Dataset inspection and schema analysis
- `forge convert` - Format conversion with camera mapping
- `forge visualize` - Interactive dataset viewer (backends: `web`, `matplotlib`, `rerun`)
- `forge quality` - Episode-level quality scoring ([details](forge/quality/README.md))
- `forge filter` - Quality-based episode filtering ([details](forge/filter/README.md))
- `forge registry` - Browse and search the dataset registry
- `forge demo` - Quick-start with a demo dataset
- `forge segment` - Episode segmentation via changepoint detection ([details](forge/segment/README.md))
- `forge stats` - Compute dataset statistics
- `forge export-video` - Extract camera videos as MP4
- `forge hub` - Search and download from HuggingFace

## Configuration

For complex conversions, use a YAML config:

```bash
forge inspect my_dataset/ --generate-config config.yaml
forge convert my_dataset/ output/ --config config.yaml
```

See [docs/configuration.md](docs/configuration.md) for details.

## Roadmap

Planned features (contributions welcome!):

- [ ] **Dataset merging** - Combine multiple datasets into one (`forge merge ds1/ ds2/ --output combined/`)
- [ ] **Train/val/test splitting** - Split datasets with stratification (`--split 80/10/10`)
- [x] **Dataset registry** - Curated catalog of 23+ robotics datasets with CLI browser and HTML viewer
- [x] **MCAP first-class support** - Read + write, ROS2 CDR + Foxglove Protobuf, no ROS install required
- [ ] **Streaming reads** - Process HuggingFace datasets without full download
- [x] **Episode filtering** - Filter by quality score, flags, or episode IDs (`forge filter --min-quality 6.0`)
- [ ] **Depth/point cloud support** - Preserve depth streams from RLDS/Open-X
- [ ] **GR00T writer** - Write to NVIDIA Isaac GR00T training format (read support complete)
- [ ] **Distributed conversion** - Scale to 100K+ episode datasets across nodes
- [ ] **Conversion verification** - Automated diff between source and converted data

## Development

```bash
make venv && source .venv/bin/activate
make install-dev
make test
```

## License

MIT
