<p align="center">
<pre>
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
</pre>
<h2>⚒ Robotics Data Toolkit &amp; Data Engine ⚒</h2>
<i>Convert between every major robotics format — then turn a pile of datasets into a queryable, searchable, curatable corpus.</i>
<br><br>
<a href="https://pypi.org/project/forge-robotics/"><img alt="PyPI" src="https://img.shields.io/pypi/v/forge-robotics?style=flat-square&color=6c9fff"></a>
<a href="https://colab.research.google.com/github/arpitg1304/forge/blob/main/notebooks/forge_quickstart.ipynb"><img alt="Open in Colab" src="https://img.shields.io/badge/Colab-try%20it-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"></a>
<a href="https://arpitg1304.github.io/forge/"><img alt="Website" src="https://img.shields.io/badge/website-live-6c9fff?style=flat-square"></a>
<a href="https://github.com/arpitg1304/forge"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"></a>
<a href="https://github.com/arpitg1304/forge/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
</p>

Forge is two things that share one core:

- **A format toolkit** — convert, inspect, score, lint, filter, segment, and visualize a single dataset across RLDS, LeRobot, HDF5, MCAP, Zarr, Rosbag, and more.
- **A data engine** — register every episode you collect into an append-only **catalog**, then query it with SQL, search it by natural language, dedup and curate it, and explore it in a visual **Studio**.

Everything works on local paths, `hf://` datasets, and `s3://` / `gs://` buckets.

<table>
<tr><td>

**Toolkit** &nbsp;·&nbsp; [Install](#install) · [Convert](#convert--interop) · [Quality](#score--clean) · [Filter](#score--clean) · [Segment](#understand) · [Tokenize](#understand) · [Visualize](#understand)

</td></tr>
<tr><td>

**Data engine** &nbsp;·&nbsp; [Catalog](#the-data-engine) · [Ingest & query](#1-ingest--query) · [Search](#2-search-semantically) · [Dedup & curate](#3-dedup--curate) · [Studio](#4-forge-studio)

</td></tr>
<tr><td>

**Reference** &nbsp;·&nbsp; [Cloud storage](#cloud-storage-s3--gcs) · [Registry](#dataset-registry) · [Formats](#supported-formats) · [Command cheatsheet](#command-cheatsheet) · [Roadmap](#roadmap)

</td></tr>
</table>

---

## Install

```bash
pip install forge-robotics                  # base CLI + LeRobot v3 read/write
pip install "forge-robotics[all]"           # everything
```

Pick only the extras you need:

| Extra | Adds | Extra | Adds |
|---|---|---|---|
| `[rlds]` | RLDS / Open-X (TensorFlow) | `[s3]` | Read from Amazon S3 (`s3://`) |
| `[lerobot]` | LeRobot v2/v3 (Parquet) | `[gcs]` | Read from Google Cloud (`gs://`) |
| `[mcap]` | MCAP (ROS2 + Foxglove) | `[catalog]` | The catalog (DuckDB) |
| `[hdf5]` | HDF5 (ALOHA, robomimic) | `[embed]` | Semantic search (SigLIP) |
| `[zarr]` | Zarr (Diffusion Policy, UMI) | `[video]` | Video quality + Studio thumbnails |
| `[rosbag]` | ROS1/ROS2 bags | `[rerun]` | Rerun 3D viewer |

> **Heads up:** the published PyPI build is the stable **format toolkit**. The **data-engine** features — catalog, semantic search, dedup/curation, and Forge Studio — are newer than the last release tag. To use those today, install from source:
>
> ```bash
> git clone https://github.com/arpitg1304/forge.git && cd forge
> pip install -e ".[all]"
> ```

<details>
<summary><b>RoboDM (optional)</b></summary>

RoboDM's `.vla` format (up to 70× compression) needs a manual install — the PyPI build has a codec bug:

```bash
git clone https://github.com/BerkeleyAutomation/robodm.git
pip install -e robodm
```
</details>

## Try it in 60 seconds

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arpitg1304/forge/blob/main/notebooks/forge_quickstart.ipynb) — pick a public LeRobot dataset, score every episode on 8 quality metrics, drill into the worst demos. No GPU, no auth, ~60 seconds.

Or locally:

```bash
forge inspect hf://lerobot/pusht                          # what's in it?
forge convert hf://lerobot/pusht ./out --format lerobot-v3  # convert it
forge quality hf://lerobot/pusht                          # score every episode
```

Forge speaks a hub-and-spoke architecture: **any reader → `Episode`/`Frame` → any writer**. Add a reader, get every writer for free — no N×M conversion logic. See [docs/architecture.md](docs/architecture.md).

---

# The format toolkit

Per-dataset operations. Every command takes a local path, a registry id (`droid`), an `hf://` URL, or an `s3://` / `gs://` URI.

## Convert & interop

```bash
forge inspect ./dataset                                    # structure, schema, cameras
forge convert ./rlds_dataset ./out --format lerobot-v3     # convert between formats
forge convert ./data.zarr ./out --format lerobot-v3 --visualize
```

| You have | You want | One command |
|---|---|---|
| MCAP from ROS2 / teleop | LeRobot v3 for HuggingFace | `forge convert teleop.mcap ./out --format lerobot-v3` |
| RLDS from Open-X | LeRobot for finetuning | `forge convert hf://openvla/modified_libero_rlds ./out -f lerobot-v3` |
| HDF5 from ALOHA / robomimic | MCAP for Foxglove playback | `forge convert aloha.hdf5 ./out --format mcap` |
| Zarr from Diffusion Policy | LeRobot v3 | `forge convert pusht.zarr ./out --format lerobot-v3` |

For complex conversions, generate a YAML config: `forge inspect ds/ --generate-config config.yaml`, then `forge convert ds/ out/ --config config.yaml`. See [docs/configuration.md](docs/configuration.md).

## Score & clean

<b>Quality</b> — score each episode 0–10 from proprioception alone (no video needed), on 8 research-backed metrics.

```bash
forge quality ./my_dataset
forge quality ./my_dataset --video --video-level motion    # also score camera streams
```

<details>
<summary>The 8 metrics</summary>

- **Smoothness (LDLJ)** — jerk-based smoothness (Hogan & Sternad, 2009)
- **Dead actions** — zero/constant action detection (Kim et al. "OpenVLA", 2024)
- **Gripper chatter** — rapid open/close transitions (Sakr et al., 2024)
- **Static detection** — idle periods (Liu et al. "SCIZOR", 2025)
- **Timestamp regularity** — dropped frames and frequency jitter
- **Action saturation** — time spent at hardware limits
- **Action entropy** — diversity vs repetitiveness (Belkhale et al. "DemInf", 2025)
- **Path length** — wandering / hesitation in joint space

`--video` adds pixel metrics (blur, exposure, frozen frames) and, at `--video-level motion`, optical-flow motion, camera-vs-scene split, and shot-cut detection. Needs `[video]`. Details: [quality](forge/quality/README.md) · [video quality](forge/quality/video/README.md).
</details>

<b>Lint</b> — check dataset *hygiene* (missing task strings, ambiguous cameras, low-res / single-view, missing action fields) against HuggingFace's LeRobot guidelines. Exits non-zero, so it drops into CI.

```bash
forge lint ./my_dataset --strict          # fail on warnings too
```

<b>Filter</b> — drop bad episodes by quality, flags, or ids → a new dataset.

```bash
forge filter ./my_dataset ./clean --min-quality 6.0 --exclude-flags jerky,mostly_static
```

<b>Dedup</b> — remove near-duplicate episodes *within one dataset* by perceptual hashing of keyframes (numpy only, no model).

```bash
forge dedup ./my_dataset ./deduped --threshold 0.05
```

Details: [lint](forge/lint/README.md) · [filter](forge/filter/README.md) · [dedup](forge/dedup/README.md).

## Understand

<b>Segment</b> — split episodes into phases (reach / grasp / place) via PELT changepoint detection on proprio.

```bash
forge segment hf://lerobot/droid_100 --export segments.json --plot timeline.png
```

<b>Tokenize</b> — turn continuous actions into discrete tokens for VLA training; benchmark strategies on *your* data.

```bash
forge tokenize compare ./my_dataset --sample 20            # recon error / vocab util
forge tokenize write ./my_dataset ./tokenized --strategy openvla-bins
```

Built-ins: `uniform-bins` (RT-1), `openvla-bins`, `quantile-bins`, `mu-law`.

<b>Visualize</b> — three backends: browser (default), matplotlib, and [Rerun](https://rerun.io) (cameras + time-series on one timeline).

```bash
forge visualize pusht                       # web (no install)
forge visualize pusht --backend rerun --segment
```

![Rerun viewer showing camera stream alongside action and state time series](docs/assets/rerun_viz.png)

Details: [segment](forge/segment/README.md) · [tokenize](forge/tokenize/README.md).

---

# The data engine

The toolkit works one dataset at a time. The **catalog** turns Forge into a *system of record*: an append-only set of Parquet tables that registers every episode you ingest and annotates it with quality, embeddings, and curation decisions — all queryable with SQL. It's **zero-server** (Parquet + embedded DuckDB), lives on a **local dir or an `s3://` / `gs://` bucket**, and is readable by pandas / Polars / Spark without Forge.

```
ingest ──▶ query ──▶ embed ──▶ search ──▶ dedup ──▶ curate ──▶ Studio
                                                              (snapshot → soon)
```

```bash
pip install "forge-robotics[catalog]"       # + [embed] for search, [video] for Studio thumbnails
```

## 1. Ingest & query

```bash
forge catalog init ./forge-catalog                         # local dir or cloud bucket
forge ingest ./my_dataset -c ./forge-catalog               # register + quality-score each episode
forge ingest s3://lab-bucket/raw/2026-07-18/ -c ./forge-catalog   # re-runs are a no-op (content hash)

forge query "SELECT task, count(*) FROM episodes GROUP BY task" -c ./forge-catalog
forge catalog stats -c ./forge-catalog
```

Ingestion reuses the same readers as `forge inspect` and the same scorer as `forge quality`, so the catalog stays consistent with the toolkit. Writes go through pyarrow; reads through DuckDB (views: `episodes`, `quality_scores`, `v_latest_quality`).

```python
from forge.catalog import Catalog
from forge.catalog.ingest import ingest

cat = Catalog.init("s3://lab-bucket/forge-catalog")
ingest(["s3://lab-bucket/raw/2026-07-18/"], cat)
df = cat.sql("SELECT robot, avg(overall_score) FROM episodes "
             "JOIN v_latest_quality USING(episode_id) GROUP BY robot").to_pandas()
```

## 2. Search semantically

Embed episodes with [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) (a shared image–text model), then search by natural language — text queries match episode *video*, not just metadata.

```bash
pip install "forge-robotics[embed]"

forge embed -c ./forge-catalog                             # GPU auto: CUDA → Apple MPS → CPU
forge search "picks up the red cup" -c ./forge-catalog --top 10
forge search --like <episode_id> -c ./forge-catalog        # visually-similar episodes
```

Vectors are versioned per model (`siglip-so400m@<ckpt-hash>`) and stored in the catalog. Details: [forge/embed/README.md](forge/embed/README.md).

## 3. Dedup & curate

Find near-duplicate episodes across the whole corpus (cosine over embeddings), then curate a clean, labeled training set. Near-dup pairs are recorded as **facts**; which episode wins is decided by **policy** at curation time.

```bash
forge catalog dedup -c ./forge-catalog --threshold 0.97    # store near-dup pairs (dedup_edges)

forge curate -c ./forge-catalog \
    --where "overall_score > 6 AND task = 'pick_place'" \
    --dedup 0.97 --dedup-policy keep-higher-quality --label approved
```

Curation isn't only SQL-driven — **semantic search and visual review can drive it too**. Save a search result (or Studio's keep/reject decisions) to a selection file and label it, provenance included:

```bash
forge search "regrasp after a failed pick" -c ./forge-catalog --top 20 --save sel.json
forge curate -c ./forge-catalog --from sel.json --label approved     # or --ids e1,e2,e3
```

Curation is an append-log (`curation_labels`, latest-wins) — nothing is deleted. Policies: `keep-higher-quality`, `keep-longer`, `keep-first`.

## 4. Forge Studio

A self-contained, themed HTML app to explore the catalog — **Overview · Corpus** (thumbnails + quality rings) **· Dedup review** (keep/reject pairs) **· Snapshot**. One shareable file, no server; real data and video thumbnails embedded.

```bash
forge studio -c ./forge-catalog -o studio.html && open studio.html
```

Details, storage layout, and commit protocol: [forge/catalog/README.md](forge/catalog/README.md). A ready-to-explore [example catalog](forge/catalog/catalog_example_droid_100/) (droid_100) ships in the repo.

---

# Reference

## Cloud storage (S3 & GCS)

Every command that takes a dataset or catalog path also accepts `s3://` and `gs://` URIs. Cloud datasets are downloaded to a temp dir on first access and cleaned up automatically, so every format works exactly as it does locally.

```bash
pip install "forge-robotics[s3]"            # or [gcs]
forge inspect s3://my-bucket/datasets/run_0413
forge convert gs://lab-data/rosbags ./out --format lerobot-v3
```

**Streaming reads.** For **LeRobot-v3** and **Zarr**, `forge inspect` and `forge ingest` read over the network with **range requests** instead of downloading. Inspecting a 464 MB cloud LeRobot dataset fetches ~3 KB (its `info.json`); **ingesting it — register + quality-score every episode — reads only the proprio parquet columns (~3 MB), never the videos.** `forge inspect --deep` forces a full download when you need per-episode stats; embeddings/video quality still fetch frames.

Auth uses each provider's standard credential chain (AWS env vars / profiles / IAM roles; GCP Application Default Credentials) — Forge never handles credentials itself. Catalogs can be read *and written* in the cloud; per-format conversion **outputs** are local-only for now. Full guide + connectivity troubleshooting: [forge/io/README.md](forge/io/README.md).

## Dataset registry

A curated catalog of 23+ prominent robotics datasets — browse and use by name instead of memorizing URIs. **[Browse online](https://arpitg1304.github.io/forge/registry.html)**

```bash
forge registry list --format rlds --embodiment franka      # filter
forge registry search "franka manipulation"
forge inspect droid                                        # ids work in any command
forge demo                                                 # download + inspect + score a demo
```

Add datasets via [forge/registry/CONTRIBUTING.md](forge/registry/CONTRIBUTING.md).

## Supported formats

| Format | Read | Write | Notes |
|--------|:----:|:-----:|-------|
| RLDS | ✓ | ✓ | Open-X, TensorFlow Datasets |
| LeRobot v2/v3 | ✓ | ✓ | HuggingFace, Parquet + MP4 |
| GR00T | ✓ | – | NVIDIA Isaac, LeRobot v2 + embodiment metadata |
| RoboDM | ✓ | ✓ | Berkeley `.vla`, up to 70× compression (manual install) |
| Zarr | ✓ | – | Diffusion Policy, UMI |
| HDF5 | ✓ | – | robomimic, ACT/ALOHA |
| MCAP | ✓ | ✓ | ROS2 CDR + Foxglove Protobuf, no ROS install required |
| Rosbag | ✓ | – | ROS1 `.bag`, ROS2 SQLite3 |

Which models use which format: [docs/model_formats.md](docs/model_formats.md) · format specs: [docs/format_reference.md](docs/format_reference.md).

## Command cheatsheet

**Toolkit** (per dataset): `inspect` · `convert` · `quality` · `lint` · `filter` · `dedup` · `segment` · `tokenize` · `visualize` · `stats` · `export-video`

**Data engine** (per catalog): `catalog init` · `ingest` · `query` · `catalog stats` · `embed` · `search` · `catalog dedup` · `curate` · `studio`

**Discovery**: `hub` · `local` · `registry` · `demo` · `formats` · `version`

Full reference: [docs/cli.md](docs/cli.md). Run `forge <command> --help` for any command.

## Roadmap

- [x] Dataset registry — curated catalog of 23+ datasets with CLI + HTML browser
- [x] MCAP first-class support — read + write, ROS2 CDR + Foxglove, no ROS install
- [x] Episode filtering, quality scoring, linting, segmentation, tokenization
- [x] Cloud storage — `s3://` / `gs://` on every command
- [x] The catalog — ingest, SQL query, semantic search, dedup, curation, Studio
- [ ] **Snapshots + export** — freeze a curated selection → LeRobot/RLDS for training
- [~] **Streaming reads** — `forge inspect` + `forge ingest` stream LeRobot-v3 / Zarr (metadata + proprio via range reads, no video download); frame/embedding streaming next
- [ ] **Dataset merging & splitting** — combine datasets; stratified train/val/test
- [ ] **Depth / point-cloud support** · **GR00T writer** · **distributed conversion**

## Development

```bash
make venv && source .venv/bin/activate
make install-dev
make test
```

Contributions welcome. See [docs/architecture.md](docs/architecture.md) for the design.

## License

MIT
