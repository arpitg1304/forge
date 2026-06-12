# Forge Roadmap

## Current Status (v0.1)

### Supported Formats
| Format | Read | Write | Notes |
|--------|------|-------|-------|
| RLDS | ✅ | ✅ | Open X-Embodiment, OpenVLA, Octo |
| LeRobot v2 | ✅ | - | HuggingFace robotics |
| LeRobot v3 | ✅ | ✅ | Target format for all conversions |
| Zarr | ✅ | - | Diffusion Policy datasets |
| HDF5 | ✅ | - | robomimic, ACT/ALOHA datasets |
| MCAP | ✅ | ✅ | ROS2 CDR + Foxglove Protobuf, no ROS install required |
| ROS bags | ✅ | - | ROS1 .bag, ROS2 SQLite3 |

### Core Features
- [x] Format auto-detection
- [x] CLI (`forge inspect`, `convert`, `visualize`, `formats`, `hub`)
- [x] YAML configuration files
- [x] Parallel episode processing (`--workers N`)
- [x] Unified visualizer
- [x] HuggingFace Hub integration (`hf://` URLs)

---

## Completed (v0.1.1)

### ✅ HuggingFace Hub Integration
**Status:** Done

```bash
# Download and inspect
forge inspect hf://lerobot/pusht

# Download and convert
forge convert hf://openvla/modified_libero_rlds output/ --format lerobot-v3

# Search datasets
forge hub "robot manipulation"
forge hub --author lerobot
```

Features:
- [x] `hf://` URL scheme for HuggingFace datasets
- [x] Automatic download and caching (`~/.cache/forge/datasets/`)
- [x] Dataset search/discovery (`forge hub` command)
- [ ] Streaming support for large datasets (future)

---

## Completed (v0.2)

### ✅ RLDS Writer
**Status:** Done

```bash
# Convert LeRobot to RLDS
forge convert lerobot_dataset/ output/ --format rlds

# Round-trip conversion (RLDS → LeRobot v3 → RLDS)
forge convert original_rlds/ temp/ --format lerobot-v3
forge convert temp/ reconverted_rlds/ --format rlds
```

Features:
- [x] Write RLDS format (TFRecord)
- [x] Generate proper episode metadata (dataset_info.json, features.json)
- [x] Support OXE-compatible schema
- [x] Round-trip conversion verified

### ✅ Action Tokenization
**Status:** Done

```bash
forge tokenize compare ./dataset --export report.json
forge tokenize write ./dataset ./tokenized --strategy openvla-bins
```

Features:
- [x] `ActionTokenizer` protocol + `TokenizerRegistry` (one-decorator extension)
- [x] Built-in per-step strategies: `uniform-bins`, `openvla-bins`, `quantile-bins`, `mu-law` (numpy-only)
- [x] `TokenizerComparator` — reconstruction error / tokens-per-step / vocab utilization
- [x] Tokenized LeRobot v3 writer (`action_tokens` column) + saved tokenizer for inference decode
- [ ] FAST (DCT+BPE) and k-means (BeT) strategies — dependency-gated stretch
- [ ] Chunk-granularity dataset writer (per-step only in v1)
- [ ] Parallel write path (sequential in v1)

---

## High Priority (v0.3)

### 1. Validation Tools
**Why:** Ensure converted datasets are correct and complete.

```bash
forge validate output/ --check-frames --check-videos
```

- [ ] Frame count validation
- [ ] Video integrity checks
- [ ] Schema compatibility checks
- [ ] Compare source vs converted statistics

---

## Medium Priority (v0.3)

### 4. Additional Formats

#### DROID Dataset
- Large-scale bimanual manipulation dataset
- Already converted to LeRobot: `IPEC-COMMUNITY/droid_lerobot`

#### BridgeData V2
- 60k+ trajectories from WidowX robots
- Core component of Octo training

#### LIBERO
- Simulation benchmark used by OpenVLA and Pi0
- Pre-converted versions exist on HF Hub

### 5. Streaming & Large Dataset Support
- [ ] Lazy loading for datasets > RAM
- [ ] Chunked processing
- [ ] Resume interrupted conversions

### ✅ Statistics & Analysis
```bash
forge stats dataset/ --plot
forge stats dataset/ --sample 100 --output stats.json
```
- [x] Action/state distributions (min, max, mean, std per dimension)
- [x] Episode length histograms
- [x] Coverage metrics (language, success labels, rewards)
- [x] JSON export for programmatic access
- [x] Matplotlib visualization with --plot flag

---

## Low Priority (v0.4+)

### 7. HuggingFace Upload
```bash
forge upload output/ --repo my-org/my-dataset
```

### 8. Data Augmentation
- [ ] Image augmentations during conversion
- [ ] Action noise injection
- [ ] Temporal subsampling

### 9. Format Converters
- [x] HDF5 reader (robomimic, ACT/ALOHA) ✅ Done
- [ ] MuJoCo dataset support
- [ ] Isaac Gym recordings

---

## VLA Model Compatibility Matrix

| Model | Training Format | Fine-tune Format | Forge Support |
|-------|----------------|------------------|---------------|
| OpenVLA | RLDS | RLDS | ✅ Read |
| Octo | RLDS | RLDS | ✅ Read |
| Pi0/OpenPI | Proprietary | LeRobot v2 | ✅ Read/Write |
| ACT | HDF5 | HDF5 | ✅ Read |
| Diffusion Policy | Zarr | Zarr | ✅ Read |

---

## References

- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) - 1M+ trajectories, RLDS format
- [OpenVLA](https://github.com/openvla/openvla) - Uses RLDS, fine-tunes on RLDS
- [Octo](https://github.com/octo-models/octo) - Uses RLDS from OXE
- [OpenPI (Pi0)](https://github.com/Physical-Intelligence/openpi) - Uses LeRobot for fine-tuning
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robotics standard
