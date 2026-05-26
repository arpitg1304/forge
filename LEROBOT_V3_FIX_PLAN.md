# LeRobot v2/v3 fix plan

**Branch:** `arpit_lerobot_v2_v3`
**Resume here:** Phase 1 done. Start Phase 2 ("Writer rewrite — Batch B").

The TL;DR: Forge's v3 writer doesn't produce a valid upstream LeRobot v3 dataset.
13 invariants are violated. The validity test
([tests/test_lerobot_v3_upstream_validity.py](tests/test_lerobot_v3_upstream_validity.py))
codifies all of them — 8 pass, 13 fail today. Make all 21 pass.

---

## Phase 1 — DONE (don't redo)

### Test infrastructure
- ✅ Installed upstream `lerobot==0.5.1` into [.venv](.venv) (~1.5GB w/ torch + torchvision + wandb).
- ✅ Wrote [tests/test_lerobot_v3_upstream_validity.py](tests/test_lerobot_v3_upstream_validity.py) — 21 tests across 5 classes:
  - `TestInfoJson` (5) — required keys, per-feature `fps`, size fields
  - `TestTasksParquet` (3) — `task` as pandas index
  - `TestEpisodesParquet` (8) — full v3 column set
  - `TestAggregateStats` (2) — `meta/stats.json`
  - `TestUpstreamLoader` (3) — gold-standard: `LeRobotDatasetMetadata` + `LeRobotDataset` from upstream actually open Forge's output
- ✅ Module-gated by `pytest.importorskip("lerobot")`, so contributors without the heavy dep skip cleanly.

### Baseline failure surface (run `pytest tests/test_lerobot_v3_upstream_validity.py --no-cov -q`)
**8 passing, 13 failing.** The failures map 1:1 to the audit:

| Failing test | Audit # | Root cause |
|---|---|---|
| `TestInfoJson::test_required_top_level_keys` | #7 | missing `data_files_size_in_mb`, `video_files_size_in_mb` |
| `TestInfoJson::test_per_feature_fps_present_on_non_video` | #8 | `observation.state` / `action` features lack `fps` |
| `TestInfoJson::test_size_fields_are_ints` | #7 | (depends on #7 fix) |
| `TestTasksParquet::test_tasks_index_is_named_task` | #10 | tasks.parquet written as flat columns, not pandas-indexed |
| `TestEpisodesParquet::test_has_tasks_list_column` | #9 | episodes.parquet only has `episode_index, length, task_index` |
| `TestEpisodesParquet::test_has_data_chunk_pointers` | #9 | missing `data/chunk_index, data/file_index` |
| `TestEpisodesParquet::test_has_dataset_row_range` | #9 | missing `dataset_from_index, dataset_to_index` |
| `TestEpisodesParquet::test_has_per_camera_video_pointers` | #9 | missing `videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}` |
| `TestEpisodesParquet::test_has_meta_episodes_pointers` | #9 | missing `meta/episodes/chunk_index, meta/episodes/file_index` |
| `TestEpisodesParquet::test_has_flattened_stats` | #9 | missing flattened `stats/{feature}/{stat}` |
| `TestAggregateStats::test_stats_json_exists` | #11 | not written |
| `TestAggregateStats::test_stats_has_features` | #11 | (depends on #11) |
| `TestUpstreamLoader::test_dataset_loads_and_iterates` | gold standard | `KeyError: 'videos/observation.images.top/chunk_index'` from inside upstream `lerobot/datasets/dataset_metadata.py:211` |

`TestUpstreamLoader::test_metadata_loads` happens to pass — the metadata loader is more permissive than the full Dataset iterator.

### Upstream constants captured
Hard requirements (from real `lerobot==0.5.1` install — not docs):

```python
INFO_PATH                       = "meta/info.json"
STATS_PATH                      = "meta/stats.json"          # aggregate, REQUIRED
DEFAULT_TASKS_PATH              = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH           = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_DATA_PATH               = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH              = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_CHUNK_SIZE              = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB    = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB   = 200
CODEBASE_VERSION                = "v3.0"
```

Upstream code that consumes these columns (verified via `inspect.getsource`):
- `lerobot/datasets/dataset_metadata.py:211-212` reads `videos/{vid_key}/chunk_index`, `videos/{vid_key}/file_index` per episode
- `lerobot/datasets/dataset_metadata.py:359-361, 396-397` reads `dataset_from_index, dataset_to_index`
- `lerobot/datasets/dataset_metadata.py:301, 326` reads `tasks.index` (the pandas index, not a column)
- `lerobot/datasets/io_utils.py:write_tasks` does `tasks.index.name = "task"` then `to_parquet`

---

## Phase 2 — TODO tomorrow (writer rewrite — the big batch)

The writer rewrite is the largest chunk. Estimated ~1 day of focused work.
Apply in this order; each step has its own validity test that should flip green.

### Step 2.1 — Per-feature `fps` in info.json (5 min) — fixes #8
In [forge/formats/lerobot_v3/writer.py:278-292](forge/formats/lerobot_v3/writer.py#L278-L292), every non-video feature dict needs `"fps": fps`:
```python
self._features["observation.state"] = {
    "dtype": "float32",
    "shape": list(frame.state.shape),
    "names": None,
    "fps": fps,                          # ← add
}
```
Also update the standard features in `finalize()` ([writer.py:434-458](forge/formats/lerobot_v3/writer.py#L434-L458)).

**Validation:** `pytest tests/test_lerobot_v3_upstream_validity.py::TestInfoJson::test_per_feature_fps_present_on_non_video`

### Step 2.2 — `data_files_size_in_mb` + `video_files_size_in_mb` (5 min) — fixes #7
In [writer.py:524-538](forge/formats/lerobot_v3/writer.py#L524-L538) info dict, add:
```python
"data_files_size_in_mb": self.config.data_files_size_in_mb,    # default 100
"video_files_size_in_mb": self.config.video_files_size_in_mb,  # default 200
```
Add these as `LeRobotV3WriterConfig` fields with defaults `100` and `200`.

**Validation:** `pytest tests/...::test_required_top_level_keys`, `test_size_fields_are_ints`

### Step 2.3 — `tasks.parquet` with `task` as pandas index (15 min) — fixes #10
In [writer.py:586-594](forge/formats/lerobot_v3/writer.py#L586-L594), replace:
```python
tasks_table = pa.Table.from_pylist(self._task_metadata)
pq.write_table(tasks_table, meta_dir / "tasks.parquet")
```
with:
```python
import pandas as pd
df = pd.DataFrame(self._task_metadata).set_index("task")
df.to_parquet(meta_dir / "tasks.parquet")  # preserves index name
```

**Validation:** `pytest tests/...::TestTasksParquet`

### Step 2.4 — Aggregate `meta/stats.json` (1 hr) — fixes #11
After the per-episode loop, compute aggregate stats by combining per-episode means/stds (Welford or pooled-std). Stats shape per feature:
```python
{"observation.state": {"min": [...], "max": [...], "mean": [...], "std": [...], "count": [N]}, ...}
```
Write to `meta/stats.json` via `json.dump`.

To avoid a second pass over data: track running min/max/sum/sumsq/count per feature in `write_episode()`, finalize in `finalize()`.

**Validation:** `pytest tests/...::TestAggregateStats`

### Step 2.5 — episodes.parquet full schema (HALF DAY — biggest fix) — fixes #9
Fully replace [writer.py:546-584](forge/formats/lerobot_v3/writer.py#L546-L584). Per-episode row must contain:

```python
{
    # Existing
    "episode_index": int,
    "length": int,                         # frames in this episode

    # NEW from v2 metadata
    "tasks": [language_instruction],       # list[str], not just task_index!

    # NEW data pointers
    "data/chunk_index": int,
    "data/file_index": int,
    "dataset_from_index": int,             # global row offset where episode starts
    "dataset_to_index": int,               # global row offset where episode ends (exclusive)

    # NEW per-camera video pointers (one set per video key)
    "videos/observation.images.<cam>/chunk_index": int,
    "videos/observation.images.<cam>/file_index": int,
    "videos/observation.images.<cam>/from_timestamp": float,  # seconds
    "videos/observation.images.<cam>/to_timestamp": float,    # seconds

    # NEW meta pointer
    "meta/episodes/chunk_index": int,
    "meta/episodes/file_index": int,

    # NEW flattened stats (one column per feature × stat)
    "stats/observation.state/min": list[float],   # length = state_dim
    "stats/observation.state/max": list[float],
    "stats/observation.state/mean": list[float],
    "stats/observation.state/std": list[float],
    "stats/observation.state/count": list[int],   # [N]
    "stats/action/min": list[float],
    # ... and so on per feature
}
```

To compute these, the writer needs to track during `write_episode`:
- `_global_frame_offset` — running total across all episodes
- `_current_data_chunk_index`, `_current_data_file_index` — where current data file lives
- per-camera `_current_video_chunk_index`, `_current_video_file_index`, `_video_frames_in_current_file[cam]` — for video timestamp computation

**Validation:** `pytest tests/...::TestEpisodesParquet` (8 tests)

### Step 2.6 — Pack multiple episodes per chunk file (HALF DAY) — fixes #12
The current "one episode per chunk dir" pattern works (Forge can read its own output) but is non-idiomatic and produces too many files.

Replace the `_flush_chunk()` per-episode-flush pattern with:
- Estimate buffered parquet size in MB (rows × bytes/row); flush when adding next episode would exceed `data_files_size_in_mb`
- Same for video: flush MP4 when projected size exceeds `video_files_size_in_mb`
- Episodes within a chunk file get sequential `dataset_from_index`/`to_index`
- Update `chunks_size` semantics in info.json — it's max files per chunk dir, not episodes

**Validation:** `pytest tests/...::TestUpstreamLoader::test_dataset_loads_and_iterates` (the gold standard) should now pass.

---

## Phase 3 — Reader fixes (do AFTER writer is right)

### Step 3.1 — v3 reader: handle multi-episode chunks (4 hrs) — fixes #5, #6
[forge/formats/lerobot_v3/reader.py:1097](forge/formats/lerobot_v3/reader.py#L1097) — `read_episode` only matches when `pq_file.stem == episode_id`, which never matches a packed chunk.

New logic:
1. Read `meta/episodes/.../*.parquet` to get the episode index → (data_chunk, data_file, from_index, to_index) mapping
2. For `read_episode(path, episode_id)`: look up the row, open the data file, slice `[from_index:to_index]`, build Episode
3. For `read_episodes(path)`: iterate all episode metadata rows and yield Episodes lazily

**Validation:** add a new test class `TestForgeV3ReadsForgeV3Output` that writes via `LeRobotV3Writer`, reads back via `LeRobotV3Reader`, asserts state/action equality.

### Step 3.2 — v2 reader: surface tasks (30 min) — fixes #1
[forge/formats/lerobot_v2/reader.py:340-441](forge/formats/lerobot_v2/reader.py#L340-L441) never reads `meta/tasks.jsonl` or `meta/episodes.jsonl`.

Add helpers:
```python
def _load_tasks(path): return {row["task_index"]: row["task"] for row in jsonl_iter(path / "meta/tasks.jsonl")}
def _load_episodes_meta(path): return {row["episode_index"]: row for row in jsonl_iter(path / "meta/episodes.jsonl")}
```
In `_load_episode`, look up the episode's task_index → task string and set `Episode.language_instruction`.

**Validation:** `pytest tests/test_readers.py` (existing). Maybe add a new test that reads a known v2 dataset and asserts `language_instruction` is non-None.

### Step 3.3 — v2 reader: stats parsing + v2.0/v2.1 distinction (2 hrs) — fixes #3, #9
- [reader.py:127](forge/formats/lerobot_v2/reader.py#L127) `detect_version`: check if `meta/episodes_stats.jsonl` exists → `"2.1"`, else `"2.0"`.
- New helper to read either `meta/stats.json` (v2.0) or `meta/episodes_stats.jsonl` (v2.1) and surface in `DatasetInfo.metadata`.

---

## Phase 4 — Verification before merge

```bash
# All upstream-validity tests must pass
pytest tests/test_lerobot_v3_upstream_validity.py -v

# Forge-internal round-trip (write → read → assert state/action equal)
pytest tests/test_lerobot_v3_writer.py -v
pytest tests/test_readers.py::TestLeRobotReader -v   # if exists; else add

# No regressions
pytest tests/test_mcap_*.py -q                        # 101 mcap tests
pytest --no-cov                                       # full suite

# Real-world smoke
forge convert hf://lerobot/pusht /tmp/pusht_v3 --format lerobot-v3
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='forge/pusht', root='/tmp/pusht_v3')
print(len(ds), 'frames loadable upstream')
"
```

If the last `LeRobotDataset` line works on a real Forge-converted dataset, the v3 writer is upstream-valid.

---

## Things to commit before resuming

The validity test file is currently untracked:
```
?? tests/test_lerobot_v3_upstream_validity.py
```

Recommend commit it now (even with 13 failing) so it's the literal spec for the fix work. Use `pytest -m "not upstream_validity"` or skip the file in CI temporarily until Phase 2 lands.

```bash
git add tests/test_lerobot_v3_upstream_validity.py LEROBOT_V3_FIX_PLAN.md
git commit -m "lerobot v3: capture upstream-validity gaps as failing tests + fix plan"
```

---

## Estimates

| Phase | Effort | Effect |
|---|---|---|
| 2.1 + 2.2 + 2.3 | ~30 min | 5 of 13 tests flip green |
| 2.4 (stats.json) | ~1 hr | 7 of 13 green |
| 2.5 (episodes.parquet) | half day | 12 of 13 green |
| 2.6 (multi-ep chunks) | half day | 13 of 13 green ← gold standard |
| 3.1 (v3 reader fix) | 4 hrs | Forge can read its own + upstream output |
| 3.2 + 3.3 (v2 reader) | 3 hrs | language_instruction preserved end-to-end |
| **Total** | **~2.5 days** | Bidirectional v3 compatibility with upstream |

## Notes for tomorrow's first 5 minutes

1. `cd /Users/agog13/developer/forge && git status` — confirm on `arpit_lerobot_v2_v3`
2. `source .venv/bin/activate` — `lerobot==0.5.1` already installed
3. `pytest tests/test_lerobot_v3_upstream_validity.py --no-cov -q` — should still show 8 passing, 13 failing
4. Open [forge/formats/lerobot_v3/writer.py](forge/formats/lerobot_v3/writer.py) and start Step 2.1
5. After each step, re-run the validity test and watch failures flip to green
